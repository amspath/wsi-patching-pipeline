"""Pipeline drop auditing.

Records, per stage per slide, which patches each stage kept vs dropped, so a
user can see *what gets thrown out by what* and tune the pipeline. Mirrors the
`profiling` module: a thread-local `AuditRecorder` per producer (one per slide)
is snapshotted and shipped to a `PipelineAuditAggregator` on the pipeline.

Attribution is generic: every stage's output patch coords are recorded, and the
set-difference between consecutive stages is the patches that stage dropped. No
per-filter code needed. Per-patch metadata ("why") is filled in at the single
drop choke point (`CollatedPatchBatch.filter_on_mask`) for dropped patches, and
at the last stage's output for the survivors.

Knobs
-----
A filter whose decision is `metric op threshold`, where the metric does *not*
depend on the threshold, can declare an `AUDIT_KNOBS` tuple of `Knob`s. Then
`dry_run()` runs it *permissively* (threshold set so nothing drops), every patch
gets a metric, and keep/drop can be recomputed for any threshold with no re-run
-- in Python here (`_resolve`) and identically in JS in the report's sliders.

ponytail: per-patch metadata is held in memory for whole slides. Fine for the
1-few-slide tuning runs this targets; if it ever bites on a huge run, cap
`_patch_meta` or subsample. Transport is an in-process `queue.Queue` (threads,
not processes), so objects pass by reference -- no pickling.
"""

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Knob:
    """A stage parameter whose effect can be recomputed without re-running.

    Declare on a Stage as `AUDIT_KNOBS = (Knob(...),)` to get a live slider in
    the audit report. Only valid when `column`'s value is independent of
    `param` -- i.e. changing `param` must not change the metric, only the
    comparison against it.

    Attributes:
        param: attribute name on the stage, e.g. "min_tissue_fraction".
        column: metadata column holding the per-patch metric, e.g. "tissue_fraction".
        op: ">=" or "<=" -- the keep condition is `column op param`.
        permissive: value of `param` at which the stage drops nothing.
        integer: whether `param` is an integer (slider steps by 1).
    """

    param: str
    column: str
    op: str
    permissive: float
    integer: bool = False


def _coord_set(coords: np.ndarray) -> set:
    """(N,2) int array -> set of (x, y) tuples."""
    return {(int(x), int(y)) for x, y in coords}


def _keep(metric: Any, op: str, value: float) -> bool:
    return metric >= value if op == ">=" else metric <= value


class AuditRecorder:
    """Per-slide, thread-local collector. One per producer thread."""

    def __init__(self, enabled: bool, slide_id: str) -> None:
        self.enabled = bool(enabled)
        self.slide_id = slide_id
        # idx -> {"name": str, "coords": (N,2) np.ndarray, "wsi_dims": (W,H)}
        self._stages: Dict[int, Dict[str, Any]] = {}
        # (x, y) -> {col: value}; drops filled at filter_on_mask, survivors at the last stage.
        self._patch_meta: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def record_output(self, idx: int, name: str, coords_list: List[np.ndarray], wsi_dims: Any) -> None:
        """Record the full set of patch coords a stage emitted for this slide."""
        if not self.enabled:
            return
        coords = np.concatenate(coords_list) if coords_list else np.empty((0, 2), dtype=np.int64)
        self._stages[idx] = {"name": name, "coords": coords, "wsi_dims": wsi_dims}

    def record_meta(self, coords: np.ndarray, metas: List[Dict[str, Any]]) -> None:
        """Record per-patch metadata, keyed by coord."""
        if not self.enabled:
            return
        for (x, y), meta in zip(coords, metas):
            self._patch_meta[(int(x), int(y))] = meta

    def snapshot(self) -> Dict[str, Any]:
        return {"slide_id": self.slide_id, "stages": self._stages, "patch_meta": self._patch_meta}


def _py_dropped(ordered: List[Dict[str, Any]]) -> Dict[int, set]:
    """Coords each stage actually dropped in Python, by consecutive-output set-difference."""
    out: Dict[int, set] = {}
    prev_set: Optional[set] = None
    for stage in ordered:
        cur_set = _coord_set(stage["coords"])
        out[stage["idx"]] = (prev_set - cur_set) if prev_set is not None else set()
        prev_set = cur_set
    return out


def _seen(ordered: List[Dict[str, Any]]) -> Dict[int, set]:
    """Coords that actually reached each stage, i.e. the previous stage's output.

    A stage has no verdict on a patch it never saw. That only matters to a report
    that re-orders or removes stages -- it must not read "absent from the drop
    list" as "this stage passed it". Knobbed stages encode the same thing by
    having no metric for the patch.
    """
    out: Dict[int, set] = {}
    prev_set: Optional[set] = None
    for stage in ordered:
        cur_set = _coord_set(stage["coords"])
        out[stage["idx"]] = prev_set if prev_set is not None else cur_set
        prev_set = cur_set
    return out


def _resolve(
    ordered: List[Dict[str, Any]],
    patch_meta: Dict[Tuple[int, int], Dict[str, Any]],
    knobs_by_idx: Dict[int, Dict[str, Any]],
    values: Dict[int, float],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], np.ndarray]:
    """Walk the stages in order and attribute every patch to its first dropper.

    This is the single source of truth for the funnel. It has an exact twin in
    the report's JavaScript -- keep the two in step.

    A knobbed stage judges a patch by `metric op value`; every other stage is
    "fixed" and judged by the coords it actually dropped in Python (the
    set-difference recorded during the run). With no knobs this reduces to plain
    coord-diff attribution, which is what a non-tuning run gets.

    Returns:
        (funnel, dropped, kept) -- see `PipelineAuditAggregator.get_audit`.
    """
    universe = ordered[0]["coords"] if ordered else np.empty((0, 2), dtype=np.int64)
    n = len(universe)
    coord_of = [(int(x), int(y)) for x, y in universe]
    py_dropped = _py_dropped(ordered)

    alive = [True] * n
    funnel: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    n_in = n
    for stage in ordered:
        idx = stage["idx"]
        knob = knobs_by_idx.get(idx)
        drops: List[int] = []
        if knob is not None:
            value = values[idx]
            for i, c in enumerate(alive):
                if not c:
                    continue
                metric = patch_meta.get(coord_of[i], {}).get(knob["column"])
                # No metric means an earlier fixed stage already dropped it, so it
                # is never alive here -- but stay defensive and keep it.
                if metric is None:
                    continue
                if not _keep(metric, knob["op"], value):
                    drops.append(i)
        else:
            fixed = py_dropped[idx]
            for i, c in enumerate(alive):
                if c and coord_of[i] in fixed:
                    drops.append(i)

        for i in drops:
            alive[i] = False
        funnel.append({"stage": stage["name"], "in": n_in, "out": n_in - len(drops), "dropped": len(drops)})
        if drops:
            dropped.append({"stage": stage["name"], "coords": universe[drops]})
        n_in -= len(drops)

    kept = universe[[i for i in range(n) if alive[i]]]
    return funnel, dropped, kept


class PipelineAuditAggregator:
    """Collects per-slide `AuditRecorder` snapshots into a queryable funnel."""

    def __init__(self) -> None:
        self._knobs: List[Dict[str, Any]] = []
        self._tuning = False
        self.reset()

    def reset(self) -> None:
        self._by_slide: Dict[str, Dict[str, Any]] = {}

    def set_knobs(self, knobs: List[Dict[str, Any]], tuning: bool) -> None:
        """Declare the tunable params of the run (set by `dry_run`, post-run)."""
        self._knobs = knobs
        self._tuning = tuning

    def ingest_msg(self, msg: Dict[str, Any]) -> None:
        self._by_slide[msg["slide_id"]] = {"stages": msg["stages"], "patch_meta": msg["patch_meta"]}

    def get_audit(self, values: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
        """Return {"by_slide": {slide_id: {...}}}.

        Args:
            values: knob values keyed by stage index. Defaults to the values the
                user actually configured, so the funnel matches their pipeline.

        Each slide entry has:
          - funnel:      ordered [{"stage", "in", "out", "dropped"}]
          - kept:        (K,2) coords surviving the whole pipeline
          - dropped:     ordered [{"stage", "coords": (D,2)}] per dropping stage
          - candidates:  (N,2) every coord that entered the pipeline
          - patch_meta:  {(x,y): {col: value}}
          - py_dropped:  {stage_idx: set of coords that stage really dropped in Python}
          - seen:        {stage_idx: set of coords that actually reached that stage}
          - stage_list:  ordered [{"idx", "name"}]
          - knobs:       [{"stage_idx", "stage", "param", "column", "op", "init", "integer"}]
          - tuning:      whether knobs were run permissively (sliders are valid)
          - wsi_dims:    (W, H) at patched resolution
        """
        knobs_by_idx = {k["stage_idx"]: k for k in self._knobs}
        vals = {k["stage_idx"]: k["init"] for k in self._knobs}
        if values:
            vals.update(values)

        out: Dict[str, Any] = {"by_slide": {}}
        for slide_id, entry in self._by_slide.items():
            ordered = [{"idx": i, **entry["stages"][i]} for i in sorted(entry["stages"].keys())]
            if not ordered:
                continue
            funnel, dropped, kept = _resolve(ordered, entry["patch_meta"], knobs_by_idx, vals)
            wsi_dims = next((s["wsi_dims"] for s in reversed(ordered) if s["wsi_dims"]), None)
            out["by_slide"][slide_id] = {
                "funnel": funnel,
                "kept": kept,
                "dropped": dropped,
                "candidates": ordered[0]["coords"],
                "patch_meta": entry["patch_meta"],
                "py_dropped": _py_dropped(ordered),
                "seen": _seen(ordered),
                "stage_list": [{"idx": s["idx"], "name": s["name"]} for s in ordered],
                "knobs": self._knobs,
                "tuning": self._tuning,
                "wsi_dims": wsi_dims,
            }
        return out

    def print_audit(self) -> None:
        audit = self.get_audit()
        if not audit["by_slide"]:
            print("No audit data. Run with audit=True or use dry_run().")
            return
        for slide_id, entry in audit["by_slide"].items():
            print(f"\n=== Audit: {slide_id} ===")
            print(f"{'Stage':30} {'In':>10} {'Out':>10} {'Dropped':>10} {'Drop %':>8}")
            for row in entry["funnel"]:
                pct = (100.0 * row["dropped"] / row["in"]) if row["in"] else 0.0
                print(f"{row['stage']:30} {row['in']:>10} {row['out']:>10} {row['dropped']:>10} {pct:>7.1f}%")


_audit_local = threading.local()


def set_current_audit(rec: Optional[AuditRecorder]) -> None:
    _audit_local.rec = rec


def get_current_audit() -> Optional[AuditRecorder]:
    return getattr(_audit_local, "rec", None)
