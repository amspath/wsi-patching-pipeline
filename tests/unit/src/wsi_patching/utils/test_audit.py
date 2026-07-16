import numpy as np
import pytest

from wsi_patching.filtering.remove_edge_tiles import _edge_distance
from wsi_patching.utils.audit import AuditRecorder, Knob, PipelineAuditAggregator, _coord_set


def _grid(n, step=128):
    """n x n grid of top-left coords."""
    return np.array([[x * step, y * step] for y in range(n) for x in range(n)], dtype=np.int64)


def test_recorder_snapshot_disabled_is_empty():
    rec = AuditRecorder(enabled=False, slide_id="s")
    rec.record_output(0, "Extractor", [_grid(2)], (256, 256))
    snap = rec.snapshot()
    assert snap["stages"] == {}


def test_aggregator_funnel_and_attribution():
    all16 = _grid(4)  # 16 patches
    kept4 = all16[[5, 6, 9, 10]]  # arbitrary inner subset

    rec = AuditRecorder(enabled=True, slide_id="slideA")
    rec.record_output(0, "Extractor", [all16], (512, 512))
    rec.record_output(1, "EdgeFilter", [kept4], (512, 512))

    agg = PipelineAuditAggregator()
    agg.ingest_msg({"_audit": True, **rec.snapshot()})
    entry = agg.get_audit()["by_slide"]["slideA"]

    funnel = {r["stage"]: r for r in entry["funnel"]}
    assert funnel["Extractor"]["out"] == 16
    assert funnel["Extractor"]["dropped"] == 0  # source stage drops nothing
    assert funnel["EdgeFilter"]["in"] == 16
    assert funnel["EdgeFilter"]["out"] == 4
    assert funnel["EdgeFilter"]["dropped"] == 12

    assert len(entry["kept"]) == 4
    assert entry["dropped"][0]["stage"] == "EdgeFilter"
    assert len(entry["dropped"][0]["coords"]) == 12


def test_dropped_metadata_join_by_coord():
    all4 = _grid(2)  # 4 patches
    kept2 = all4[[0, 3]]

    rec = AuditRecorder(enabled=True, slide_id="s")
    rec.record_output(0, "Extractor", [all4], (256, 256))
    rec.record_output(1, "Otsu", [kept2], (256, 256))
    # Simulate filter_on_mask capturing the dropped rows' metadata.
    dropped = all4[[1, 2]]
    rec.record_meta(dropped, [{"tissue_fraction": 0.01}, {"tissue_fraction": 0.02}])

    agg = PipelineAuditAggregator()
    agg.ingest_msg({"_audit": True, **rec.snapshot()})
    entry = agg.get_audit()["by_slide"]["s"]

    meta = entry["patch_meta"]
    for x, y in dropped:
        assert (int(x), int(y)) in meta
        assert "tissue_fraction" in meta[(int(x), int(y))]


# ---- knobs -------------------------------------------------------------------


def _permissive_agg():
    """Aggregator for a 4-patch permissive run: nothing dropped, all metrics present."""
    all4 = _grid(2)
    rec = AuditRecorder(enabled=True, slide_id="s")
    rec.record_output(0, "Extractor", [all4], (256, 256))
    rec.record_output(1, "OtsuFilter", [all4], (256, 256))  # permissive: dropped nothing
    rec.record_meta(all4, [{"tissue_fraction": f} for f in (0.1, 0.2, 0.3, 0.4)])

    agg = PipelineAuditAggregator()
    agg.ingest_msg({"_audit": True, **rec.snapshot()})
    agg.set_knobs(
        [
            {
                "stage_idx": 1,
                "stage": "OtsuFilter",
                "param": "min_tissue_fraction",
                "column": "tissue_fraction",
                "op": ">=",
                "init": 0.25,
                "integer": False,
            }
        ],
        tuning=True,
    )
    return agg


@pytest.mark.parametrize(
    "value, expected_kept",
    [(0.0, 4), (0.15, 3), (0.25, 2), (0.35, 1), (0.9, 0)],
)
def test_knob_value_changes_the_funnel(value, expected_kept):
    agg = _permissive_agg()
    entry = agg.get_audit(values={1: value})["by_slide"]["s"]
    assert len(entry["kept"]) == expected_kept
    assert entry["funnel"][1]["out"] == expected_kept


def test_knob_defaults_to_the_users_configured_value():
    entry = _permissive_agg().get_audit()["by_slide"]["s"]
    assert len(entry["kept"]) == 2  # init=0.25 keeps 0.3 and 0.4
    assert entry["tuning"] is True


def test_attribution_goes_to_the_first_stage_that_drops():
    """A patch failing two stages is owned by the earlier one, as in the real chain."""
    all2 = _grid(1, step=128)  # 1 patch
    all2 = np.array([[0, 0], [128, 0]], dtype=np.int64)
    rec = AuditRecorder(enabled=True, slide_id="s")
    rec.record_output(0, "Extractor", [all2], (256, 256))
    rec.record_output(1, "A", [all2], (256, 256))
    rec.record_output(2, "B", [all2], (256, 256))
    rec.record_meta(all2, [{"m1": 0.0, "m2": 0.0}, {"m1": 1.0, "m2": 1.0}])

    agg = PipelineAuditAggregator()
    agg.ingest_msg({"_audit": True, **rec.snapshot()})
    knob = {"integer": False, "op": ">=", "init": 0.5}
    agg.set_knobs(
        [
            {"stage_idx": 1, "stage": "A", "param": "t", "column": "m1", **knob},
            {"stage_idx": 2, "stage": "B", "param": "t", "column": "m2", **knob},
        ],
        tuning=True,
    )
    entry = agg.get_audit()["by_slide"]["s"]
    # Patch (0,0) fails both A and B; A is first, so A owns it and B sees only 1 patch.
    assert [(d["stage"], len(d["coords"])) for d in entry["dropped"]] == [("A", 1)]
    assert entry["funnel"][2] == {"stage": "B", "in": 1, "out": 1, "dropped": 0}


def test_filters_without_knobs_still_work_alongside_sliders():
    """A stage with no AUDIT_KNOBS keeps dropping in Python and stays attributed.

    Its patches never reach the knobbed stage, so they carry no metric for it --
    the slider must not be able to steal or resurrect them at any value.
    """
    all4 = _grid(2)
    after_fixed = all4[[1, 2, 3]]  # a fixed classifier dropped (0,0)

    rec = AuditRecorder(enabled=True, slide_id="s")
    rec.record_output(0, "Extractor", [all4], (256, 256))
    rec.record_output(1, "FixedClassifier", [after_fixed], (256, 256))
    rec.record_output(2, "OtsuFilter", [after_fixed], (256, 256))  # permissive
    rec.record_meta(all4[[0]], [{}])  # dropped by the fixed stage: no Otsu metric
    rec.record_meta(after_fixed, [{"tissue_fraction": f} for f in (0.1, 0.5, 0.9)])

    agg = PipelineAuditAggregator()
    agg.ingest_msg({"_audit": True, **rec.snapshot()})
    agg.set_knobs(
        [
            {
                "stage_idx": 2,
                "stage": "OtsuFilter",
                "param": "min_tissue_fraction",
                "column": "tissue_fraction",
                "op": ">=",
                "init": 0.0,
                "integer": False,
            }
        ],
        tuning=True,
    )

    for v in (0.0, 0.3, 0.7, 1.0):
        entry = agg.get_audit(values={2: v})["by_slide"]["s"]
        by_stage = {d["stage"]: _coord_set(d["coords"]) for d in entry["dropped"]}
        # (0,0) is owned by the fixed stage at every slider position...
        assert by_stage["FixedClassifier"] == {(0, 0)}
        # ...and never reappears among the kept patches.
        assert (0, 0) not in _coord_set(entry["kept"])
        assert entry["funnel"][1] == {"stage": "FixedClassifier", "in": 4, "out": 3, "dropped": 1}


def test_seen_records_what_reached_each_stage_not_what_survived():
    """`seen` is what a stage judged; a report that reorders stages needs it.

    Without it, "absent from the drop list" reads as "this stage passed it" -- a lie
    for a patch the stage never saw.
    """
    all4 = _grid(2)
    after_a = all4[[1, 2, 3]]  # A dropped (0,0)
    after_b = all4[[2, 3]]  # B then dropped (128,0)

    rec = AuditRecorder(enabled=True, slide_id="s")
    rec.record_output(0, "Extractor", [all4], (256, 256))
    rec.record_output(1, "A", [after_a], (256, 256))
    rec.record_output(2, "B", [after_b], (256, 256))

    agg = PipelineAuditAggregator()
    agg.ingest_msg({"_audit": True, **rec.snapshot()})
    seen = agg.get_audit()["by_slide"]["s"]["seen"]

    assert seen[0] == _coord_set(all4)  # the source produced them all
    assert seen[1] == _coord_set(all4)  # A saw everything
    assert seen[2] == _coord_set(after_a)  # B never saw what A dropped
    assert (0, 0) not in seen[2]


def test_knob_is_a_frozen_declaration():
    k = Knob("depth", "edge_distance", ">=", permissive=0.0, integer=True)
    with pytest.raises(Exception):
        k.param = "other"  # type: ignore[misc]


# ---- edge_distance -----------------------------------------------------------


@pytest.mark.parametrize("depth", [0, 1, 2, 3, 4])
def test_edge_distance_reproduces_the_original_keep_condition(depth):
    """`edge_distance >= depth` must equal RemoveEdgeTiles' original four clauses.

    The grid deliberately stops short of the right/bottom border so the test also
    covers tiles that never reach the slide edge.
    """
    ts = 100
    wsi_dims = (1000, 700)
    coords = np.array([[x, y] for x in range(0, 950, ts) for y in range(0, 650, ts)], dtype=np.int64)

    margin = depth * ts
    expected = (
        (coords[:, 0] >= margin)
        & (coords[:, 0] < wsi_dims[0] - margin)
        & (coords[:, 1] >= margin)
        & (coords[:, 1] < wsi_dims[1] - margin)
    )
    assert np.array_equal(_edge_distance(coords, wsi_dims, ts) >= depth, expected)


def test_edge_distance_is_zero_on_the_border_ring():
    ts = 100
    ed = _edge_distance(np.array([[0, 0], [900, 600], [100, 100]]), (1000, 700), ts)
    assert ed[0] == 0 and ed[1] == 0  # corners sit in the outermost ring
    assert ed[2] == 1
