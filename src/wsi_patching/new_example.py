#!/usr/bin/env python3
"""
Minimal streaming WSI patch pipeline with region-prefetch (cuCIM if available),
per-WSI multiprocessing producers, and a single async WebDataset writer.

Pipeline (as used in main()):
    WSIGrid -> FilterByROI -> Regionize -> RegionReadAndBatch -> DummyTissueClassifier -> PNGEncoder -> WebDatasetWriter

Notes
-----
- This is a barebones, runnable skeleton designed to match the requested API and flow.
- cuCIM is optional at runtime. If unavailable, we fall back to Pillow for small images (level=0 only).
  For real WSIs, install cuCIM and pass real slide paths.
- Multiprocessing model:
    * One producer process per slide (or up to cpu_processes concurrently)
    * One writer process drains a bounded MP queue and writes tar shards continuously
- GPU ops:
    * DummyTissueClassifier simulates a batched GPU step if torch+CUDA are available.
    * It waits for batches (default size 200) emitted by RegionReadAndBatch, then returns the batch.
- WebDataset writer:
    * The writer process owns the only ShardWriter.
    * Samples are written as they arrive; no ordering guarantees.

Profiling
---------
- Enable via Pipeline.run(..., profile=True).
- Each producer process profiles its stages (writer excluded) and sends a summary
  to the parent via a dedicated queue.
- After run(), call Pipeline.get_profile() for a dict or Pipeline.print_profile() for a summary.
"""

from __future__ import annotations

import io
import logging
import multiprocessing as mp
import random
import sys
import time
from dataclasses import dataclass
from multiprocessing.queues import Queue as MPQueue
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
import webdataset as wds
from cucim import CuImage
from PIL import Image


def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(processName)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


Sample = Dict[str, Any]


class PipelineContext(dict):
    """Lightweight, picklable context for cross-stage config & checks."""

    def require_key(self, key: str):
        if key not in self:
            raise KeyError(f"Missing required context key: '{key}'")


class Stage:
    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        raise NotImplementedError

    def then(self, nxt: "Stage") -> "Pipeline":
        return Pipeline([self, nxt])

    def for_slide(self, slide_path: str) -> "Stage":
        return self

    # --- new hooks ---
    def attach_context(self, ctx: PipelineContext) -> None:
        self._ctx = ctx  # type: ignore[attr-defined]

    def export_context(self, ctx: PipelineContext) -> None:
        """Optional: seed/override context keys (called before preflight)."""
        pass

    def validate(self) -> None:
        """Optional: validate config using self._ctx (called once before run)."""
        pass

    @property
    def ctx(self) -> PipelineContext:
        return getattr(self, "_ctx", PipelineContext())


@dataclass
class RectAreaROI:
    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def subdivide(self, max_size: int, tile_size: int, stride: int) -> List["RectAreaROI"]:
        """
        Split this rectangle into smaller rectangles if width or height > max_size.
        Splits are aligned to stride so that tiles stay consistent.
        """
        sub_rois: List[RectAreaROI] = []
        x_end = self.x + self.w
        y_end = self.y + self.h

        for yy in range(self.y, y_end, max_size):
            for xx in range(self.x, x_end, max_size):
                ww = min(max_size, x_end - xx)
                hh = min(max_size, y_end - yy)

                # Align to stride boundaries: ensure splits produce valid tile starts
                # (optional: round ww/hh up so they cover full tiles)
                aligned_w = (ww // stride) * stride
                aligned_h = (hh // stride) * stride
                if aligned_w < tile_size or aligned_h < tile_size:
                    continue

                sub_rois.append(RectAreaROI(xx, yy, aligned_w, aligned_h))

        return sub_rois


class _Profiler:
    """Per-process profiler, but only used for isolated timings (manual calls)."""

    def __init__(self, enabled: bool, slide_id: str):
        self.enabled = bool(enabled)
        self.slide_id = slide_id
        self._stats: Dict[str, Dict[str, float | int]] = {}

    def _ensure(self, stage_name: str):
        if stage_name not in self._stats:
            self._stats[stage_name] = {"wall_time_sec": 0.0, "yields": 0}

    def add_time(self, stage_name: str, dt: float, yielded: bool):
        if not self.enabled:
            return
        self._ensure(stage_name)
        self._stats[stage_name]["wall_time_sec"] = float(self._stats[stage_name]["wall_time_sec"]) + float(dt)
        if yielded:
            self._stats[stage_name]["yields"] = int(self._stats[stage_name]["yields"]) + 1

    def serialize(self) -> Dict[str, Any]:
        return {"slide_id": self.slide_id, "stages": self._stats}


class PipelineProfileAggregator:
    """
    Collects per-process _Profiler summaries and exposes an aggregated view
    with the same shape you had before:
      - get_profile(): {"by_stage": {...}, "by_slide": {...}}
      - print_profile(): pretty console output
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._agg: Dict[str, Any] = {"by_stage": {}, "by_slide": {}}

    def ingest_msg(self, msg: Dict[str, Any]) -> None:
        """Ingest a single producer summary message."""
        slide_id = msg.get("slide_id", "<unknown>")
        stages: Dict[str, Dict[str, float | int]] = msg.get("stages", {})

        self._agg["by_slide"][slide_id] = {}
        for stage_name, stats in stages.items():
            wall = float(stats.get("wall_time_sec", 0.0))
            n = int(stats.get("yields", 0))
            avg = (wall / n * 1000.0) if n > 0 else 0.0

            # Per-slide
            self._agg["by_slide"][slide_id][stage_name] = {"wall_time_sec": wall, "yields": n, "avg_ms_per_yield": avg}

            # Aggregate by stage
            agg = self._agg["by_stage"].setdefault(stage_name, {"wall_time_sec": 0.0, "yields": 0})
            agg["wall_time_sec"] = float(agg["wall_time_sec"]) + wall
            agg["yields"] = int(agg["yields"]) + n

    def get_profile(self) -> Dict[str, Any]:
        out = {"by_stage": {}, "by_slide": self._agg.get("by_slide", {})}
        for stage_name, agg in self._agg.get("by_stage", {}).items():
            wall = float(agg["wall_time_sec"])
            n = int(agg["yields"])
            out["by_stage"][stage_name] = {
                "wall_time_sec": wall,
                "yields": n,
                "avg_ms_per_yield": (wall / n * 1000.0) if n > 0 else 0.0,
            }
        return out

    def print_profile(self) -> None:
        prof = self.get_profile()
        if not prof["by_stage"]:
            print("[profile] No profile data (did you run with profile=True?)")
            return

        def fmt(stats: Dict[str, float | int]) -> str:
            return (
                f"{int(stats['yields']):10d} {float(stats['wall_time_sec']):12.3f} "
                f"{float(stats['avg_ms_per_yield']):16.3f}"
            )

        print("\n=== Pipeline Profile (isolated timings only) ===")
        print(f"{'Stage':30} {'Yields':>10} {'Wall (s)':>12} {'Avg (ms/yield)':>16}")
        for name, stats in sorted(prof["by_stage"].items(), key=lambda kv: kv[1]["wall_time_sec"], reverse=True):
            print(f"{name:30} {fmt(stats)}")

        print("\n--- Per slide breakdown ---")
        for slide_id, stages in prof["by_slide"].items():
            print(f"[{slide_id}]")
            for name, stats in sorted(stages.items(), key=lambda kv: kv[1]["wall_time_sec"], reverse=True):
                print(
                    f"  {name:28} yields={int(stats['yields']):6d} "
                    f"wall={float(stats['wall_time_sec']):8.3f}s avg={float(stats['avg_ms_per_yield']):8.3f}ms"
                )


# Global per-process profiler handle
_CURRENT_PROFILER: Optional["_Profiler"] = None


def _set_current_profiler(p: Optional["_Profiler"]) -> None:
    global _CURRENT_PROFILER
    _CURRENT_PROFILER = p


def _get_current_profiler() -> Optional["_Profiler"]:
    return _CURRENT_PROFILER


def _producer_worker(
    slide_path: str, stage_specs: List[Stage], queue: MPQueue, profile: bool, prof_queue: Optional[MPQueue]
):
    init_logging()
    logging.info("Starting processing.")
    try:
        slide_id = Path(slide_path).stem
        profiler = _Profiler(enabled=profile, slide_id=slide_id)
        _set_current_profiler(profiler)

        # Generic per-slide adaptation: each stage decides if/how it needs to specialize
        local_stages: List[Stage] = [st.for_slide(slide_path) for st in stage_specs]

        # Run per-slide pipeline up to (but excluding) the writer sink.
        pipe = Pipeline(local_stages)
        it = pipe(iter(()))  # sources ignore input

        for out in it:
            if out is None or out.get("_eos"):
                continue
            if "__key__" not in out or "png_bytes" not in out or "json_bytes" not in out:
                continue
            queue.put(out)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.info(f"Error: {e}", file=sys.stderr)
    finally:
        if profile and prof_queue is not None:
            try:
                prof_queue.put({"_profile": True, **profiler.serialize()})
            except Exception:
                logging.info("Failed to send profile message.", file=sys.stderr)
                pass
        _set_current_profiler(None)
        queue.put({"_eos": True})


@dataclass
class Pipeline(Stage):
    stages: List[Stage]
    prof_agg: Optional["PipelineProfileAggregator"] = None  # new

    def __init__(
        self,
        stages: List[Stage],
        prof_agg: Optional["PipelineProfileAggregator"] = None,
        context: Optional[PipelineContext] = None,
    ):
        self.stages = stages
        self.prof_agg = prof_agg
        self._context = context or PipelineContext()

    @property
    def context(self) -> PipelineContext:
        return self._context

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        for s in self.stages:
            it = s(it)
        return it

    def then(self, nxt: Stage) -> "Pipeline":
        return Pipeline(self.stages + [nxt], prof_agg=self.prof_agg)

    def get_profile(self) -> Dict[str, Any]:
        if self.prof_agg is None:
            return {"by_stage": {}, "by_slide": {}}
        return self.prof_agg.get_profile()

    def print_profile(self) -> None:
        if self.prof_agg is None:
            print("[profile] No profile data (did you run with profile=True?)")
            return
        self.prof_agg.print_profile()

    def _ensure_prof_agg(self) -> None:
        if self.prof_agg is None:
            self.prof_agg = PipelineProfileAggregator()

    def run(self, cpu_processes: int = 4, queue_maxsize: int = 4000, profile: bool = False):
        assert isinstance(self.stages[0], WSIGrid)
        assert isinstance(self.stages[-1], WebDatasetWriter)

        writer_stage: WebDatasetWriter = self.stages[-1]  # type: ignore
        producer_stages: List[Stage] = self.stages[:-1]

        grid: WSIGrid = self.stages[0]  # type: ignore
        slides = list(grid.slides)
        if not slides:
            logging.info("[WARN] No slides provided. Nothing to do.")
            return

        # 1) Let stages contribute to context
        for s in self.stages[:-1]:
            s.export_context(self._context)

        # 2) Attach context then run preflight validations
        for s in self.stages[:-1]:
            s.attach_context(self._context)
            s.validate()

        if mp.get_start_method(allow_none=True) != "spawn":
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass

        q: MPQueue = mp.Queue(maxsize=queue_maxsize)
        prof_q: Optional[MPQueue] = mp.Queue() if profile else None

        if profile:
            self._ensure_prof_agg()
            self.prof_agg.reset()  # type: ignore[union-attr]

        writer_proc = mp.Process(target=writer_stage.start_writer, args=(q,), name="webdataset-writer")
        writer_proc.start()

        pending = list(slides)
        active: List[mp.Process] = []

        def spawn_for(path: str):
            p = mp.Process(
                target=_producer_worker,
                args=(path, producer_stages, q, profile, prof_q),
                name=f"producer-{Path(path).stem}",
            )
            p.start()
            return p

        for _ in range(min(cpu_processes, len(pending))):
            slide_path = pending.pop(0)
            active.append(spawn_for(slide_path))

        while active:
            for p in list(active):
                p.join(timeout=0.1)
                if not p.is_alive():
                    active.remove(p)
            while pending and len(active) < cpu_processes:
                slide_path = pending.pop(0)
                active.append(spawn_for(slide_path))

        if profile and prof_q is not None and self.prof_agg is not None:
            received = 0
            expected = len(slides)
            while received < expected:
                try:
                    msg = prof_q.get(timeout=1.0)
                except Exception:
                    break
                if isinstance(msg, dict) and msg.get("_profile"):
                    self.prof_agg.ingest_msg(msg)  # << delegate
                    received += 1

        q.put(None)
        writer_proc.join()


class WSIGrid(Stage):
    """
    Minimal source that yields one 'slide' sample per input slide.
    (MVP simplified: we do not enumerate *all tiles* here; Regionize will do per-ROI tiling.)
    """

    def __init__(self, slides: List[str], tile_size: int, stride: int, level: int = 0):
        self.slides = list(slides)
        self.tile_size = tile_size
        self.stride = stride
        self.level = level

    def export_context(self, ctx: PipelineContext) -> None:
        # Seed/override global grid parameters for other stages to read.
        ctx["tile_size"] = self.tile_size
        ctx["stride"] = self.stride
        ctx["level"] = self.level

    def for_slide(self, slide_path: str) -> "Stage":
        return WSIGrid(slides=[slide_path], tile_size=self.tile_size, stride=self.stride, level=self.level)

    def __call__(self, _it: Iterable[Sample]) -> Iterable[Sample]:
        for path in self.slides:
            wsi_id = Path(path).stem
            W, H = self._get_level_dims(path, self.level)
            logging.info(f"Starting on Slide {wsi_id}")
            yield {"type": "slide", "wsi_id": wsi_id, "wsi_path": path, "dims": (W, H), "meta": {"backend": "cucim"}}

    def _get_level_dims(path: str, level: int) -> Tuple[int, int]:
        img = CuImage(path)
        W, H = img.resolutions["level_dimensions"][level]
        return int(W), int(H)


class FilterByROI(Stage):
    def __init__(self, rois: Dict[str, List[Tuple[int, int, int, int]]]):
        self.rois = {k: [RectAreaROI(*r) for r in v] for k, v in rois.items()}

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        for s in it:
            if s.get("type") != "slide":
                continue
            wsi_id = s["wsi_id"]
            rects = self.rois.get(wsi_id, [])
            W, H = s["dims"]
            slide_rect = RectAreaROI(0, 0, W, H)
            valid = [r for r in rects if self._bbox_intersects(r.as_tuple(), slide_rect.as_tuple())]
            s["rois"] = valid if valid else [slide_rect]
            yield s

    def _bbox_intersects(self, a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


class Regionize(Stage):
    def __init__(self, max_region_size: int = 2048):
        self.max_region_size = max_region_size

    def validate(self) -> None:
        self.ctx.require_key("tile_size")
        self.ctx.require_key("stride")

        if self.max_region_size % self.ctx["tile_size"] != 0:
            raise ValueError(
                f"max_region_size ({self.max_region_size}) must be multiple of tile_size ({self.ctx['tile_size']})"
            )

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        for s in it:
            if s.get("type") != "slide":
                continue

            rois: List[RectAreaROI] = s.get("rois", [])
            if not rois:
                W, H = s["dims"]
                rois = [RectAreaROI(0, 0, W, H)]

            for roi in rois:
                # Subdivide large regions
                sub_rois = roi.subdivide(self.max_region_size, self.ctx["tile_size"], self.ctx["stride"])
                for sub in sub_rois or [roi]:  # if no split was needed
                    tiles = list(
                        self._tiles_in_rect(sub.x, sub.y, sub.w, sub.h, self.ctx["tile_size"], self.ctx["stride"])
                    )
                    if not tiles:
                        continue
                    yield {
                        "type": "region",
                        "wsi_id": s["wsi_id"],
                        "wsi_path": s["wsi_path"],
                        "region": sub.as_tuple(),
                        "tiles": tiles,
                        "meta": s.get("meta", {}),
                    }

    def _tiles_in_rect(
        self, x0: int, y0: int, w: int, h: int, tile_size: int, stride: int
    ) -> Iterator[Tuple[int, int]]:
        x1, y1 = x0 + w, y0 + h
        # Align start to the provided grid (assume tiles already aligned by WSIGrid setting)
        # For simplicity in MVP, start exactly at the rectangle origin (user ensures alignment).
        y = y0
        while y + tile_size <= y1:
            x = x0
            while x + tile_size <= x1:
                yield (x, y)
                x += stride
            y += stride


class RegionReadAndBatch(Stage):
    """
    For each RegionTask:
      - open slide (per-process, no sharing)
      - read the entire region once (cuCIM read_region with num_workers, else PIL crop)
      - slice region into tile patches
      - accumulate into batches of 'batch_size', yield {"batch": [samples,...]}
    """

    def __init__(self, batch_size: int = 200, num_workers: int = 8):
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)

    def validate(self) -> None:
        self.ctx.require_key("tile_size")
        self.ctx.require_key("level")

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        for task in it:
            if task.get("type") != "region":
                continue

            path = task["wsi_path"]
            x0, y0, w, h = task["region"]
            tiles: List[Tuple[int, int]] = task["tiles"]

            # Read region into memory
            region_img = _read_region(path, x0, y0, w, h, self.ctx["level"], num_workers=self.num_workers)

            # Slice into patches
            batch: List[Sample] = []
            for tx, ty in tiles:
                rx, ry = tx - x0, ty - y0
                patch = region_img[ry : ry + self.ctx["tile_size"], rx : rx + self.ctx["tile_size"], :]
                if patch.shape[0] != self.ctx["tile_size"] or patch.shape[1] != self.ctx["tile_size"]:
                    # Skip partial tiles on edges (shouldn't happen if tiles computed within region)
                    continue
                sample: Sample = {
                    "type": "sample",
                    "wsi_id": task["wsi_id"],
                    "coord": (tx, ty),
                    "meta": {"path": path, **task.get("meta", {})},
                    "patch": patch,
                }
                batch.append(sample)

                if len(batch) >= self.batch_size:
                    logging.info(f"Yielding batch from wsi: {task['wsi_id']} size: {len(batch)}")
                    yield {"batch": batch}
                    batch = []

            if batch:
                yield {"batch": batch}


class DummyTissueClassifier(Stage):
    """
    Simulates a batched GPU op. For each batch:
      - Convert to tensor (if torch available)
      - Compute a trivial "tissue score" (mean intensity)
      - Attach score & binary label; return the same batch structure

    device:
      - "cuda" to prefer GPU if available (default)
      - "cpu" to force CPU path
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        for item in it:
            batch: List[Sample] = item["batch"]
            patches = [s["patch"] for s in batch if s.get("patch") is not None]

            # Convert to tensor (B,H,W,C) -> normalize to [0,1]
            arr = np.stack(patches, axis=0)  # uint8
            ten = torch.from_numpy(arr).float() / 255.0  # B,H,W,C
            ten = ten.permute(0, 3, 1, 2)  # B,C,H,W
            if self.device == "cuda":
                ten = ten.cuda(non_blocking=True)
            # Simple "score": mean over (C,H,W)
            scores = ten.mean(dim=(1, 2, 3)).detach().cpu().numpy()
            for s, sc in zip(batch, scores):
                s["tissue_score"] = float(sc)
                s["is_tissue"] = bool(sc > 0.5)

            logging.info(f"Yielding batch from wsi: {batch[0]['wsi_id']} size: {len(batch)}")

            yield {"batch": batch}


class PNGEncoder(Stage):
    """
    Encodes patches to PNG bytes and flattens batches into single-sample items ready for the writer.
    Output items contain: "__key__", "png_bytes", "json_bytes"
    """

    def validate(self) -> None:
        self.ctx.require_key("level")

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        prof = _get_current_profiler()  # may be None if profiling is disabled
        for item in it:
            # Handle batched items
            if isinstance(item, dict) and "batch" in item:
                batch: List[Sample] = item["batch"]
                for s in batch:
                    t0 = time.perf_counter()
                    out = self._encode_one(s)
                    dt = time.perf_counter() - t0
                    if prof is not None and out is not None:
                        # record isolated encode time only (no upstream waiting)
                        prof.add_time("PNGEncoder.isolated", dt, yielded=True)
                    if out is not None:
                        yield out
                continue

            # Single item path
            if isinstance(item, dict) and item.get("type") == "sample":
                t0 = time.perf_counter()
                out = self._encode_one(item)
                dt = time.perf_counter() - t0
                if prof is not None and out is not None:
                    prof.add_time("PNGEncoder.isolated", dt, yielded=True)
                if out is not None:
                    yield out

    def _encode_one(self, s: Sample) -> Optional[Sample]:
        patch = s.get("patch")
        key = f"{s['wsi_id']}-{s['coord'][0]}-{s['coord'][1]}-L{self.ctx['level']}"

        # Encode to PNG
        buf = io.BytesIO()
        Image.fromarray(patch, mode="RGB").save(buf, format="PNG")
        png_bytes = buf.getvalue()

        # Build json sidecar (exclude heavy fields)
        meta = {k: v for k, v in s.items() if k not in ("patch",)}

        return {"__key__": key, "png_bytes": png_bytes, "json_bytes": meta}


class WebDatasetWriter:
    """
    Writer for WebDataset shards.

    Usage:
    - Single process: call with an iterable of samples.
    - Multi process: call `start_writer(queue, outdir, shard_size, shuffle_buffer_size)`.

    Each sample should have:
      - "__key__" (str)
      - "png_bytes" (bytes)
      - "json_bytes" (bytes)
    """

    def __init__(self, outdir: Path = "./output/", shard_size: int = 200, shuffle_buffer_size: int = 500):
        self.outdir = Path(outdir)
        self.shard_size = int(shard_size)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.shard_pattern = str(self.outdir / "shard-%06d.tar")
        self.write_count = 0

    def start_writer(self, queue) -> None:
        """Multi-process mode: consume from queue and write shards."""
        init_logging()
        logging.info("Writer process started.")
        self.outdir.mkdir(parents=True, exist_ok=True)
        sink = wds.ShardWriter(self.shard_pattern, maxcount=self.shard_size, verbose=0)

        buffer: List[Dict[str, Any]] = []
        while True:
            sample = queue.get()
            if sample is None:  # shutdown signal
                logging.info("Received shutdown signal.")
                break
            if sample.get("_eos"):
                continue
            buffer.append(sample)
            if len(buffer) >= self.shuffle_buffer_size:
                self._flush_buffer(buffer, sink)

        if buffer:
            self._flush_buffer(buffer, sink)

        logging.info(f"Writer processed {self.write_count} samples.")
        sink.close()

    def _flush_buffer(self, buffer: List[Dict[str, Any]], sink: wds.ShardWriter) -> None:
        logging.info(f"Flushing buffer of size: {len(buffer)}")
        random.shuffle(buffer)
        for _ in range(min(self.shard_size, len(buffer))):
            s = buffer.pop()
            self.write_count += 1
            sink.write({"__key__": s["__key__"], "png": s["png_bytes"], "json": s["json_bytes"]})
        logging.info(f"Buffer size after flush: {len(buffer)}")


def _read_region(path: str, x: int, y: int, w: int, h: int, level: int, num_workers: int = 8) -> Optional[np.ndarray]:
    """
    Return HxWxC uint8 array for the requested region at the given level.
    Prefer cuCIM; fallback to PIL (level 0 only).
    """
    img = CuImage(path)
    region = img.read_region(location=(x, y), size=(w, h), level=level, num_workers=num_workers)
    # Ensure HxWxC uint8
    arr = np.asarray(region)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Minimal streaming WSI patcher with region-prefetch and WebDataset sink."
    )
    parser.add_argument("--out", type=str, default="./output/train-%06d.tar", help="Shard pattern for WebDataset.")
    parser.add_argument("--procs", type=int, default=4, help="Max producer processes (one per slide concurrently).")
    parser.add_argument("--batch", type=int, default=200, help="Batch size for GPU micro-batching.")
    parser.add_argument("--num-workers", type=int, default=8, help="cuCIM num_workers per region read.")
    parser.add_argument(
        "--profile", action="store_true", help="Enable per-stage profiling for producers.", default=True
    )
    args = parser.parse_args(argv)

    init_logging()

    # Example usage (adjust 'slides' to your real paths)
    slides = [
        "./data/RBIO-GC072-HE-01.tiff",
        "./data/RBIO-GC072-HE-02.tiff",
        "./data/RBIO-GC072-HE-03.tiff",
        "./data/RBIO-GC072-HE-04.tiff",
        "./data/RBIO-GC072-HE-05.tiff",
        "./data/RBIO-GC072-HE-06.tiff",
        "./data/RBIO-GC072-HE-07.tiff",
        "./data/RBIO-GC072-HE-08.tiff",
    ]

    rois = {Path(s).stem: [(0, 0, 4000, 4000)] for s in slides}

    p = (
        WSIGrid(slides=slides, tile_size=256, stride=256, level=0)
        .then(FilterByROI(rois))
        .then(Regionize())
        .then(RegionReadAndBatch(batch_size=args.batch, num_workers=args.num_workers))
        .then(DummyTissueClassifier("cuda"))
        .then(PNGEncoder())
        .then(WebDatasetWriter())
    )

    start_time = time.time()
    logging.info(f"Starting pipeline at time {start_time:.1f}")
    p.run(cpu_processes=args.procs, profile=args.profile)
    logging.info(f"Done in {time.time() - start_time:.1f} seconds.")

    if args.profile:
        # Print a summary on completion if requested
        p.print_profile()
