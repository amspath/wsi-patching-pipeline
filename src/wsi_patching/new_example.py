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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import webdataset as wds
from cucim import CuImage
from PIL import Image

from wsi_patching.profiling import PipelineProfileAggregator, Profiler, get_current_profiler, set_current_profiler


def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(processName)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


Box = Tuple[int, int, int, int]  # (x, y, w, h) in level-0 pixels


class ROI:
    """Geometry-agnostic region of interest in level-0 coordinates."""

    def bounds(self) -> Box:
        raise NotImplementedError

    def contains_point(self, x: float, y: float) -> bool:
        """Return True if the (x,y) center lies in ROI. Used by center-in-ROI selection."""
        raise NotImplementedError


@dataclass
class BoxROI(ROI):
    x: int
    y: int
    w: int
    h: int

    def bounds(self) -> Box:
        return (self.x, self.y, self.w, self.h)

    def contains_point(self, x: float, y: float) -> bool:
        return (self.x <= x < self.x + self.w) and (self.y <= y < self.y + self.h)


class ROIProvider:
    """Source of ROIs for a slide."""

    def for_slide(self, slide: Sample) -> List[ROI]:
        raise NotImplementedError


@dataclass
class RectROIProvider(ROIProvider):
    """Compatibility provider using a dict: {wsi_id: [(x,y,w,h), ...]}.
    Raises ValueError if any ROI lies outside the slide bounds."""

    rois: Dict[str, List[Tuple[int, int, int, int]]]

    def for_slide(self, slide: Sample) -> List[ROI]:
        wsi_id = slide["wsi_id"]
        W, H = slide["dims"]
        out: List[ROI] = []
        for tpl in self.rois.get(wsi_id, []):
            x, y, w, h = tpl
            if x < 0 or y < 0 or (x + w) > W or (y + h) > H:
                raise ValueError(f"ROI {tpl} for slide {wsi_id} lies outside slide dimensions {(W, H)}")
            out.append(BoxROI(x, y, w, h))
        return out


class WholeSlideProvider(ROIProvider):
    """Provides a single ROI covering the full slide extent."""

    def for_slide(self, slide: Sample) -> List[ROI]:
        W, H = slide["dims"]
        return [BoxROI(0, 0, int(W), int(H))]


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


def _producer_worker(
    slide_path: str, stage_specs: List[Stage], queue: MPQueue, profile: bool, prof_queue: Optional[MPQueue]
):
    init_logging()
    logging.info("Starting processing.")
    try:
        slide_id = Path(slide_path).stem
        profiler = Profiler(enabled=profile, slide_id=slide_id)
        set_current_profiler(profiler)

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
        set_current_profiler(None)
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

    def _get_level_dims(self, path: str, level: int) -> Tuple[int, int]:
        img = CuImage(path)
        W, H = img.resolutions["level_dimensions"][level]
        return int(W), int(H)


class AttachROIs(Stage):
    """Attach a list[ROI] to each slide using one or more providers."""

    def __init__(self, providers: List[ROIProvider], default_whole_slide: bool = True, preclip_to_slide: bool = True):
        self.providers = list(providers)
        self.default_whole_slide = bool(default_whole_slide)
        self.preclip = bool(preclip_to_slide)

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        for s in it:
            if s.get("type") != "slide":
                continue
            all_rois: List[ROI] = []
            for prov in self.providers:
                try:
                    rois = prov.for_slide(s)
                except Exception as e:
                    logging.info(f"[AttachROIs] provider {type(prov).__name__} failed: {e}")
                    rois = []
                all_rois.extend(rois)

            if not all_rois and self.default_whole_slide:
                all_rois.extend(WholeSlideProvider().for_slide(s))

            s2 = dict(s)
            s2["type"] = "roi_list"
            s2["rois"] = all_rois
            yield s2


class TilePlanner(Stage):
    """
    Enumerate tiles per ROI using a selection policy and the global tile grid.

    tile_selection_mode:
      - "full_inside_bounds" (default): tile must fit fully within ROI.bounds().
      - "center_in_roi": tile center must be inside ROI.
    """

    def __init__(self, tile_selection_mode: str = "full_inside_bounds"):
        self.tile_selection_mode = tile_selection_mode

    def validate(self) -> None:
        self.ctx.require_key("tile_size")
        self.ctx.require_key("stride")
        self.ctx.require_key("level")

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        for s in it:
            if s.get("type") != "roi_list":
                continue
            tile_size = int(self.ctx["tile_size"])
            stride = int(self.ctx["stride"])
            rois: List[ROI] = s.get("rois", [])
            W, H = s["dims"]

            for idx, roi in enumerate(rois):
                bx, by, bw, bh = roi.bounds()
                # Enumerate grid-aligned tiles within the ROI bounding box
                x0 = _align_to_grid(max(0, bx), stride)
                y0 = _align_to_grid(max(0, by), stride)
                x1 = min(bx + bw, W)
                y1 = min(by + bh, H)

                tiles: List[Tuple[int, int]] = []
                y = y0
                while y + tile_size <= y1:
                    x = x0
                    while x + tile_size <= x1:
                        if self._accept_tile(roi, x, y, tile_size):
                            tiles.append((x, y))
                        x += stride
                    y += stride

                if not tiles:
                    continue

                yield {
                    "type": "roi_tiles",
                    "wsi_id": s["wsi_id"],
                    "wsi_path": s["wsi_path"],
                    "dims": s["dims"],
                    "roi_index": idx,
                    "roi_bounds": (bx, by, bw, bh),
                    "tiles": tiles,
                    "meta": s.get("meta", {}),
                }

    def _accept_tile(self, roi: ROI, tx: int, ty: int, tile_size: int) -> bool:
        mode = self.tile_selection_mode
        if mode == "full_inside_bounds":
            # Conservative: ensure full tile lies inside ROI bounds rectangle.
            if isinstance(roi, BoxROI):
                bx, by, bw, bh = roi.bounds()
                return (tx >= bx) and (ty >= by) and (tx + tile_size <= bx + bw) and (ty + tile_size <= by + bh)
            # For non-box geometries, fall back to center-in-ROI
            mode = "center_in_roi"

        if mode == "center_in_roi":
            cx = tx + tile_size / 2.0
            cy = ty + tile_size / 2.0
            return roi.contains_point(cx, cy)

        # Default fallback
        return True


class ReadWindowChunker(Stage):
    """
    Packs tiles into rectangular read windows of max_window_size.

    Strategy: subdivide the ROI's bounding box into stride-aligned windows
    of size up to max_window_size; emit a window only if it contains tiles.
    """

    def __init__(self, max_window_size: int = 2048, align_to_stride: bool = True):
        self.max_window_size = int(max_window_size)
        self.align_to_stride = bool(align_to_stride)

    def validate(self) -> None:
        self.ctx.require_key("tile_size")
        self.ctx.require_key("stride")

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        tile_size = int(self.ctx["tile_size"])
        stride = int(self.ctx["stride"])
        for s in it:
            if s.get("type") != "roi_tiles":
                continue

            bx, by, bw, bh = s["roi_bounds"]
            W, H = s["dims"]
            tiles: List[Tuple[int, int]] = s.get("tiles", [])
            if not tiles:
                continue

            # Define stride-aligned grid for windows
            x_start = _align_to_grid(max(0, bx), stride) if self.align_to_stride else bx
            y_start = _align_to_grid(max(0, by), stride) if self.align_to_stride else by
            x_end = min(bx + bw, W)
            y_end = min(by + bh, H)

            for yy in range(y_start, y_end, self.max_window_size):
                for xx in range(x_start, x_end, self.max_window_size):
                    ww = min(self.max_window_size, x_end - xx)
                    hh = min(self.max_window_size, y_end - yy)

                    # Collect tiles fully inside this window
                    in_window: List[Tuple[int, int]] = []
                    wx1, wy1 = xx + ww, yy + hh
                    for tx, ty in tiles:
                        if tx >= xx and ty >= yy and (tx + tile_size) <= wx1 and (ty + tile_size) <= wy1:
                            in_window.append((tx, ty))

                    if not in_window:
                        continue

                    yield {
                        "type": "region",
                        "wsi_id": s["wsi_id"],
                        "wsi_path": s["wsi_path"],
                        "region": (xx, yy, ww, hh),
                        "tiles": in_window,
                        "meta": s.get("meta", {}),
                    }


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
        prof = get_current_profiler()  # may be None if profiling is disabled
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


def _align_to_grid(v: int, stride: int, origin: int = 0) -> int:
    """Return the smallest grid value >= v on grid defined by origin & stride."""
    if stride <= 0:
        return v
    r = (v - origin) % stride
    return v if r == 0 else v + (stride - r)


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

    # Example ROI dict (compat with old code)
    rois_dict = {Path(s).stem: [(0, 0, 4000, 4000)] for s in slides}

    p = (
        WSIGrid(slides=slides, tile_size=256, stride=256, level=0)
        .then(AttachROIs(providers=[RectROIProvider(rois_dict)]))
        .then(TilePlanner())
        .then(ReadWindowChunker(max_window_size=args.max_window, align_to_stride=True))
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
