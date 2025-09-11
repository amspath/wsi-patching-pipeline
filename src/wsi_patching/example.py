# example.py
# A minimal, single-file skeleton that matches your desired pipeline:
#   WSIGrid → FilterByROI → Regionize → RegionReadAndBatch → GPUOps → PNGEncoder → ToWebDataset
#
# Design goals:
# - Region-prefetch with cuCIM: read a big region once, then slice into tiles (cheap).
# - Multi-WSI parallelism via multiprocessing (one producer process per WSI).
# - Micro-batched GPU ops (e.g., 200 tiles) in a dedicated GPU process.
# - Single, continuous WebDataset writer draining a bounded queue ("write when ready").
#
# Notes:
# - This file is intentionally compact and pragmatic. Many parts are simplified stubs you can extend.
# - cuCIM, torch, and webdataset are optional. If they are not installed, certain parts will no-op or raise
#   a helpful error. Wire in your real kernels and readers where marked with TODOs.
# - ROI format: rectangles in level coordinates [(x, y, w, h), ...] per WSI.
#
# Run style (pseudo):
#   pipeline = (
#       WSIGrid(slides, tile_size=256, stride=256, level=0)
#       .then(FilterByROI(roi_by_wsi={"slideA": [(0,0,4096,4096)]}))
#       .then(Regionize(max_region_mp=96))
#       .then(RegionReadAndBatch(cucim_workers=8))
#       .then(GPUOps(device=0, batch_size=200, batch_timeout_ms=75))
#       .then(PNGEncoder())
#       .then(ToWebDataset(pattern="/out/train-%06d.tar", maxcount=25000))
#   )
#   pipeline.run(max_producers=4, gpu_devices=[0])
#
# Author: you + ChatGPT (quick-start skeleton)

from __future__ import annotations

from collections import defaultdict
import io
import json
import math
import multiprocessing as mp
import os
import queue
import random
import tarfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

# Optional deps (these can be added to your env when ready)
try:
    import cucim
    from cucim import CuImage  # type: ignore
except Exception:  # cucim optional; code will warn if missing for real runs
    CuImage = None

try:
    import torch  # type: ignore
except Exception:
    torch = None

try:
    import imageio.v3 as iio  # type: ignore
except Exception:
    iio = None

try:
    import webdataset as wds  # type: ignore
except Exception:
    wds = None


# ------------------------------
# Shared types & small utilities
# ------------------------------

Sample = Dict[str, Any]
Rect = Tuple[int, int, int, int]  # (x, y, w, h)


def _make_emitter(ctx: RuntimeCtx, placement: str):
    q = ctx.metrics_q

    def emit(m: dict):
        if q is not None:
            m["placement"] = placement
            q.put(m, block=True)

    return emit


def _sample_bytes(s: Sample) -> int:
    if "png" in s and isinstance(s["png"], (bytes, bytearray)):
        return len(s["png"])
    p = s.get("patch")
    if p is None:
        return 0
    if hasattr(p, "nbytes"):
        return int(p.nbytes)
    try:
        return int(np.asarray(p).nbytes)
    except Exception:
        return 0


def rect_intersects(a: Rect, b: Rect) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (ax < bx + bw) and (bx < ax + aw) and (ay < by + bh) and (by < ay + ah)


def clamp_region(region: Rect, W: int, H: int) -> Rect:
    x, y, w, h = region
    x = max(0, min(x, W))
    y = max(0, min(y, H))
    w = max(0, min(w, W - x))
    h = max(0, min(h, H - y))
    return x, y, w, h


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def level_dims_with_cucim(path: str, level: int) -> Tuple[int, int]:
    if CuImage is None:
        raise RuntimeError("cuCIM not available. Install cucim to read real WSI dimensions.")
    img = CuImage(path)
    # cuCIM returns size at level 0 via .size (W,H), derive other levels by downsampling factors
    if hasattr(img, "resolutions") and isinstance(img.resolutions, dict):
        # resolutions often includes level_downsamples: [1, 2, 4, ...]
        downsamples = img.resolutions.get("level_downsamples", None)
        if downsamples is not None and level < len(downsamples):
            baseW, baseH = img.size()[:2]  # (W,H)
            ds = float(downsamples[level])
            W = int(round(baseW / ds))
            H = int(round(baseH / ds))
            return W, H
    # Fallback: assume power-of-two downsampling
    baseW, baseH = img.size()[:2]
    ds = 1 << level
    return baseW // ds, baseH // ds


def read_region_with_cucim(path: str, level: int, region: Rect, num_workers: int):
    if CuImage is None:
        raise RuntimeError("cuCIM not available. Install cucim to read real WSI regions.")
    x, y, w, h = region
    img = CuImage(path)  # Open inside this process/task
    region_img = img.read_region(location=(x, y), size=(w, h), level=level, num_workers=num_workers)
    # Convert CuImage -> numpy array explicitly
    if hasattr(region_img, "to_array"):
        arr = region_img.to_array()
    elif hasattr(region_img, "toarray"):
        arr = region_img.toarray()
    else:
        # Fallback: try __array__ if implemented
        arr = np.array(region_img)
    return arr


def ensure_png_bytes(img_arr) -> bytes:
    if iio is None:
        raise RuntimeError("imageio.v3 not available. Install imageio to encode PNG.")
    buf = io.BytesIO()
    # Do not specify colors/styles; keep defaults for speed.
    iio.imwrite(buf, img_arr, extension=".png")
    return buf.getvalue()


# ------------------------------
# Stage base + Pipeline builder
# ------------------------------


class Stage:
    placement: str = "producer"  # "producer" | "gpu" | "writer"

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        raise NotImplementedError

    def then(self, nxt: "Stage") -> "Pipeline":
        return Pipeline([self, nxt])


@dataclass
class Pipeline(Stage):
    stages: List[Stage]

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        for s in self.stages:
            it = s(it)
        return it

    def then(self, nxt: Stage) -> "Pipeline":
        return Pipeline(self.stages + [nxt])

    # Entry point: spawns processes and wires queues according to placements
    def run(
        self,
        max_producers: int = 4,
        gpu_devices: Optional[List[int]] = None,
        prod_queue_size: int = 5000,
        writer_queue_size: int = 5000,
    ):
        gpu_devices = gpu_devices or [0]

        # Partition by placement
        prod_stages: List[Stage] = []
        gpu_stages: List[Stage] = []
        writer_stages: List[Stage] = []

        seen_gpu = False
        seen_writer = False
        for s in self.stages:
            if s.placement == "producer" and not seen_gpu and not seen_writer:
                prod_stages.append(s)
            elif s.placement == "gpu" and not seen_writer:
                seen_gpu = True
                gpu_stages.append(s)
            else:
                seen_writer = True
                writer_stages.append(s)

        if not writer_stages or writer_stages[-1].placement != "writer":
            raise RuntimeError("Pipeline must end with a writer-stage (e.g., ToWebDataset).")

        # Build shared queues
        ctx = RuntimeCtx()
        ctx.prod_to_gpu = mp.Queue(maxsize=prod_queue_size)
        ctx.gpu_to_writer = mp.Queue(maxsize=writer_queue_size)
        ctx.metrics_q = mp.Queue(maxsize=10000)

        # Discover slides from the first stage (WSIGrid) in prod_stages
        slides = discover_slides_from_pipeline(prod_stages)

        # Start aggregator (expect EOS from: all producer procs + all GPU procs + writer)
        expected_eos = len(slides) + len(gpu_devices) + 1
        metrics_p = mp.Process(
            target=metrics_aggregator_main, args=(ctx.metrics_q, expected_eos), name="metrics", daemon=True
        )
        metrics_p.start()

        # Writer process (single)
        writer_p = mp.Process(target=writer_process_main, args=(writer_stages, ctx), name="writer", daemon=True)
        writer_p.start()

        # GPU processes (one per device)
        gpu_ps: List[mp.Process] = []
        for dev in gpu_devices:
            gpu_p = mp.Process(
                target=gpu_process_main, args=(gpu_stages, ctx, len(slides), dev), name=f"gpu:{dev}", daemon=True
            )
            gpu_p.start()
            gpu_ps.append(gpu_p)

        # Producer processes (one per WSI, up to max_producers concurrently)
        # We will launch up to max_producers at a time to avoid oversubscribing.
        # Each producer runs prod_stages for its single WSI and emits to prod_to_gpu.
        active: List[mp.Process] = []
        launched = 0
        for slide in slides:
            while len(active) >= max_producers:
                # Wait for any producer to finish
                for p in list(active):
                    if not p.is_alive():
                        p.join()
                        active.remove(p)
                if len(active) >= max_producers:
                    time.sleep(0.05)

            p = mp.Process(
                target=producer_process_main,
                args=(prod_stages, ctx, slide),
                name=f"producer:{Path(slide).stem}",
                daemon=True,
            )
            p.start()
            active.append(p)
            launched += 1

        # Wait for all producers to finish
        for p in active:
            p.join()

        # Signal GPU processes that all producers finished (one EOS per producer)
        for _ in range(len(slides)):
            ctx.prod_to_gpu.put({"_eos": True})

        # Wait for GPU to finish and pass EOS downstream
        for p in gpu_ps:
            p.join()

        # Wait for writer to close shards & exit
        writer_p.join()
        metrics_p.join()


# ------------------------------
# Stages (Producer lane)
# ------------------------------


class WSIGrid(Stage):
    """
    Source. Yields one Sample per WSI, carrying config to downstream stages.
    We *don't* enumerate tiles here; RegionReadAndBatch will slice per region.
    """

    placement = "producer"

    def __init__(
        self, slides: List[str], tile_size: int, stride: int, level: int = 0, align_origin: Tuple[int, int] = (0, 0)
    ):
        self.slides = list(slides)
        self.tile_size = int(tile_size)
        self.stride = int(stride)
        self.level = int(level)
        self.align_origin = tuple(align_origin)

    def __call__(self, _it: Iterable[Sample]) -> Iterable[Sample]:
        for path in self.slides:
            yield {
                "wsi_id": Path(path).stem,
                "meta": {"path": path, "backend": "cucim"},
                "level": self.level,
                "tile_size": self.tile_size,
                "stride": self.stride,
                "align_origin": self.align_origin,
                # FilterByROI may add "roi_rects": List[Rect]
            }


class FilterByROI(Stage):
    """
    Attach ROI rectangles (level coords) per WSI.
    roi_by_wsi: { wsi_id: [(x,y,w,h), ...], ... }
    If no ROI provided, leave it empty -> Regionize will chunk full slide.
    """

    placement = "producer"

    def __init__(self, roi_by_wsi: Optional[Dict[str, List[Rect]]] = None):
        self.roi_by_wsi = roi_by_wsi or {}

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        for s in it:
            rois = self.roi_by_wsi.get(s["wsi_id"], [])
            s = dict(s)
            s["roi_rects"] = list(rois)
            yield s


class Regionize(Stage):
    """
    Convert WSI-level sample into region tasks, respecting an MP cap.
    - If ROI rects present, split large rects to fit max_region_mp.
    - Else, tile full slide into regions under max_region_mp.
    Emits RegionTasks as Samples with {"region": (x,y,w,h)}.
    """

    placement = "producer"

    def __init__(self, max_region_mp: float = 96.0):
        self.max_region_px = int(max_region_mp * 1_000_000)

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        for s in it:
            path = s["meta"]["path"]
            level = s["level"]
            W, H = level_dims_with_cucim(path, level)
            ts = s["tile_size"]
            rois: List[Rect] = s.get("roi_rects", [])

            # Helper: split a rect into subrects under pixel cap
            def split_rect(r: Rect) -> List[Rect]:
                rx, ry, rw, rh = clamp_region(r, W, H)
                if rw * rh <= self.max_region_px:
                    return [(rx, ry, rw, rh)]
                # Chunk roughly square subregions
                target_side = int(math.sqrt(self.max_region_px))
                step_w = max(target_side, ts)
                step_h = max(target_side, ts)
                out: List[Rect] = []
                for y in range(ry, ry + rh, step_h):
                    for x in range(rx, rx + rw, step_w):
                        w = min(step_w, rx + rw - x)
                        h = min(step_h, ry + rh - y)
                        out.append((x, y, w, h))
                return out

            regions: List[Rect] = []
            if rois:
                for r in rois:
                    regions.extend(split_rect(r))
            else:
                # No ROI: cover full slide under cap
                regions = split_rect((0, 0, W, H))

            for r in regions:
                yield {**s, "region": r, "dims": (W, H)}


class RegionReadAndBatch(Stage):
    """
    For each RegionTask:
      - Read the entire region with cuCIM using num_workers (heavy I/O once).
      - Slice into stride-aligned tiles of size tile_size.
      - Filter tiles by ROI (if any) at tile granularity.
      - Emit *individual* tile samples (do not batch here; GPU lane will batch).
    """

    placement = "producer"

    def __init__(self, cucim_workers: int = 8):
        self.cucim_workers = int(cucim_workers)

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        for task in it:
            path = task["meta"]["path"]
            level = task["level"]
            ts = task["tile_size"]
            st = task["stride"]
            x0, y0, w, h = task["region"]
            rois: List[Rect] = task.get("roi_rects", [])

            # Read the whole region once
            region_img = read_region_with_cucim(path, level, (x0, y0, w, h), self.cucim_workers)
            if hasattr(region_img, "get"):
                # Just in case of cupy-like, convert
                region_img = region_img.get()
            # Normalize array shape to HWC if needed
            if region_img.ndim == 2:
                # Grayscale -> add channel
                region_img = region_img[:, :, None]

            # Compute stride-aligned starting positions to respect global grid
            ax, ay = task.get("align_origin", (0, 0))
            # Find first x ≥ x0 with (x - ax) % st == 0
            start_x = x0 + ((st - ((x0 - ax) % st)) % st)
            start_y = y0 + ((st - ((y0 - ay) % st)) % st)

            # Iterate tile coordinates inside region bounds
            for yy in range(start_y, y0 + h - ts + 1, st):
                for xx in range(start_x, x0 + w - ts + 1, st):
                    # ROI coarse filter at tile granularity
                    if rois:
                        tile_rect = (xx, yy, ts, ts)
                        keep = any(rect_intersects(tile_rect, r) for r in rois)
                        if not keep:
                            continue
                    # Slice from region array (translate to region-local coords)
                    ry = yy - y0
                    rx = xx - x0
                    patch = region_img[ry : ry + ts, rx : rx + ts, ...]
                    # Emit a single-sample item; GPU lane will batch later
                    yield {
                        "wsi_id": task["wsi_id"],
                        "coord": (xx, yy),
                        "level": level,
                        "tile_size": ts,
                        "meta": task["meta"],
                        "patch": patch,  # H x W x C (numpy)
                    }


# ------------------------------
# Stages (GPU lane)
# ------------------------------


class GPUOps(Stage):
    """
    Micro-batched GPU ops. The GPU process is responsible for *collecting* batches
    from producers (size/timeout). This stage expects a batch and returns a batch.
    Default kernel is a no-op (copy to device, back to host).
    """

    placement = "gpu"

    def __init__(self, device: int = 0, batch_size: int = 200, batch_timeout_ms: int = 75):
        self.device = device
        self.batch_size = int(batch_size)
        self.batch_timeout_ms = int(batch_timeout_ms)

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        # This stage is only executed inside the GPU process loop (see gpu_process_main),
        # where we already coalesce input items into batches of the configured size/timeout.
        # Here we simply perform the "GPU work" on the batch and yield the same batch.
        if torch is None:
            # No torch installed: just pass batches through unchanged
            for item in it:
                yield item
            return

        device = torch.device(f"cuda:{self.device}") if torch.cuda.is_available() else torch.device("cpu")

        for item in it:
            # Expect item to be {"batch": [Sample, ...]}
            batch = item.get("batch", [])
            if not batch:
                continue
            # Stack into tensor (N,H,W,C) -> (N,C,H,W)
            patches = [torch.as_tensor(s["patch"]) for s in batch]
            x = torch.stack(patches, dim=0)  # (N,H,W,C)
            if x.ndim == 3:
                x = x.unsqueeze(-1)
            x = x.permute(0, 3, 1, 2).contiguous()  # (N,C,H,W)
            x = x.to(device, non_blocking=True)

            # ===== TODO: your real kernels here =====
            # e.g., stain normalization, color space, normalization, etc.
            # For now: simple pass-through (identity)
            y = x
            # ========================================

            # Move back to CPU as (N,H,W,C), write back into samples
            y_cpu = y.detach().to("cpu")
            y_cpu = y_cpu.permute(0, 2, 3, 1).contiguous()
            for i, s in enumerate(batch):
                s["patch"] = y_cpu[i].numpy()  # replace with processed patch
            yield {"batch": batch}


class PNGEncoder(Stage):
    """
    Encode each sample in the batch to PNG bytes (fast path).
    Emits individual encoded samples (unbatched) to the writer queue.
    """

    placement = "gpu"

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        for item in it:
            batch = item.get("batch", [])
            for s in batch:
                png_bytes = ensure_png_bytes(s["patch"])
                meta = {k: v for k, v in s.items() if k != "patch"}
                yield {**meta, "png": png_bytes}


# ------------------------------
# Stages (Writer lane)
# ------------------------------
class RandomizedShardWriter(Stage):
    """
    Writer stage that buffers 4×shard_size (configurable), shuffles the buffer,
    and writes exactly `shard_size` samples per shard. On EOS, flushes remaining
    samples into final shard(s). Assumes each input Sample has "png" (bytes).
    """

    placement = "writer"

    def __init__(self, pattern: str, shard_size: int = 500, buffer_multiplier: int = 4, seed: Optional[int] = None):
        self.pattern = pattern
        self.shard_size = int(shard_size)
        self.buffer_limit = int(buffer_multiplier) * self.shard_size
        self.rng = random.Random(seed) if seed is not None else random

        # If webdataset is unavailable, fall back to an internal tar shard writer
        self._use_wds = wds is not None

    # ---------------- internal helpers ----------------
    class _TarShardWriter:
        def __init__(self, pattern: str, shard_size: int):
            self.pattern = pattern
            self.shard_size = shard_size
            self._tar = None
            self._count = 0
            self._idx = 0
            self._open_new()

        def _open_new(self):
            if self._tar is not None:
                self._tar.close()
            path = self.pattern.replace("%06d", f"{self._idx:06d}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._tar = tarfile.open(path, "w")
            self._count = 0
            self._idx += 1

        def write(self, key: str, png_bytes: bytes, meta: dict):
            # rotate?
            if self._count >= self.shard_size:
                self._open_new()
            # png
            info = tarfile.TarInfo(name=f"{key}.png")
            info.size = len(png_bytes)
            self._tar.addfile(info, io.BytesIO(png_bytes))
            # json
            jbytes = json.dumps(meta).encode("utf-8")
            jinfo = tarfile.TarInfo(name=f"{key}.json")
            jinfo.size = len(jbytes)
            self._tar.addfile(jinfo, io.BytesIO(jbytes))
            self._count += 1

        def close(self):
            if self._tar is not None:
                self._tar.close()
                self._tar = None

    def _open_writer(self):
        if self._use_wds:
            return wds.ShardWriter(self.pattern, maxcount=self.shard_size)
        else:
            return self._TarShardWriter(self.pattern, self.shard_size)

    def _close_writer(self, writer):
        # both classes expose close()
        writer.close()

    def _write_one(self, writer, s: Sample):
        key = s.get("__key__")
        if not key:
            x, y = s["coord"]
            key = f"{s['wsi_id']}-{x}-{y}-L{s['level']}"
        meta = {k: v for k, v in s.items() if k not in ("png", "__key__")}
        if self._use_wds:
            writer.write({"__key__": key, "png": s["png"], "json": json.dumps(meta).encode("utf-8")})
        else:
            writer.write(key, s["png"], meta)

    # --------------- stage entrypoint -----------------
    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        """
        Consume an iterator of encoded samples (each with 'png' bytes) and
        write randomized shards. Yields nothing (sink).
        """
        buf: List[Sample] = []
        writer = self._open_writer()

        def write_shard_from_buffer():
            # assumes len(buf) >= self.shard_size
            shard = buf[: self.shard_size]
            del buf[: self.shard_size]
            for s in shard:
                self._write_one(writer, s)

        try:
            for s in it:
                buf.append(s)
                if len(buf) >= self.buffer_limit:
                    # Shuffle the entire buffer, then write exactly one shard
                    self.rng.shuffle(buf)
                    write_shard_from_buffer()

            # End of stream: flush everything left (may be multiple shards + final partial)
            while len(buf) >= self.shard_size:
                self.rng.shuffle(buf)
                write_shard_from_buffer()

            if buf:
                # Final partial shard (write as is after one last shuffle)
                self.rng.shuffle(buf)
                for s in buf:
                    self._write_one(writer, s)
                buf.clear()

        finally:
            self._close_writer(writer)

        # sinks yield nothing
        return iter(())


class ToWebDataset(Stage):
    """
    Single-process writer that drains encoded samples and writes to shards.
    Uses webdataset.ShardWriter if available; otherwise a minimal tar fallback.
    Each sample must have:
      - a unique key (we derive from wsi_id + coord + level),
      - "png" bytes payload,
      - optional JSON sidecar with metadata (auto-generated).
    """

    placement = "writer"

    def __init__(self, pattern: str, maxcount: int = 25_000):
        self.pattern = pattern
        self.maxcount = int(maxcount)

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        # Writer stages are executed inside the writer process (see writer_process_main).
        # We consume everything and yield nothing (sink).
        if wds is not None:
            shard = wds.ShardWriter(self.pattern, maxcount=self.maxcount)
            try:
                for s in it:
                    key = s.get("__key__") or f"{s['wsi_id']}-{s['coord'][0]}-{s['coord'][1]}-L{s['level']}"
                    meta = {k: v for k, v in s.items() if k not in ("png", "__key__")}
                    shard.write({"__key__": key, "png": s["png"], "json": json.dumps(meta).encode("utf-8")})
            finally:
                shard.close()
        else:
            # Minimal tar fallback (rolls by count)
            shard_idx = 0
            shard_count = 0
            tar = None

            def open_new_tar(idx: int):
                nonlocal tar
                tar_path = self.pattern.replace("%06d", f"{idx:06d}")
                os.makedirs(os.path.dirname(tar_path), exist_ok=True)
                tar = tarfile.open(tar_path, "w")

            def close_tar():
                nonlocal tar
                if tar is not None:
                    tar.close()
                    tar = None

            try:
                open_new_tar(shard_idx)
                for s in it:
                    if shard_count >= self.maxcount:
                        close_tar()
                        shard_idx += 1
                        shard_count = 0
                        open_new_tar(shard_idx)

                    key = s.get("__key__") or f"{s['wsi_id']}-{s['coord'][0]}-{s['coord'][1]}-L{s['level']}"
                    png_bytes = s["png"]
                    meta = {k: v for k, v in s.items() if k not in ("png", "__key__")}
                    # Write PNG
                    info = tarfile.TarInfo(name=f"{key}.png")
                    info.size = len(png_bytes)
                    tar.addfile(info, io.BytesIO(png_bytes))
                    # Write JSON
                    jbytes = json.dumps(meta).encode("utf-8")
                    jinfo = tarfile.TarInfo(name=f"{key}.json")
                    jinfo.size = len(jbytes)
                    tar.addfile(jinfo, io.BytesIO(jbytes))
                    shard_count += 1
            finally:
                close_tar()

        # Sink: yields nothing
        return iter(())


# ------------------------------
# Runtime: process/queue plumbing
# ------------------------------


@dataclass
class RuntimeCtx:
    prod_to_gpu: Optional[mp.Queue] = None
    gpu_to_writer: Optional[mp.Queue] = None
    metrics_q: Optional[mp.Queue] = None


def discover_slides_from_pipeline(prod_stages: List[Stage]) -> List[str]:
    """
    Execute just the WSIGrid stage to list slides. We detect it by type.
    """
    grid = next((s for s in prod_stages if isinstance(s, WSIGrid)), None)
    if grid is None:
        raise RuntimeError("WSIGrid stage is required in the producer pipeline.")
    # We don't want to run everything; just extract slide paths
    return list(grid.slides)


def execute_stages_locally(stages: List[Stage], it: Iterable[Sample], emit_metric) -> Iterable[Sample]:
    """
    Run stages synchronously in this process, timing each stage.
    """
    for s in stages:
        stage_name = s.__class__.__name__
        placement = getattr(s, "placement", "producer")

        t0 = time.perf_counter()
        out = s(it)  # call the stage
        call_time = time.perf_counter() - t0

        def stage_iter(out_iter):
            items_out = 0
            bytes_out = 0
            t_iter = 0.0
            for item in out_iter:
                t1 = time.perf_counter()
                yield item
                t_iter += time.perf_counter() - t1
                items_out += 1
                bytes_out += _sample_bytes(item)
            emit_metric(
                {
                    "type": "stage",
                    "stage": stage_name,
                    "placement": placement,
                    "items_out": items_out,
                    "bytes_out": bytes_out,
                    "time_s": call_time + t_iter,
                }
            )

        it = stage_iter(out)
    return it


def producer_process_main(prod_stages: List[Stage], ctx: RuntimeCtx, slide_path: str):
    emit = _make_emitter(ctx, "producer")
    try:
        first = prod_stages[0]
        if not isinstance(first, WSIGrid):
            raise RuntimeError("First producer stage must be WSIGrid.")
        grid = WSIGrid(
            slides=[slide_path],
            tile_size=first.tile_size,
            stride=first.stride,
            level=first.level,
            align_origin=first.align_origin,
        )
        local_stages = [grid] + prod_stages[1:]

        it: Iterable[Sample] = iter(())
        it = execute_stages_locally(local_stages, it=it, emit_metric=emit)

        q = ctx.prod_to_gpu
        assert q is not None
        items = 0
        bytes_tot = 0
        t_queue = 0.0
        for s in it:
            items += 1
            bytes_tot += _sample_bytes(s)
            t0 = time.perf_counter()
            q.put(s)  # may block (backpressure)
            t_queue += time.perf_counter() - t0

        # report queue time
        emit({"type": "queue_put", "queue": "prod→gpu", "items": items, "bytes": bytes_tot, "time_s": t_queue})

    except Exception:
        traceback.print_exc()
    finally:
        # EOS is sent by parent to GPU procs; but the metrics aggregator expects an EOS per producer
        emit({"type": "eos"})


def gpu_process_main(gpu_stages: List[Stage], ctx: RuntimeCtx, num_producers: int, device_id: int):
    emit = _make_emitter(ctx, "gpu")

    inQ = ctx.prod_to_gpu
    outQ = ctx.gpu_to_writer
    assert inQ is not None and outQ is not None

    gpu_ops = next((s for s in gpu_stages if isinstance(s, GPUOps)), None)
    batch_size = gpu_ops.batch_size if gpu_ops else 200
    timeout_ms = gpu_ops.batch_timeout_ms if gpu_ops else 75

    def run_gpu_pipeline(batch: List[Sample]) -> List[Sample]:
        # Run the GPU stages with per-stage timing
        out = [{"batch": batch}]
        for s in gpu_stages:
            t0 = time.perf_counter()
            # stage may yield multiple items; collect them
            tmp = []
            for item in s(out):
                tmp.append(item)
            dt = time.perf_counter() - t0
            emit(
                {
                    "type": "stage",
                    "stage": s.__class__.__name__,
                    "items_out": len(tmp),
                    "bytes_out": sum(_sample_bytes(x) for x in tmp),
                    "time_s": dt,
                }
            )
            out = tmp
        # PNGEncoder produces individual samples
        return out

    eos_seen = 0
    buffer: List[Sample] = []
    last_flush = time.time()
    items_put = 0
    bytes_put = 0
    t_put = 0.0

    def flush_if_ready(force: bool = False):
        nonlocal buffer, last_flush, items_put, bytes_put, t_put
        now = time.time()
        if not buffer:
            last_flush = now
            return
        age_ms = (now - last_flush) * 1000.0
        if force or (len(buffer) >= batch_size) or (age_ms >= timeout_ms):
            encoded_samples = run_gpu_pipeline(buffer)
            for s in encoded_samples:
                t0 = time.perf_counter()
                outQ.put(s)
                t_put += time.perf_counter() - t0
                items_put += 1
                bytes_put += _sample_bytes(s)
            buffer = []
            last_flush = time.time()

    try:
        while True:
            try:
                item = inQ.get(timeout=0.1)
            except queue.Empty:
                item = None

            if item is None:
                flush_if_ready(False)
                continue

            if isinstance(item, dict) and item.get("_eos"):
                eos_seen += 1
                flush_if_ready(True)
                if eos_seen >= num_producers:
                    outQ.put({"_eos": True})
                    break
                continue

            buffer.append(item)
            flush_if_ready(False)
    except Exception:
        traceback.print_exc()
        try:
            outQ.put({"_eos": True})
        except Exception:
            pass
    finally:
        emit({"type": "queue_put", "queue": "gpu→writer", "items": items_put, "bytes": bytes_put, "time_s": t_put})
        emit({"type": "eos"})


def writer_process_main(writer_stages: List[Stage], ctx: RuntimeCtx):
    emit = _make_emitter(ctx, "writer")
    inQ = ctx.gpu_to_writer
    assert inQ is not None

    class QIter:
        def __iter__(self):
            return self

        def __next__(self):
            item = inQ.get()
            if isinstance(item, dict) and item.get("_eos"):
                raise StopIteration
            return item

    it: Iterable[Sample] = QIter()
    try:
        for _ in execute_stages_locally(writer_stages, it, emit_metric=emit):
            pass
    except StopIteration:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        emit({"type": "eos"})


def metrics_aggregator_main(q: mp.Queue, expected_eos: int):
    # aggregate by (placement, stage)
    agg = defaultdict(lambda: {"time_s": 0.0, "items_out": 0, "bytes_out": 0})
    eos = 0
    while True:
        m = q.get()
        if m.get("type") == "eos":
            eos += 1
            if eos >= expected_eos:
                break
            continue
        if m.get("type") == "stage":
            key = (m["placement"], m["stage"])
            a = agg[key]
            a["time_s"] += m.get("time_s", 0.0)
            a["items_out"] += m.get("items_out", 0)
            a["bytes_out"] += m.get("bytes_out", 0)
        elif m.get("type") == "queue_put":
            key = (m["placement"], f"QueuePut@{m.get('queue')}")
            a = agg[key]
            a["time_s"] += m.get("time_s", 0.0)
            a["items_out"] += m.get("items", 0)
            a["bytes_out"] += m.get("bytes", 0)

    # Print summary
    rows = []
    for (placement, stage), v in agg.items():
        t = v["time_s"]
        n = v["items_out"]
        b = v["bytes_out"]
        ips = (n / t) if t > 0 else 0.0
        mbps = (b / (1024 * 1024)) / t if t > 0 else 0.0
        rows.append((t, placement, stage, n, b, ips, mbps))
    rows.sort(reverse=True)  # by time

    print("\n=== Pipeline profile (aggregated) ===")
    print(f"{'time_s':>9}  {'where':<9}  {'stage':<28}  {'items':>10}  {'MB_out':>10}  {'items/s':>10}  {'MB/s':>10}")
    for t, pl, st, n, b, ips, mbps in rows:
        print(f"{t:9.2f}  {pl:<9}  {st:<28}  {n:10d}  {b / (1024 * 1024):10.2f}  {ips:10.1f}  {mbps:10.2f}")
    print("====================================\n")


# ------------------------------
# Demo / scaffolding
# ------------------------------


def _demo_build_pipeline(slides: List[str]) -> Pipeline:
    """
    Small helper to show how you'd wire the stages together.
    """
    rois = {Path(s).stem: [(0, 0, 4000, 4000)] for s in slides}

    pipeline = (
        WSIGrid(slides=slides, tile_size=256, stride=256, level=0)
        .then(FilterByROI(roi_by_wsi=rois))
        .then(Regionize(max_region_mp=96))
        .then(RegionReadAndBatch(cucim_workers=8))
        .then(GPUOps(device=0, batch_size=200, batch_timeout_ms=75))  # placement = gpu
        .then(PNGEncoder())  # placement = gpu
        .then(
            RandomizedShardWriter(pattern="./output/train-%06d.tar", shard_size=500, buffer_multiplier=2)
        )  # placement = writer
    )
    return pipeline


def main(argv=None):
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
        "./data/RBIO-GC072-HE-09.tiff",
        "./data/RBIO-GC072-HE-10.tiff",
    ]
    if not slides:
        print("Populate 'slides' with real WSI paths before running.")
        exit(0)

    p = _demo_build_pipeline(slides)
    # Tune concurrency as needed
    start_time = time.time()
    p.run(max_producers=10, gpu_devices=[0])
    print(f"Done in {time.time() - start_time:.1f} seconds.")
