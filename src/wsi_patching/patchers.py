import math
from typing import Iterable, List, Tuple

import numpy as np
from cucim import CuImage

from wsi_patching.core import Stage
from wsi_patching.typing import Rect, Sample


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


def clamp_region(region: Rect, W: int, H: int) -> Rect:
    x, y, w, h = region
    x = max(0, min(x, W))
    y = max(0, min(y, H))
    w = max(0, min(w, W - x))
    h = max(0, min(h, H - y))
    return x, y, w, h


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


def rect_intersects(a: Rect, b: Rect) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (ax < bx + bw) and (bx < ax + aw) and (ay < by + bh) and (by < ay + ah)


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
