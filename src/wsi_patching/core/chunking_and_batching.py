import logging
from typing import Iterable, List, Optional, Tuple

import numpy as np
from cucim import CuImage

from wsi_patching.core.pipeline import Sample, Stage
from wsi_patching.core.regions_of_interest import ROI, BoxROI, WholeSlideProvider


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
            if s.get("type") != "slide":
                logging.warning(f"TilePlanner skipping non-slide item: {s.get('type')}")
                continue

            tile_size = int(self.ctx["tile_size"])
            stride = int(self.ctx["stride"])
            rois: List[ROI] = s.get("rois", [])
            W, H = s["dims"]

            if rois is None or len(rois) == 0:
                logging.warning(f"No ROIs found for slide {s['wsi_id']}, defaulting to whole slide.")
                rois = WholeSlideProvider().for_slide(s)

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
                    "type": "slide",
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
            if s.get("type") != "slide":
                logging.warning(f"TilePlanner skipping non-slide item: {s.get('type')}")
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
