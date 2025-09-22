from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Union

if TYPE_CHECKING:
    import cupy as cp
import numpy as np

from wsi_patching.backends.cucim_openslide import read_region
from wsi_patching.backends.cupy_numpy import get_xp_backend
from wsi_patching.core.pipeline import Stage
from wsi_patching.core.regions_of_interest import ROI, BoxROI, WholeSlideProvider
from wsi_patching.utils.types import CollatedPatchBatch, RegionTask, Slide, SlideWithROIs, TilePlan


class TilePlanner(Stage):
    """
    Divide slides (with or without ROIs) into patches on a regular grid.

    The TilePlanner emits a TilePlan. Each TilePlan corresponds to a single slide,
    and might has a list of coords corresponding to coordinates of patches to be extracted.
    If ROIs are attached to the slide, all generated coordinates will lie within the ROIs.
    If no ROIs are attached, the WholeSlideProvider is used to generate a single ROI
    covering the entire slide.
    """

    def __init__(self, tile_selection_mode: str = "full_inside_bounds"):
        self.tile_selection_mode = tile_selection_mode

    def validate(self) -> None:
        self.ctx.require_key("tile_size")
        self.ctx.require_key("stride")
        self.ctx.require_key("level")

    def __call__(self, it: Iterable[Union[Slide, SlideWithROIs]]) -> Iterable[TilePlan]:
        tile_size = int(self.ctx["tile_size"])
        stride = int(self.ctx["stride"])

        for s in it:
            rois = getattr(s, "rois", None) or WholeSlideProvider().for_slide(s)
            W, H = s.dims
            for idx, roi in enumerate(rois):
                bx, by, bw, bh = roi.bounds()
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

                if tiles:
                    yield TilePlan(s.wsi_id, s.wsi_path, s.dims, idx, (bx, by, bw, bh), tiles, meta=s.meta)
                else:
                    self.log.warning(
                        f"TilePlanner: no tiles found for slide {s.wsi_id} ROI {idx} bounds {bx, by, bw, bh}"
                    )

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

    The goal here is to batch together coordinates that lie closely together.
    These can be read as a single large region read from the WSI,
    and then sliced into individual patches in numpy.
    Since the region read is square, we group together patches that fit in a square.
    Each slide object is split into one or more region tasks dependend on the max_window_size.
    Controlling the max_window_size is a tradeoff between memory use and read efficiency.
    A larger window size means fewer, larger reads, but more memory use.
    A smaller window size means more, smaller reads, and less memory use.
    As the library multiprocesses over slides, reading in complete slides for each cpu might be too much memory.

    Strategy: subdivide the ROI's bounding box into stride-aligned windows
    of size up to max_window_size; emit a window only if it contains tiles.
    """

    def __init__(self, max_window_size: Optional[int] = None, align_to_stride: bool = True):
        self.max_window_size = max_window_size
        self.align_to_stride = bool(align_to_stride)

    def validate(self) -> None:
        self.ctx.require_key("tile_size")
        self.ctx.require_key("stride")

        if self.max_window_size is None:
            self.max_window_size = 20 * int(self.ctx["tile_size"])
            self.log.info(f"Defaulting max_window_size to 20*tile_size={self.max_window_size}")

        if self.max_window_size % self.ctx["tile_size"] != 0:
            raise ValueError(
                "ReadWindowChunker: max_window_size must be a multiple of tile_size to avoid unnecessary padding"
            )

        if self.max_window_size > 10000:
            self.log.warning(
                f"ReadWindowChunker: max_window_size {self.max_window_size} is quite large, "
                "this may lead to high memory usage and OOM errors. Consider reducing it."
            )

    def __call__(self, it: Iterable[TilePlan]) -> Iterable[RegionTask]:
        tile_size = int(self.ctx["tile_size"])
        stride = int(self.ctx["stride"])

        for plan in it:
            bx, by, bw, bh = plan.roi_bounds
            W, H = plan.dims
            if not plan.tiles:
                self.log.warning(f"ReadWindowChunker: no tiles in plan for slide {plan.wsi_id} ROI {plan.roi_index}")
                continue

            x_start = _align_to_grid(max(0, bx), stride) if self.align_to_stride else bx
            y_start = _align_to_grid(max(0, by), stride) if self.align_to_stride else by
            x_end, y_end = min(bx + bw, W), min(by + bh, H)

            for yy in range(y_start, y_end, self.max_window_size):
                for xx in range(x_start, x_end, self.max_window_size):
                    ww, hh = min(self.max_window_size, x_end - xx), min(self.max_window_size, y_end - yy)
                    in_window: List[Tuple[int, int]] = []
                    wx1, wy1 = xx + ww, yy + hh
                    for tx, ty in plan.tiles:
                        if tx >= xx and ty >= yy and (tx + tile_size) <= wx1 and (ty + tile_size) <= wy1:
                            in_window.append((tx, ty))

                    if in_window:
                        yield RegionTask(plan.wsi_id, plan.wsi_path, (xx, yy, ww, hh), in_window, meta=plan.meta)


class RegionReadAndBatch(Stage):
    """
    For each RegionTask:
      - open slide (per-process, no sharing)
      - read the entire region once (cuCIM read_region with num_workers, else PIL crop)
      - slice region into tile patches
      - accumulate into batches of 'batch_size', yield {"batch": [samples,...]}
      - Output patch batches are of shape [B, H, W, C]
      - Output patches are of type dtype (default np.uint8) and are within range [0, 255]
      - Changing dtype to e.g. np.float32 is allowed, but no normalization is applied
    """

    def __init__(self, batch_size: int = 200, num_workers: int = 8, dtype: str = np.uint8):
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.dtype = dtype

    def validate(self) -> None:
        self.ctx.require_key("tile_size")
        self.ctx.require_key("level")
        self.ctx.require_key("use_gpu")

    def __call__(self, it: Iterable[RegionTask]) -> Iterable[CollatedPatchBatch]:
        xp = get_xp_backend(self.ctx["use_gpu"])
        tile_size = int(self.ctx["tile_size"])
        level = int(self.ctx["level"])

        for task in it:
            x0, y0, w, h = task.region
            region_img = read_region(
                task.wsi_path, x0, y0, w, h, level, use_gpu=self.ctx["use_gpu"], num_workers_cucim=self.num_workers
            )

            coords: List[Tuple[int, int]] = []
            patches: List[Union[np.ndarray, "cp.ndarray"]] = []

            for tx, ty in task.tiles:
                rx, ry = tx - x0, ty - y0
                patch = region_img[ry : ry + tile_size, rx : rx + tile_size, :]
                if patch.shape[:2] != (tile_size, tile_size):
                    continue

                coords.append((tx, ty))
                patches.append(patch)

                if len(patches) >= self.batch_size:
                    yield self._make_batch(task, coords, patches, xp)
                    coords, patches = [], []

            if patches:
                yield self._make_batch(task, coords, patches, xp)

    def _make_batch(self, task, coords, patches, xp):
        batch_array = np.stack(patches, axis=0)
        patches_xp = xp.asarray(batch_array, dtype=self.dtype)
        return CollatedPatchBatch(wsi_id=task.wsi_id, coords=coords, patches=patches_xp, meta=task.meta)


def _align_to_grid(v: int, stride: int, origin: int = 0) -> int:
    """Return the smallest grid value >= v on grid defined by origin & stride."""
    if stride <= 0:
        return v
    r = (v - origin) % stride
    return v if r == 0 else v + (stride - r)
