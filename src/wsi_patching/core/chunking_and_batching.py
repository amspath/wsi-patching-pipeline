from typing import TYPE_CHECKING, Iterable, List, Literal, Optional, Tuple, Union

import cv2

from wsi_patching.backends.cucim_openslide_isyntax import read_region, get_level_for_resolution, get_level_downsamples
from wsi_patching.backends.cupy_numpy import get_xp_backend
from wsi_patching.core.pipeline import Stage
from wsi_patching.core.types.types import CollatedPatchBatch, RegionTask, Slide, SlideWithROIs, TilePlan
from wsi_patching.regions_of_interest.roi_providers import WholeSlideProvider
from wsi_patching.regions_of_interest.rois import ROI
from wsi_patching.utils.meta_typing import PipelineContext

if TYPE_CHECKING:
    import cupy as cp
import numpy as np

# Mapping from human-readable interpolation names to OpenCV interpolation flags.
# Used by RegionReadAndBatch when resampling regions to the requested resolution.
_CV2_INTERPOLATION: dict[str, int] = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


class TilePlanner(Stage):
    """
    Divide slides (with or without ROIs) into patches on a regular grid.

    The TilePlanner emits a TilePlan. Each TilePlan corresponds to a single slide,
    and has a list of coordinates corresponding to patches to be extracted.
    If ROIs are attached to the slide, all generated coordinates will lie within the ROIs.
    If no ROIs are attached, the WholeSlideProvider is used to generate a single ROI
    covering the entire slide.

    tile_selection_mode:
      - "any_overlap" (default): accept tile if any pixel overlaps ROI.
      - "full_inside_bounds": accept tile only if fully inside ROI bounds rectangle (exact for BoxROI).
      - "center_in_roi": accept tile if its center is inside ROI.
    """

    def __init__(
        self,
        tile_size: int,
        stride: int,
        tile_selection_mode: Literal["any_overlap", "full_inside_bounds", "center_in_roi"] = "any_overlap",
    ):
        """
        Args:
            tile_size: size of square patches to extract
            stride: spacing between patch top-left corners of each of the patches
            tile_selection_mode: how to select tiles with respect to ROIs.
                - "any_overlap": accept tile if any pixel overlaps ROI.
                - "full_inside_bounds": accept tile only if fully inside ROI bounds rectangle (exact for BoxROI).
                - "center_in_roi": accept tile if its center is inside ROI.
        """
        self.tile_size = tile_size
        self.stride = stride
        self.tile_selection_mode = tile_selection_mode

    def export_context(self, ctx) -> None:
        ctx["tile_size"] = self.tile_size
        ctx["stride"] = self.stride

    def validate(self) -> None:
        self.ctx.require_key("tile_size")
        self.ctx.require_key("stride")

        if self.ctx["tile_size"] <= 0:
            raise ValueError("Tile size must be positive")
        if self.ctx["stride"] <= 0:
            raise ValueError("Stride must be positive")
        if self.ctx["stride"] > self.ctx["tile_size"]:
            self.log.warning("Stride is larger than tile size, resulting in gaps between tiles.")

    def __call__(self, it: Iterable[Union[Slide, SlideWithROIs]]) -> Iterable[TilePlan]:
        tile_size = int(self.ctx["tile_size"])
        stride = int(self.ctx["stride"])

        for s in it:
            rois = getattr(s, "rois", None) or WholeSlideProvider().for_slide(s)
            W, H = s.dims
            for idx, roi in enumerate(rois):
                bx, by, bw, bh = roi.bounds()
                x1 = min(bx + bw, W)
                y1 = min(by + bh, H)
                tiles: List[Tuple[int, int]] = []

                xs = self._axis_positions(bx, x1, tile_size, stride)
                ys = self._axis_positions(by, y1, tile_size, stride)

                for y in ys:
                    for x in xs:
                        if self._accept_tile(roi, x, y, tile_size):
                            tiles.append((x, y))

                if tiles:
                    yield TilePlan(
                        wsi_id=s.wsi_id,
                        wsi_path=s.wsi_path,
                        dims=s.dims,
                        level=s.level,
                        roi_index=idx,
                        roi_bounds=(bx, by, bw, bh),
                        tiles=tiles,
                        downsample=s.downsample,
                        resample_factor=s.resample_factor,
                        meta={
                            **s.meta,
                            "roi_bounds": (bx, by, bw, bh),
                            "slide.stride": stride,
                            "slide.tile_size": tile_size,
                        },
                    )
                else:
                    self.log.warning(f"No tiles found for slide {s.wsi_id} ROI {idx} bounds {bx, by, bw, bh}")

    def _axis_positions(self, start: int, end: int, tile_size: int, stride: int) -> List[int]:
        """
        Generate tile start positions along a single axis.

        Assumes stride < tile_size if you want overlap & full coverage.
        """
        roi_len = end - start

        # 1) ROI smaller than a tile: just one tile anchored at start.
        if roi_len <= tile_size:
            return [start]

        # 2) Normal case: stride < tile_size, ROI larger than tile.
        positions: List[int] = []
        pos = start

        # keep stepping as long as "previous start + tile_size" still needs to reach 'end'
        last_tile_correction = -stride + tile_size if stride < tile_size else 0
        while pos + last_tile_correction < end:
            positions.append(pos)
            pos += stride

        return positions

    def _accept_tile(self, roi: ROI, tx: int, ty: int, tile_size: int) -> bool:
        mode = self.tile_selection_mode

        if mode == "any_overlap":
            return roi.intersects_patch(tx, ty, tile_size, tile_size)
        elif mode == "full_inside_bounds":
            return roi.contains_patch(tx, ty, tile_size, tile_size)
        elif mode == "center_in_roi":
            cx = tx + tile_size / 2.0
            cy = ty + tile_size / 2.0
            return roi.contains_point(cx, cy)
        else:
            raise ValueError(f"TilePlanner: unknown tile_selection_mode '{mode}'")


class ReadWindowChunker(Stage):
    """
    Packs tiles into rectangular read windows of max_window_size.

    The goal here is to batch together coordinates that lie closely together.
    These can be read as a single large region read from the WSI, and then sliced into individual patches.
    Since the region read is square, we group together patches that fit in a square.
    Each slide object is split into one or more region tasks dependend on the max_window_size.
    Controlling the max_window_size is a tradeoff between memory use and read efficiency.
    A larger window size means fewer, larger reads, but more memory use.
    A smaller window size means more, smaller reads, and less memory use.
    As the library multiprocesses over slides, reading in complete slides for each cpu might be too much memory.
    """

    def __init__(self, max_window_size: Optional[int] = None):
        self.max_window_size = max_window_size

    def validate(self) -> None:
        self.ctx.require_key("stride")

        if self.max_window_size is None:
            self.max_window_size = (5000 // self.ctx["stride"]) * self.ctx["stride"]
            self.log.info(f"Defaulting max_window_size to (5000//stride)*stride={self.max_window_size}")

        if self.max_window_size % self.ctx["stride"] != 0:
            raise ValueError(
                "ReadWindowChunker: max_window_size must be a multiple of stride to avoid unnecessary padding"
            )

        if self.max_window_size > 10000:
            self.log.warning(
                f"max_window_size {self.max_window_size} is quite large, "
                "this may lead to high memory usage and OOM errors. Consider reducing it."
            )

    def __call__(self, it: Iterable[TilePlan]) -> Iterable[RegionTask]:
        tile_size = int(self.ctx["tile_size"])
        stride = int(self.ctx["stride"])
        overlap = max(0, tile_size - stride)  # e.g. 64 when tile_size=1022, stride=958

        for plan in it:
            bx, by, bw, bh = plan.roi_bounds
            W, H = plan.dims
            if not plan.tiles:
                self.log.warning(f"No tiles in plan for slide {plan.wsi_id} ROI {plan.roi_index}")
                continue

            x_roi_end, y_roi_end = min(bx + bw, W), min(by + bh, H)

            # We iterate base (non-overlapping) assignment windows of size max_window_size,
            # but we READ a region that is expanded by `overlap` to the right/bottom
            # to ensure any overlapping tiles near the boundary can be sliced without padding.
            for y_region_start in range(by, y_roi_end, self.max_window_size):
                for x_region_start in range(bx, x_roi_end, self.max_window_size):
                    in_window: List[Tuple[int, int]] = []

                    # Base window size (defines which tiles belong to this chunk)
                    base_w = min(self.max_window_size, x_roi_end - x_region_start)
                    base_h = min(self.max_window_size, y_roi_end - y_region_start)
                    x_region_end = x_region_start + base_w
                    y_region_end = y_region_start + base_h

                    # Read window size (expanded to include overlap, clamped to ROI end)
                    read_w = min(base_w + overlap, x_roi_end - x_region_start)
                    read_h = min(base_h + overlap, y_roi_end - y_region_start)

                    # Assign tiles by their top-left being inside the BASE window.
                    # The READ window is larger so tiles near the boundary still have full pixel support.
                    for tx, ty in plan.tiles:
                        if (x_region_start <= tx < x_region_end) and (y_region_start <= ty < y_region_end):
                            in_window.append((tx, ty))

                    if in_window:
                        yield RegionTask(
                            wsi_id=plan.wsi_id,
                            wsi_path=plan.wsi_path,
                            wsi_dims=plan.dims,
                            level=plan.level,
                            region=(x_region_start, y_region_start, read_w, read_h),
                            tiles=in_window,
                            downsample=plan.downsample,
                            resample_factor=plan.resample_factor,
                            meta=plan.meta,
                        )


class RegionReadAndBatch(Stage):
    """
    For each RegionTask:
      - open slide (per-process, no sharing)
      - read the entire region once (cuCIM read_region with num_workers, else PIL crop)
      - slice region into tile patches
      - Pad or drop patches at the wsi edge according to edge_policy
      - accumulate into batches of 'batch_size', yield {"batch": [samples,...]}
      - Output patch batches are of shape [B, H, W, C]
      - Output patches are of type dtype (default np.uint8) and are within range [0, 255]
      - Changing dtype to e.g. np.float32 is allowed, but no normalization is applied
    """

    def __init__(
        self,
        batch_size: int = 200,
        num_workers: int = 8,
        wsi_edge_policy: Literal["drop", "pad_with_zeros", "pad_with_edge"] = "pad_with_zeros",
        roi_edge_policy: Literal["read_from_image", "use_wsi_edge_policy"] = "use_wsi_edge_policy",
        dtype: str = np.uint8,
        fallback_mode: Literal["nearest", "floor", "ceil", "error", "resample"] = "error",
        resample_interpolation: Literal["nearest", "linear", "cubic", "area", "lanczos"] = "lanczos",
    ):
        """
        Args:
            batch_size: Maximum number of patches per output batch
            num_workers: number of parallel workers for reading WSI images using cuCIM
            wsi_edge_policy: how to handle tiles that extend beyond the WSI edge.
                - "drop": drop incomplete tiles
                - "pad_with_zeros": right/bottom pad incomplete tiles with zeros
                - "pad_with_edge": right/bottom pad incomplete tiles with edge pixel values
            roi_edge_policy: how to handle tiles that extend beyond the ROI bounds.
                - "read_from_image": expand region bounds to next full tile_size multiple, read from image,
                  and apply wsi_edge_policy to any incomplete tiles at the WSI edge.
                - "use_wsi_edge_policy": do not expand ROI bounds, read as-is from image,
                  and apply wsi_edge_policy to any incomplete tiles at the region or WSI edge.
            dtype: output patch dtype, e.g. np.uint8 or np.float32
            fallback_mode: strategy for selecting the pyramid level when the exact requested resolution
                is not available. Options: "nearest", "floor", "ceil", "error" (default), "resample".
                - "nearest": pick the level whose resolution is closest to the requested value.
                - "floor": pick the coarsest level at least as sharp as the requested resolution.
                - "ceil": pick the finest level no sharper than the requested resolution.
                - "error": raise an error if no level matches the requested resolution exactly.
                - "resample": like "ceil", but read from the finer level and downsample to the exact
                  requested resolution using OpenCV (interpolation set by resample_interpolation).
            resample_interpolation: interpolation method used when resampling regions to the requested
                resolution (only active when fallback_mode="resample").
                Options: "nearest", "linear", "cubic", "area", "lanczos" (default).
        """
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.dtype = dtype
        self.roi_edge_policy = roi_edge_policy
        self.wsi_edge_policy = wsi_edge_policy
        if resample_interpolation not in _CV2_INTERPOLATION:
            raise ValueError(
                f"Unknown resample_interpolation '{resample_interpolation}'. Choose from: {list(_CV2_INTERPOLATION)}"
            )
        self.resample_interpolation = resample_interpolation
        self.fallback_mode = fallback_mode

    def export_context(self, ctx: "PipelineContext") -> None:
        ctx["fallback_mode"] = self.fallback_mode

    def validate(self) -> None:
        self.ctx.require_key("tile_size")
        self.ctx.require_key("use_gpu")
        self.ctx.require_key("resolution")
        self.ctx.require_key("unit")

        if self.wsi_edge_policy not in {"drop", "pad_with_zeros", "pad_with_edge"}:
            raise ValueError(f"Unknown edge_policy '{self.wsi_edge_policy}'")

    def __call__(self, it: Iterable[RegionTask]) -> Iterable[CollatedPatchBatch]:
        xp = get_xp_backend(self.ctx["use_gpu"])
        tile_size = int(self.ctx["tile_size"])
        resolution = self.ctx["resolution"]
        unit = self.ctx["unit"]

        # Cache level selection per wsi_path: the selected level is the same for every
        # region chunk belonging to the same slide.
        _level_cache: dict = {}

        for task in it:
            x0, y0, w, h = task.region

            if self.roi_edge_policy == "read_from_image":
                # Compute the minimum region size needed so that every tile can be fully
                # sliced without running off the end of the read buffer.  Using
                # `w % tile_size` is fragile: when w happens to be a multiple of
                # tile_size the check is silently skipped even though the rightmost /
                # bottom-most tiles may still extend well beyond w (e.g. tile_size=256,
                # stride=192, ROI width=512 → last tile at x=384 needs 640 px but the
                # read window is only 512 wide).
                required_w = max(tx - x0 + tile_size for tx, ty in task.tiles)
                required_h = max(ty - y0 + tile_size for tx, ty in task.tiles)
                if w < required_w:
                    w = min(required_w, task.wsi_dims[0] - x0)
                if h < required_h:
                    h = min(required_h, task.wsi_dims[1] - y0)

            # Select the actual reading level for this slide (cached per wsi_path).
            if task.wsi_path not in _level_cache:
                actual_level = get_level_for_resolution(task.wsi_path, resolution, unit, self.fallback_mode)
                actual_ds = get_level_downsamples(task.wsi_path)[actual_level]
                _level_cache[task.wsi_path] = (actual_level, actual_ds)
            actual_level, actual_ds = _level_cache[task.wsi_path]

            # task.downsample is the virtual (target-resolution) downsample exported by WSIGrid:
            # virtual_ds = level-0 pixels per virtual pixel at the requested resolution.
            # Convert virtual (target-resolution) coordinates to level-0 for read_region(),
            # which all backends expect in level-0 space.
            target_ds = task.downsample
            x0_l0 = round(x0 * target_ds)
            y0_l0 = round(y0 * target_ds)

            # Compute the read dimensions at actual_level.
            # rf = target_ds / actual_ds:
            #   > 1.0 → actual level is finer than requested → read more pixels, then downsample.
            #   = 1.0 → exact match → no resize needed.
            #   < 1.0 → actual level is coarser than requested → read fewer pixels, then upsample.
            rf = target_ds / actual_ds
            _RF_TOLERANCE = 1e-6
            if abs(rf - 1.0) > _RF_TOLERANCE:
                read_w = round(w * rf)
                read_h = round(h * rf)
            else:
                rf = 1.0
                read_w, read_h = w, h

            region_img = read_region(
                task.wsi_path,
                x0_l0,
                y0_l0,
                read_w,
                read_h,
                actual_level,
                use_gpu=self.ctx["use_gpu"],
                num_workers_cucim=self.num_workers,
            )

            if rf != 1.0:
                # Resize from read dimensions back to virtual (requested-resolution) dimensions.
                # cv2.resize expects (width, height) as dsize and works on HxWxC arrays.
                cv2_interpolation = _CV2_INTERPOLATION[self.resample_interpolation]
                region_img = cv2.resize(region_img, (w, h), interpolation=cv2_interpolation)

            coords: List[Tuple[int, int]] = []
            patches: List[Union[np.ndarray, "cp.ndarray"]] = []

            for tx, ty in task.tiles:
                rx, ry = tx - x0, ty - y0
                patch = region_img[ry : ry + tile_size, rx : rx + tile_size, :]
                if patch.shape[:2] != (tile_size, tile_size):
                    if self.wsi_edge_policy == "drop":
                        continue
                    patch = self._pad_to_tile_size(patch, tile_size, xp)

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
        cpb = CollatedPatchBatch(task.wsi_id, np.asarray(coords), patches_xp, use_gpu=self.ctx["use_gpu"])
        for k, v in task.meta.items():
            cpb.add_meta_column(k, np.array([v] * len(coords)))
        return cpb

    def _pad_to_tile_size(self, patch, tile_size: int, xp_module):
        """Right/bottom pad to (tile_size, tile_size). Works for numpy or cupy arrays."""
        h, w = int(patch.shape[0]), int(patch.shape[1])
        pad_h = max(0, tile_size - h)
        pad_w = max(0, tile_size - w)
        if pad_h == 0 and pad_w == 0:
            return patch

        pad_spec = ((0, pad_h), (0, pad_w), (0, 0))

        if self.wsi_edge_policy == "pad_with_zeros":
            return np.pad(patch, pad_spec, mode="constant", constant_values=0)
        elif self.wsi_edge_policy == "pad_with_edge":
            return np.pad(patch, pad_spec, mode="edge")
        else:
            raise ValueError(f"Unknown edge_policy '{self.wsi_edge_policy}'")


class PatchExtractor(Stage):
    """
    PatchExtractor is a composite stage that combines TilePlanner, ReadWindowChunker, and RegionReadAndBatch.

    It takes slides (with or without ROIs) as input, and outputs batches of image patches.
    It handles tile planning, region chunking, reading, and batching internally.
    This is a convenience stage for common use cases where you want to extract patches from slides.
    """

    def __init__(
        self,
        *,
        tile_size: int,
        stride: int,
        tile_selection_mode: Literal["any_overlap", "full_inside_bounds", "center_in_roi"] = "any_overlap",
        max_batch_size: int = 200,
        num_workers: int = 8,
        wsi_edge_policy: Literal["drop", "pad_with_zeros", "pad_with_edge"] = "pad_with_zeros",
        roi_edge_policy: Literal["read_from_image", "use_wsi_edge_policy"] = "use_wsi_edge_policy",
        dtype: str = np.uint8,
        max_window_size: Optional[int] = None,
        resample_interpolation: Literal["nearest", "linear", "cubic", "area", "lanczos"] = "lanczos",
        fallback_mode: Literal["nearest", "floor", "ceil", "error", "resample"] = "error",
    ):
        """
        Args:
            tile_size: size of square patches to extract
            stride: spacing between patch top-left corners of each of the patches
            tile_selection_mode: how to select tiles with respect to ROIs.
            max_batch_size: Maximum number of patches per output batch
            num_workers: number of parallel workers for reading WSI images using cuCIM
            wsi_edge_policy: how to handle tiles that extend beyond the WSI edge.
                - "drop": drop incomplete tiles
                - "pad_with_zeros": right/bottom pad incomplete tiles with zeros
                - "pad_with_edge": right/bottom pad incomplete tiles with edge pixel values
            roi_edge_policy: how to handle tiles that extend beyond the ROI bounds.
                - "read_from_image": expand region bounds to next full tile_size multiple, read from image,
                  and apply wsi_edge_policy to any incomplete tiles at the WSI edge.
                - "use_wsi_edge_policy": do not expand ROI bounds, read as-is from image,
                  and apply wsi_edge_policy to any incomplete tiles at the region or WSI edge.
            dtype: output patch dtype, e.g. np.uint8 or np.float32
            max_window_size: maximum size of square read windows. If None, defaults to 20*tile_size.
            resample_interpolation: interpolation method used when resampling regions to the requested
                resolution (only active when fallback_mode="resample").
                Options: "nearest", "linear", "cubic", "area", "lanczos" (default).
            fallback_mode: strategy for selecting the pyramid level when the exact requested resolution
                is not available. Options: "nearest", "floor", "ceil", "error" (default), "resample".
                - "nearest": pick the level whose resolution is closest to the requested value.
                - "floor": pick the coarsest level at least as sharp as the requested resolution.
                - "ceil": pick the finest level no sharper than the requested resolution.
                - "error": raise an error if no level matches the requested resolution exactly.
                - "resample": like "ceil", but read from the finer level and downsample to the exact
                  requested resolution using OpenCV (interpolation set by resample_interpolation).
        """
        self.params = dict(
            tile_size=tile_size,
            stride=stride,
            tile_selection_mode=tile_selection_mode,
            max_window_size=max_window_size,
            batch_size=max_batch_size,
            num_workers=num_workers,
            dtype=dtype,
            wsi_edge_policy=wsi_edge_policy,
            roi_edge_policy=roi_edge_policy,
            resample_interpolation=resample_interpolation,
            fallback_mode=fallback_mode,
        )

        # Internal stages (created now; context attached later)
        self._tp = TilePlanner(tile_size=tile_size, stride=stride, tile_selection_mode=tile_selection_mode)
        self._rwc = ReadWindowChunker(max_window_size=max_window_size)
        self._rbb = RegionReadAndBatch(
            batch_size=max_batch_size,
            num_workers=num_workers,
            dtype=dtype,
            wsi_edge_policy=wsi_edge_policy,
            roi_edge_policy=roi_edge_policy,
            resample_interpolation=resample_interpolation,
            fallback_mode=fallback_mode,
        )
        self._substages: List[Stage] = [self._tp, self._rwc, self._rbb]

    def export_context(self, ctx) -> None:
        for s in self._substages:
            s.export_context(ctx)

    def attach_context(self, ctx) -> None:
        super().attach_context(ctx)  # gives self.ctx, self.log
        # Propagate context to inner stages
        for s in self._substages:
            s.attach_context(ctx)

    def validate(self) -> None:
        # Export context to inner stages, then validate each
        for s in self._substages:
            s.export_context(self.ctx)
        for s in self._substages:
            s.validate()

    def __call__(self, it: Iterable[Union[Slide, SlideWithROIs]]) -> Iterable[CollatedPatchBatch]:
        stream = it
        for s in self._substages:
            stream = s(stream)
        return stream
