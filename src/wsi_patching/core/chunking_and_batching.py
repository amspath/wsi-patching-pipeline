import math
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np

from wsi_patching.backends.cupy_numpy import get_xp_backend
from wsi_patching.backends.fastslide import read_region
from wsi_patching.core.pipeline import Stage
from wsi_patching.core.types.types import CollatedPatchBatch, RegionTask, Slide, SlideWithROIs, TilePlan
from wsi_patching.regions_of_interest.roi_providers import WholeSlideProvider
from wsi_patching.regions_of_interest.rois import ROI, BoxROI

_CV2_INTERPOLATION = {
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

                tiles = self._generate_tiles_vectorized(roi, bx, by, bw, bh, x1, y1, tile_size, stride)

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

    def _generate_tiles_vectorized(
        self, roi: ROI, bx: int, by: int, bw: int, bh: int, x1: int, y1: int, tile_size: int, stride: int
    ) -> List[Tuple[int, int]]:
        """Generate accepted tile positions using numpy operations where possible.

        For BoxROI, acceptance checks are fully vectorized. For other ROI types,
        grid generation is vectorized but per-tile filtering falls back to scalar.
        """
        last_tile_correction = max(0, tile_size - stride)

        # Generate x positions
        roi_w = x1 - bx
        if roi_w <= tile_size:
            xs = np.array([bx], dtype=np.int64)
        else:
            xs = np.arange(bx, x1 - last_tile_correction, stride, dtype=np.int64)

        # Generate y positions
        roi_h = y1 - by
        if roi_h <= tile_size:
            ys = np.array([by], dtype=np.int64)
        else:
            ys = np.arange(by, y1 - last_tile_correction, stride, dtype=np.int64)

        # Build full grid of (tx, ty) pairs
        gx, gy = np.meshgrid(xs, ys)
        tx = gx.ravel()
        ty = gy.ravel()

        if isinstance(roi, BoxROI):
            mode = self.tile_selection_mode
            if mode == "any_overlap":
                # All positions are within ROI bounding box by construction; top-left
                # corner is always inside, so intersects_patch is always True.
                pass  # tx/ty already the full set
            elif mode == "full_inside_bounds":
                mask = (tx + tile_size <= bx + bw) & (ty + tile_size <= by + bh)
                tx, ty = tx[mask], ty[mask]
            elif mode == "center_in_roi":
                cx = tx + tile_size / 2.0
                cy = ty + tile_size / 2.0
                mask = (cx < bx + bw) & (cy < by + bh)
                tx, ty = tx[mask], ty[mask]
            return list(zip(tx.tolist(), ty.tolist()))

        # Non-BoxROI: scalar fallback for per-tile acceptance
        result: List[Tuple[int, int]] = []
        for txx, tyy in zip(tx.tolist(), ty.tolist()):
            if self._accept_tile(roi, txx, tyy, tile_size):
                result.append((txx, tyy))
        return result

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
        mws = self.max_window_size
        assert mws is not None

        for plan in it:
            bx, by, bw, bh = plan.roi_bounds
            W, H = plan.dims
            if not plan.tiles:
                self.log.warning(f"No tiles in plan for slide {plan.wsi_id} ROI {plan.roi_index}")
                continue

            x_roi_end, y_roi_end = min(bx + bw, W), min(by + bh, H)

            # Number of base windows along each axis
            n_x_win = max(1, math.ceil((x_roi_end - bx) / mws))
            n_y_win = max(1, math.ceil((y_roi_end - by) / mws))

            # Assign each tile to a window index using numpy vectorised division
            tiles_arr = np.array(plan.tiles, dtype=np.int64)  # (N, 2)
            win_x = np.clip((tiles_arr[:, 0] - bx) // mws, 0, n_x_win - 1)
            win_y = np.clip((tiles_arr[:, 1] - by) // mws, 0, n_y_win - 1)

            # Stable sort by combined window key so groups come out in scan order
            win_key = win_y * n_x_win + win_x
            order = np.argsort(win_key, kind="stable")
            sorted_keys = win_key[order]

            boundaries = np.where(np.diff(sorted_keys) != 0)[0] + 1
            groups = np.split(order, boundaries)
            unique_keys = sorted_keys[np.concatenate([[0], boundaries])]

            for group_indices, ukey in zip(groups, unique_keys.tolist()):
                wx = int(ukey % n_x_win)
                wy = int(ukey // n_x_win)

                x_region_start = bx + wx * mws
                y_region_start = by + wy * mws

                # Base window size (defines ownership; READ window is expanded by overlap)
                base_w = min(mws, x_roi_end - x_region_start)
                base_h = min(mws, y_roi_end - y_region_start)
                read_w = min(base_w + overlap, x_roi_end - x_region_start)
                read_h = min(base_h + overlap, y_roi_end - y_region_start)

                in_window: List[Tuple[int, int]] = [tuple(t) for t in tiles_arr[group_indices].tolist()]

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
      - read the entire region once
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
        wsi_edge_policy: Literal["drop", "pad_with_zeros", "pad_with_edge"] = "pad_with_zeros",
        roi_edge_policy: Literal["read_from_image", "use_wsi_edge_policy"] = "use_wsi_edge_policy",
        dtype: Any = np.uint8,
    ):
        """
        Args:
            batch_size: Maximum number of patches per output batch
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
        """
        self.batch_size = int(batch_size)
        self.dtype = dtype
        self.roi_edge_policy = roi_edge_policy
        self.wsi_edge_policy = wsi_edge_policy

    def validate(self) -> None:
        self.ctx.require_key("tile_size")
        self.ctx.require_key("use_gpu")

        if self.wsi_edge_policy not in {"drop", "pad_with_zeros", "pad_with_edge"}:
            raise ValueError(f"Unknown edge_policy '{self.wsi_edge_policy}'")

    def __call__(self, it: Iterable[RegionTask]) -> Iterable[CollatedPatchBatch]:
        xp = get_xp_backend(self.ctx["use_gpu"])
        tile_size = int(self.ctx["tile_size"])

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

            # Convert level-N coordinates to level-0 for the read_region() call, which
            # backend expects in level-0 space.
            # round() gives the nearest integer, which is correct for both integer and
            # non-integer downsample factors.
            ds = task.downsample
            x0_l0 = round(x0 * ds)
            y0_l0 = round(y0 * ds)

            rf = task.resample_factor
            if rf != 1.0:
                read_w = round(w * rf)
                read_h = round(h * rf)
            else:
                read_w, read_h = w, h

            region_img = read_region(task.wsi_path, x0_l0, y0_l0, read_w, read_h, task.level)

            if rf != 1.0:
                interp = _CV2_INTERPOLATION[self.ctx.get("resample_interpolation", "lanczos")]
                region_img = cv2.resize(region_img, (w, h), interpolation=interp)

            n_tiles = len(task.tiles)
            buf_size = min(n_tiles, self.batch_size)
            batch_buf = np.empty((buf_size, tile_size, tile_size, 3), dtype=np.uint8)
            coords_buf = np.empty((buf_size, 2), dtype=np.int64)
            fill_idx = 0

            for tx, ty in task.tiles:
                rx, ry = tx - x0, ty - y0
                patch = region_img[ry : ry + tile_size, rx : rx + tile_size, :]
                if patch.shape[:2] != (tile_size, tile_size):
                    if self.wsi_edge_policy == "drop":
                        continue
                    patch = self._pad_to_tile_size(patch, tile_size, xp)

                batch_buf[fill_idx] = patch
                coords_buf[fill_idx, 0] = tx
                coords_buf[fill_idx, 1] = ty
                fill_idx += 1

                if fill_idx >= self.batch_size:
                    yield self._make_batch_from_buf(task, coords_buf[:fill_idx], batch_buf[:fill_idx], xp)
                    fill_idx = 0

            if fill_idx > 0:
                yield self._make_batch_from_buf(task, coords_buf[:fill_idx], batch_buf[:fill_idx], xp)

    def _make_batch_from_buf(self, task, coords_slice: np.ndarray, batch_slice: np.ndarray, xp):
        # Both .copy() calls are mandatory: xp.asarray on CPU with matching dtype
        # returns a view of the buffer, which is reused on the next batch iteration.
        coords_out = coords_slice.copy()
        patches_xp = xp.asarray(batch_slice.copy(), dtype=self.dtype)
        cpb = CollatedPatchBatch(task.wsi_id, coords_out, patches_xp, use_gpu=self.ctx["use_gpu"])
        for k, v in task.meta.items():
            cpb.add_meta_column(k, np.array([v] * len(coords_out)))
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
        wsi_edge_policy: Literal["drop", "pad_with_zeros", "pad_with_edge"] = "pad_with_zeros",
        roi_edge_policy: Literal["read_from_image", "use_wsi_edge_policy"] = "use_wsi_edge_policy",
        dtype: Any = np.uint8,
        max_window_size: Optional[int] = None,
    ):
        """
        Args:
            tile_size: size of square patches to extract
            stride: spacing between patch top-left corners of each of the patches
            tile_selection_mode: how to select tiles with respect to ROIs.
            max_batch_size: Maximum number of patches per output batch
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
        """
        self.params = dict(
            tile_size=tile_size,
            stride=stride,
            tile_selection_mode=tile_selection_mode,
            max_window_size=max_window_size,
            batch_size=max_batch_size,
            dtype=dtype,
            wsi_edge_policy=wsi_edge_policy,
            roi_edge_policy=roi_edge_policy,
        )

        # Internal stages (created now; context attached later)
        self._tp = TilePlanner(tile_size=tile_size, stride=stride, tile_selection_mode=tile_selection_mode)
        self._rwc = ReadWindowChunker(max_window_size=max_window_size)
        self._rbb = RegionReadAndBatch(
            batch_size=max_batch_size, dtype=dtype, wsi_edge_policy=wsi_edge_policy, roi_edge_policy=roi_edge_policy
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
        return stream  # type: ignore
