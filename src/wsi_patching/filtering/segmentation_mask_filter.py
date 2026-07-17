from typing import Dict, Iterable, Optional, Tuple, Union

import cv2
import numpy as np

from wsi_patching.core.pipeline import Stage
from wsi_patching.core.types.types import CollatedPatchBatch

# (start_x, start_y, scale_x, scale_y): slide-space origin of mask pixel (0, 0)
# and slide pixels per mask pixel along each axis.
Placement = Tuple[float, float, float, float]


class SegmentationMaskFilter(Stage):
    """Drop patches that fall outside a per-slide segmentation mask (e.g. a tumor ROI).

    For each patch at top-left ``(x, y)`` the tile footprint is projected into the
    mask's pixel space and the patch is kept iff that footprint covers at least
    ``min_foreground_pixels`` pixels equal to ``foreground_value``.

    Put this stage *before* an expensive transform/encoder (e.g. a stain normalizer
    or a foundation encoder) so the costly work only runs on the patches you keep.

    Coordinate mapping
    ------------------
    A patch coord ``(x, y)`` maps to mask pixel
    ``((x - start_x) / scale_x, (y - start_y) / scale_y)`` where ``(scale_x, scale_y)``
    are *slide pixels per mask pixel*. The placement
    ``(start_x, start_y, scale_x, scale_y)`` is resolved per slide, in priority order:

    1. ``wsi_placements[wsi_id]`` when provided -- use this when the mask covers a
       region that is **not** the patched ROI (e.g. a tissue-detected sub-window whose
       geometry lives in an external metadata file). Compute it yourself and pass it in.
    2. otherwise it is derived from the batch ``roi_bounds`` metadata (seeded by the
       patch planner as ``(x, y, w, h)``): ``start = (x, y)`` and
       ``scale = (w / mask_W, h / mask_H)`` -- i.e. the mask is assumed to span the
       patched ROI exactly.

    The footprint size uses truncating division ``int(tile_size / scale)``, matching a
    simple ``mask[y:y+ps, x:x+ps]`` slice test; if a mask is downsampled so hard that a
    tile is sub-pixel (``int(...) == 0``) that tile is dropped.

    Masks
    -----
    Supplied per slide via ``wsi_to_mask`` keyed by ``wsi_id`` (the slide file stem, the
    same id WSIGrid emits). Each value is either a path to a grayscale image or a
    preloaded 2-D uint8 ``np.ndarray`` -- pass an array when the caller already applies
    its own loading / fallback policy. The binarized mask's summed-area table is built
    once per slide and cached.

    Parameters
    ----------
    wsi_to_mask : dict[str, str | np.ndarray]
        Mask source per slide id.
    wsi_placements : dict[str, tuple], optional
        Per-slide ``(start_x, start_y, scale_x, scale_y)`` overrides (see above).
    foreground_value : int, default 255
        Mask pixel value treated as "inside the ROI".
    min_foreground_pixels : int, default 1
        Minimum foreground pixels a tile footprint must cover to be kept.
    """

    def __init__(
        self,
        wsi_to_mask: Dict[str, Union[str, np.ndarray]],
        *,
        wsi_placements: Optional[Dict[str, Placement]] = None,
        foreground_value: int = 255,
        min_foreground_pixels: int = 1,
    ):
        self.wsi_to_mask = wsi_to_mask
        self.wsi_placements = dict(wsi_placements) if wsi_placements else {}
        self.foreground_value = int(foreground_value)
        if min_foreground_pixels < 1:
            raise ValueError("min_foreground_pixels must be >= 1")
        self.min_foreground_pixels = int(min_foreground_pixels)
        # per-slide summed-area table of the binarized mask; shape (H+1, W+1)
        self._integral_cache: Dict[str, np.ndarray] = {}

    def validate(self) -> None:
        self.ctx.require_key("tile_size")

    def __call__(self, it: Iterable[CollatedPatchBatch]) -> Iterable[CollatedPatchBatch]:
        tile_size = int(self.ctx["tile_size"])

        for batch in it:
            integral = self._integral_for(batch.wsi_id)
            mask_h = integral.shape[0] - 1
            mask_w = integral.shape[1] - 1

            start_x, start_y, scale_x, scale_y = self._placement_for(batch, mask_w, mask_h)
            keep = self._keep_mask(
                batch.coords, tile_size, integral, start_x, start_y, scale_x, scale_y
            )

            in_sz = len(batch.patches)
            batch.filter_on_mask(keep)
            self.log.info(
                f"wsi={batch.wsi_id} batch_in={in_sz} batch_out={len(batch.patches)} "
                f"(SegmentationMaskFilter)"
            )

            if len(batch.patches) == 0:
                continue

            yield batch

    # ---- helpers ----
    def _integral_for(self, wsi_id: str) -> np.ndarray:
        cached = self._integral_cache.get(wsi_id)
        if cached is not None:
            return cached

        if wsi_id not in self.wsi_to_mask:
            raise KeyError(f"SegmentationMaskFilter: no mask provided for wsi_id '{wsi_id}'")

        src = self.wsi_to_mask[wsi_id]
        if isinstance(src, np.ndarray):
            mask = src
        else:
            mask = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"SegmentationMaskFilter: could not read mask at '{src}'")
        if mask.ndim != 2:
            raise ValueError(
                f"SegmentationMaskFilter: mask for '{wsi_id}' must be 2-D grayscale, "
                f"got shape {mask.shape}"
            )

        binary = (mask == self.foreground_value).astype(np.int64)
        # Summed-area table with a zero top/left border -> integral[a, b] is the sum of
        # binary[:a, :b], so any rectangle sum is 4 lookups (O(1) per patch).
        integral = np.zeros((binary.shape[0] + 1, binary.shape[1] + 1), dtype=np.int64)
        np.cumsum(np.cumsum(binary, axis=0), axis=1, out=integral[1:, 1:])

        self._integral_cache[wsi_id] = integral
        return integral

    def _placement_for(self, batch: CollatedPatchBatch, mask_w: int, mask_h: int) -> Placement:
        placement = self.wsi_placements.get(batch.wsi_id)
        if placement is not None:
            return (float(placement[0]), float(placement[1]), float(placement[2]), float(placement[3]))

        roi = batch.metadata.get("roi_bounds")
        if roi is None:
            raise KeyError(
                "SegmentationMaskFilter needs either a wsi_placements entry or 'roi_bounds' "
                "metadata (seeded by WSIGrid) to place the mask over the slide."
            )
        rx, ry, rw, rh = (float(v) for v in roi[0])  # (x, y, w, h), shared across the batch
        if rw <= 0 or rh <= 0:
            raise ValueError(
                f"SegmentationMaskFilter: non-positive roi_bounds {tuple(roi[0])} for '{batch.wsi_id}'"
            )
        return rx, ry, rw / mask_w, rh / mask_h

    def _keep_mask(
        self,
        coords: np.ndarray,
        tile_size: int,
        integral: np.ndarray,
        start_x: float,
        start_y: float,
        scale_x: float,
        scale_y: float,
    ) -> np.ndarray:
        mask_h = integral.shape[0] - 1
        mask_w = integral.shape[1] - 1

        x = coords[:, 0].astype(np.float64)
        y = coords[:, 1].astype(np.float64)

        # Tile footprint in mask pixels (truncating, to match a plain array-slice test).
        psx = int(tile_size / scale_x)
        psy = int(tile_size / scale_y)

        # Keep-condition is evaluated on the original (pre-truncation) coords so that a
        # coord just left/above the ROI is excluded rather than snapping to pixel 0.
        inside = (
            (x >= start_x)
            & (x < start_x + mask_w * scale_x)
            & (y >= start_y)
            & (y < start_y + mask_h * scale_y)
        )

        mx = ((x - start_x) / scale_x).astype(np.int64)
        my = ((y - start_y) / scale_y).astype(np.int64)

        x0 = np.clip(mx, 0, mask_w)
        y0 = np.clip(my, 0, mask_h)
        x1 = np.clip(mx + psx, 0, mask_w)
        y1 = np.clip(my + psy, 0, mask_h)

        counts = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]

        keep = inside & (x1 > x0) & (y1 > y0) & (counts >= self.min_foreground_pixels)
        return np.asarray(keep, dtype=np.bool_)
