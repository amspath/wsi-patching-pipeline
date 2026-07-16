from typing import Iterable

from wsi_patching.backends.cupy_numpy import ensure_array_matches_use_gpu, ensure_numpy, get_xp_backend
from wsi_patching.core.pipeline import Stage
from wsi_patching.core.types.types import CollatedPatchBatch
from wsi_patching.utils.audit import Knob


class OtsuFilter(Stage):
    """
    Pure-xp (NumPy/CuPy) batched Otsu thresholding tissue filter.

    Assumptions:
      - `patches` are uint8, or float32 scaled in [0, 255].
      - If ctx['use_gpu'] is True, `patches` is a CuPy array; otherwise NumPy.
      - All computation stays on the active xp backend; we convert to NumPy only
        at pipeline boundaries (`add_col`, `filter`) via `ensure_numpy`.

    Method (per batch):
      1) Convert RGB/RGBA to grayscale in [0,1]
      2) Compute one Otsu threshold per patch (batched histograms).
      3) Build tissue mask by comparing gray to the threshold:
         - if `tissue_is_darker=True`:  tissue := gray < thr   (common for H&E)
         - if `tissue_is_darker=False`: tissue := gray >= thr
      4) Optionally drop patches with insufficient tissue coverage.
    """

    # `tissue_fraction` does not depend on `min_tissue_fraction`, so the audit report
    # can re-decide keep/drop for any threshold without re-running. `num_bins` and
    # `tissue_is_darker` change the metric itself, so they get no slider.
    AUDIT_KNOBS = (Knob("min_tissue_fraction", "tissue_fraction", ">=", permissive=0.0),)

    def __init__(
        self,
        *,
        tissue_is_darker: bool = True,
        num_bins: int = 256,
        float_precision: str = "float32",
        min_tissue_fraction: float = 0.0,
    ):
        """
        Parameters
        ----------
        tissue_is_darker : bool, default True
            Defines the tissue polarity relative to the Otsu threshold.
            - True  -> classify pixels darker than the threshold as tissue (gray < thr).
            - False -> classify pixels brighter/equal as tissue (gray >= thr).

        num_bins : int, default 256
            Number of histogram bins for Otsu. Must be >= 2. For 8-bit images,
            256 is typical. Larger values may add noise; smaller values coarsen thresholds.

        float_precision : {"float32","float64"}, default "float32"
            Working precision for grayscale + histogram math. "float32" is usually
            sufficient for uint8 inputs and is faster. "float64" will match skimage’s
            historical behavior more closely.

        min_tissue_fraction : float in [0,1], default 0.0
            Minimum required fraction of tissue pixels per patch. Patches below this
            fraction are dropped. Set to 0.0 to keep all patches.
        """
        if num_bins < 2:
            raise ValueError("num_bins must be >= 2")
        if float_precision not in ("float32", "float64"):
            raise ValueError("float_precision must be 'float32' or 'float64'")
        if not (0.0 <= min_tissue_fraction <= 1.0):
            raise ValueError("min_tissue_fraction must be in [0, 1]")

        self.tissue_is_darker = bool(tissue_is_darker)
        self.num_bins = int(num_bins)
        self.float_precision = float_precision
        self.min_tissue_fraction = float(min_tissue_fraction)

    def validate(self):
        """No-op: backend is resolved per call from ctx['use_gpu']."""
        self.ctx.require_key("use_gpu")

    def __call__(self, it: Iterable[CollatedPatchBatch]) -> Iterable[CollatedPatchBatch]:
        """
        For each batch, compute Otsu mask, attach coverage stats, optionally filter, then yield.
        """
        use_gpu = bool(self.ctx["use_gpu"])
        xp = get_xp_backend(use_gpu)

        for collated_patch_batch in it:
            in_sz = len(collated_patch_batch.patches)

            patches = ensure_array_matches_use_gpu(collated_patch_batch.patches, use_gpu)

            # 1) RGB/RGBA -> grayscale in [0,1]
            gray = self._rgb_to_gray_xp(patches, xp)

            # 2) Per-patch Otsu thresholds (in [0,1])
            thresholds = self._batched_otsu_thresholds_xp(gray, self.num_bins, xp)

            # 3) Tissue mask based on desired polarity
            if self.tissue_is_darker:
                tissue_mask = gray < thresholds[:, None, None]
            else:
                tissue_mask = gray >= thresholds[:, None, None]

            # 4) Coverage stats per patch
            B = tissue_mask.shape[0]
            tissue_pixel_count = tissue_mask.reshape(B, -1).sum(axis=1)
            total_pixels = tissue_mask.shape[1] * tissue_mask.shape[2]
            tissue_fraction = tissue_pixel_count.astype(xp.float32) / float(total_pixels)

            # Optional filtering
            if self.min_tissue_fraction > 0.0:
                keep_batch_mask = tissue_fraction >= self.min_tissue_fraction
            else:
                keep_batch_mask = xp.ones((B,), dtype=bool)

            # Record stats + filter (convert to NumPy only at the boundary)
            collated_patch_batch.add_meta_column("otsu_threshold", ensure_numpy(thresholds))
            collated_patch_batch.add_meta_column("tissue_fraction", ensure_numpy(tissue_fraction))
            collated_patch_batch.add_meta_column("tissue_pixel_count", ensure_numpy(tissue_pixel_count))
            collated_patch_batch.filter_on_mask(ensure_numpy(keep_batch_mask))

            self.log.info(
                f"batch_in={in_sz} -> "
                f"batch_out={len(collated_patch_batch.patches)} "
                f"(tissue_is_darker={self.tissue_is_darker}, num_bins={self.num_bins}, "
                f"float_precision={self.float_precision}, min_tissue_fraction={self.min_tissue_fraction})"
            )
            yield collated_patch_batch

    # ---- helpers (xp-native) ----
    def _rgb_to_gray_xp(self, imgs, xp):
        """
        Convert (B,H,W,C) to grayscale (B,H,W) in [0,1] using luminance weights.
        Supports C=1 (already gray), C=3 (RGB), C=4 (RGBA → use first 3).
        Output dtype honors `float_precision`.
        """
        if imgs.ndim != 4:
            raise ValueError(f"Expected 4D (B,H,W,C); got {imgs.shape}")
        C = imgs.shape[-1]
        if C not in (1, 3, 4):
            raise ValueError(f"Expected channels 1,3,4; got {C}")

        prec = xp.float32 if self.float_precision == "float32" else xp.float64

        if imgs.dtype == xp.uint8:
            f = imgs.astype(prec, copy=False) / prec(255.0)
        else:
            # Accept float32/float64 (0..255) or other numeric types
            f = imgs.astype(prec, copy=False)
            f = xp.clip(f, prec(0.0), prec(255.0)) / prec(255.0)

        if C == 1:
            return f[..., 0]

        rgb = f[..., :3]
        # ITU-R BT.709-ish weights (close to skimage defaults)
        coeffs = xp.asarray([0.2125, 0.7154, 0.0721], dtype=prec)
        return (rgb * coeffs).sum(axis=-1, dtype=prec)

    def _batched_otsu_thresholds_xp(self, gray, num_bins: int, xp):
        """
        Batched Otsu thresholds on grayscale images in [0,1].

        Parameters
        ----------
        gray : (B,H,W) float32/float64
        num_bins : int

        Returns
        -------
        thresholds : (B,) same dtype as `gray`, each in [0,1] (gray domain)
        """
        B, _, _ = gray.shape
        gray_flat = gray.reshape(B, -1)

        # Per-image normalization to [0,1] (robust to constants)
        g_min = gray_flat.min(axis=1)
        g_max = gray_flat.max(axis=1)
        is_constant = g_max <= g_min

        eps = xp.asarray(1e-12, dtype=gray.dtype)
        span = xp.maximum(g_max - g_min, eps)
        gray_norm = (gray - g_min[:, None, None]) / span[:, None, None]
        xp.clip(gray_norm, 0.0, 1.0, out=gray_norm)

        # Quantize normalized gray into bins (ensure max maps into last bin)
        tiny = xp.asarray(1e-7, dtype=gray.dtype)
        bin_idx = xp.floor(gray_norm * num_bins - tiny).astype(xp.int32)
        xp.clip(bin_idx, 0, num_bins - 1, out=bin_idx)

        # Batched histograms via offset bincount
        bin_idx_flat = bin_idx.reshape(B, -1).astype(xp.int64, copy=False)
        offsets = (num_bins * xp.arange(B, dtype=xp.int64))[:, None]
        keys = (bin_idx_flat + offsets).ravel()
        counts_flat = xp.bincount(keys, minlength=B * num_bins)
        hist = counts_flat.reshape(B, num_bins).astype(gray.dtype, copy=False)

        # Bin centers in normalized space
        centers = (xp.arange(num_bins, dtype=gray.dtype) + 0.5) / float(num_bins)

        # Cumulative class weights/sums
        w1 = xp.cumsum(hist, axis=1)  # (B, num_bins)
        total_w = w1[:, -1:]  # (B, 1)
        sum1 = xp.cumsum(hist * centers[None, :], axis=1)
        total_sum = sum1[:, -1:]

        w2 = total_w - w1
        sum2 = total_sum - sum1

        # Safe means (0 where class empty)
        zero = xp.asarray(0, dtype=gray.dtype)
        m1 = xp.where(w1 > 0, sum1 / xp.maximum(w1, 1), zero)
        m2 = xp.where(w2 > 0, sum2 / xp.maximum(w2, 1), zero)

        # Between-class variance for thresholds between bins
        var_between = w1[:, :-1] * w2[:, 1:] * (m1[:, :-1] - m2[:, 1:]) ** 2

        # Best threshold in normalized space, then map back to original gray domain
        best_idx = xp.argmax(var_between, axis=1)
        thr_norm = ((best_idx.astype(gray.dtype) + 0.5) / float(num_bins)).astype(gray.dtype)
        thresholds = g_min.astype(gray.dtype) + thr_norm * span.astype(gray.dtype)

        # For constant images, use the (degenerate) gray value
        thresholds = xp.where(is_constant, g_min.astype(gray.dtype), thresholds)
        return thresholds
