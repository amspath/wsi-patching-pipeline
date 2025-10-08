from typing import Iterable

from wsi_patching.backends.cupy_numpy import ensure_array_matches_use_gpu, ensure_numpy, get_xp_backend
from wsi_patching.core.pipeline import Stage
from wsi_patching.core.types.types import CollatedPatchBatch


class LowContrastBackgroundFilter(Stage):
    """
    Drop patches whose grayscale dynamic range (max - min) is below a threshold.

    Assumptions:
      - `patches` are uint8, or float32 scaled in [0, 255].
      - If ctx['use_gpu'] is True, `patches` is a CuPy array; otherwise NumPy.
      - All computation stays on the active xp backend; we convert to NumPy only
        at pipeline boundaries (`add_col`, `filter`) via `ensure_numpy`.

    Method (per batch):
      1) Convert RGB/RGBA to grayscale in [0,1] (float32/float64).
      2) Compute per-patch dynamic range: max(gray) - min(gray).
      3) Keep patch iff range >= `range_threshold`.
    """

    def __init__(
        self,
        *,
        range_threshold: float = 0.2,  # threshold in grayscale [0,1] units
        float_precision: str = "float32",  # working precision for grayscale conversion
    ):
        """
        Parameters
        ----------
        range_threshold : float in [0,1], default 0.2 (20%)
            Minimum required grayscale dynamic range (max - min) to keep a patch.

        float_precision : {"float32","float64"}, default "float32"
            Precision for grayscale math. "float32" is typically sufficient/faster.
        """
        if not (0.0 <= range_threshold <= 1.0):
            raise ValueError("range_threshold must be in [0, 1]")
        if float_precision not in ("float32", "float64"):
            raise ValueError("float_precision must be 'float32' or 'float64'")

        self.range_threshold = float(range_threshold)
        self.float_precision = float_precision

    def validate(self):
        """No-op: backend is resolved per call from ctx['use_gpu']."""
        self.ctx.require_key("use_gpu")

    def __call__(self, it: Iterable[CollatedPatchBatch]) -> Iterable[CollatedPatchBatch]:
        """
        For each batch:
          - Ensure dtype/backend
          - Convert to grayscale in [0,1]
          - Compute per-patch range and filter
          - Attach stats and yield
        """
        use_gpu = bool(self.ctx["use_gpu"])
        xp = get_xp_backend(use_gpu)

        thr = xp.asarray(self.range_threshold, dtype=xp.float32 if self.float_precision == "float32" else xp.float64)

        for collated_patch_batch in it:
            patches = ensure_array_matches_use_gpu(collated_patch_batch.patches, use_gpu)

            # 1) RGB/RGBA -> grayscale in [0,1]
            gray = self._rgb_to_gray_xp(patches, xp)

            # 2) Per-patch min/max and dynamic range
            B = gray.shape[0]
            gray_flat = gray.reshape(B, -1)
            g_min = gray_flat.min(axis=1)  # (B,)
            g_max = gray_flat.max(axis=1)  # (B,)
            g_range = (g_max - g_min).astype(gray.dtype)  # (B,)

            # 3) Keep iff range >= threshold
            keep_batch_mask = g_range >= thr

            # Record stats + filter (convert to NumPy only at the boundary)
            collated_patch_batch.add_meta_column("gray_min", ensure_numpy(g_min))
            collated_patch_batch.add_meta_column("gray_max", ensure_numpy(g_max))
            collated_patch_batch.add_meta_column("gray_range", ensure_numpy(g_range))
            collated_patch_batch.add_meta_column("range_threshold", ensure_numpy(xp.full((B,), thr, dtype=gray.dtype)))
            collated_patch_batch.filter_on_mask(ensure_numpy(keep_batch_mask))

            self.log.info(
                f"wsi={collated_patch_batch.wsi_id} "
                f"batch_out={len(collated_patch_batch.patches)} (range_threshold={self.range_threshold})"
            )

            if len(collated_patch_batch.patches) == 0:
                continue

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
