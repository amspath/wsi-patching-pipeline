from typing import Iterable, Optional, Tuple

from wsi_patching.backends.cupy_numpy import get_xp_backend
from wsi_patching.core.pipeline import Stage
from wsi_patching.core.types.types import CollatedPatchBatch


class MacenkoNormalizer(Stage):
    """
    Macenko Normalization stage.

    - Fits (H, max_sat) on the first batch (stays on that batch's backend)
    - Applies to all subsequent batches without leaving the backend
    - Adds meta_col 'macenko_normalized' (bool)
    """

    def __init__(
        self, alpha: float = 1.0, beta: float = 0.15, light_intensity: int = 255, pixel_limit: Optional[int] = 500_000
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.I0 = int(light_intensity)
        self.pixel_limit = pixel_limit

        self._fitted = False
        self._H_mat = None  # xp.ndarray (3,2)
        self._max_sat = None  # xp.ndarray (2,1)
        self._xp = None  # module: numpy or cupy

        self.log.info(
            "MacenkoNormalizer fits on the first batch of each image. If the first batch is background-only, "
            "fitting will likely result in unwanted behavior. Use with caution."
        )

    def validate(self):
        self.ctx.require_key("use_gpu")

    def __call__(self, it: Iterable[CollatedPatchBatch]) -> Iterable[CollatedPatchBatch]:
        for batch in it:
            if not self._fitted:
                self.fit(batch)

            patches = _ensure_rgb_uint8_nhwc(batch.patches, self._xp)
            N, Hh, Ww, _ = patches.shape

            flat = patches.reshape(-1, 3)
            OD = _to_OD(flat, self.I0, self._xp)  # (N*H*W, 3)

            C, *_ = self._xp.linalg.lstsq(self._H_mat, OD.T, rcond=None)

            denom = self._xp.clip(self._max_sat.reshape(-1, 1), 1e-6, None)
            C_norm = self._xp.clip(C / denom, 0.0, 1.0)

            OD_hat = (self._H_mat @ C_norm).T  # (N*H*W, 3)
            normalized = _from_OD(OD_hat, self.I0, self._xp).reshape(N, Hh, Ww, 3)

            batch.patches = normalized

            self.log.info(f"Yielded batch for wsi='{batch.wsi_id}' size={normalized.shape[0]}")
            yield batch

    def fit(self, batch: CollatedPatchBatch) -> None:
        if self._fitted:
            return

        self._xp = get_xp_backend(self.ctx["use_gpu"])

        H_mat, max_sat = self._macenko_fit_from_batch_xp(patches_u8_nhwc=batch.patches)

        self._H_mat = H_mat
        self._max_sat = max_sat
        self._fitted = True

        self.log.info(
            f"Fitted on wsi='{batch.wsi_id}' "
            f"(alpha={self.alpha}, beta={self.beta}, I0={self.I0}, "
            f"pixel_limit={self.pixel_limit}"
        )

    def _macenko_fit_from_batch_xp(self, patches_u8_nhwc) -> Tuple:
        """
        Returns (H(3,2), max_sat(2,1)) on the same xp backend as input.
        """
        patches = _ensure_rgb_uint8_nhwc(patches_u8_nhwc, self._xp)

        flat = patches.reshape(-1, 3)  # (N*H*W, 3)
        OD = _to_OD(flat, self.I0, self._xp)  # (M, 3)

        # Background mask: keep pixels where all OD >= beta
        od_norm = self._xp.linalg.norm(OD, axis=1)
        keep = od_norm >= self.beta
        OD = OD[keep]
        if OD.size == 0:
            raise RuntimeError("Macenko fit: all pixels filtered as background.")

        # Optional subsampling
        if self.pixel_limit is not None and OD.shape[0] > self.pixel_limit:
            idx = self._xp.random.RandomState(seed=0).choice(int(OD.shape[0]), size=self.pixel_limit, replace=False)
            OD = OD[idx]

        # Center & SVD
        ODc = OD - self._xp.mean(OD, axis=0, keepdims=True)
        _, _, vt = self._xp.linalg.svd(ODc, full_matrices=False)
        eigvecs = vt.T[:, :2]  # (3,2)

        # Fix signs for determinism
        sign0 = self._xp.sign(eigvecs[0, 0]) if eigvecs[0, 0] != 0 else 1.0
        sign1 = self._xp.sign(eigvecs[0, 1]) if eigvecs[0, 1] != 0 else 1.0
        eigvecs[:, 0] *= sign0
        eigvecs[:, 1] *= sign1

        # Project & angle extremes
        proj = OD @ eigvecs  # (M,2)
        angles = self._xp.arctan2(proj[:, 1], proj[:, 0])
        min_phi = self._xp.percentile(angles, self.alpha, axis=None)
        max_phi = self._xp.percentile(angles, 100.0 - self.alpha, axis=None)

        v1 = eigvecs @ self._xp.asarray([self._xp.cos(min_phi), self._xp.sin(min_phi)], dtype=self._xp.float32)
        v2 = eigvecs @ self._xp.asarray([self._xp.cos(max_phi), self._xp.sin(max_phi)], dtype=self._xp.float32)
        H_mat = self._xp.stack([v1, v2], axis=1).astype(self._xp.float32)  # (3,2)

        # Stable column order (heuristic)
        if H_mat[0, 0] < H_mat[0, 1]:
            H_mat = H_mat[:, [1, 0]]

        # Normalize columns:
        H_mat = H_mat / self._xp.clip(self._xp.linalg.norm(H_mat, axis=0, keepdims=True), 1e-12, None)

        # Least squares for C on fit pixels to get per-stain 99th percentile
        C, *_ = self._xp.linalg.lstsq(H_mat, OD.T, rcond=None)
        max_sat = self._xp.percentile(C, 99.0, axis=(1,), keepdims=True).astype(self._xp.float32)
        max_sat = self._xp.where(max_sat == 0, self._xp.asarray(1.0, dtype=self._xp.float32), max_sat)

        return H_mat, max_sat


def _ensure_rgb_uint8_nhwc(x, xp):
    if x.ndim != 4 or x.shape[-1] < 3:
        raise ValueError(f"Expected NHWC with >=3 channels, got {x.shape}")
    if x.dtype != xp.uint8:
        x = x.astype(xp.uint8, copy=False)
    return x[..., :3]


def _to_OD(flat_rgb_u8, I0: int, xp):
    """
    Convert flat RGB values to optical density (OD).

    Args:
        flat_rgb_u8: Array of shape (N, 3), dtype uint8 or float-like in [0, 255].
        I0: Reference/white intensity (typically 255).
    """
    rgb = flat_rgb_u8.astype(xp.float32, copy=False)
    rgb = xp.clip(rgb, 1.0, float(I0))
    return -xp.log(rgb / float(I0))


def _from_OD(OD, I0: int, xp):
    """
    Convert optical density (OD) back to RGB uint8.

    Args:
        OD: Array of shape (N, 3), optical density values.
        I0: Reference/white intensity (typically 255).
    """
    white_intensity = float(I0)

    intensity_float = xp.exp(-OD) * white_intensity
    rgb_uint8 = xp.clip(intensity_float, 0.0, 255.0).astype(xp.uint8)
    return rgb_uint8
