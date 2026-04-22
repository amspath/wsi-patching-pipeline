from typing import Any, Iterable, Tuple

import numpy as np

from wsi_patching.backends.cupy_numpy import get_xp_backend
from wsi_patching.core.pipeline import Stage
from wsi_patching.core.types.types import CollatedPatchBatch


def _rgb_to_lab(rgb_nhwc: np.ndarray, xp, rgb2xyz=None, white=None) -> np.ndarray:
    """
    Convert NHWC RGB (float32, [0,1]) to NHWC Lab.
    """
    # Convert to NCHW for processing
    if rgb_nhwc.ndim == 4:
        N, H, W, C = rgb_nhwc.shape
        rgb_nchw = rgb_nhwc.transpose(0, 3, 1, 2)  # (N,3,H,W)
    else:  # single image (H,W,3)
        rgb_nchw = rgb_nhwc.transpose(2, 0, 1)[None, ...]  # (1,3,H,W)

    # sRGB -> linear RGB
    threshold = 0.04045
    lin_rgb = xp.where(
        rgb_nchw > threshold,
        xp.power(((rgb_nchw + 0.055) / 1.055), 2.4),
        rgb_nchw / 12.92,
    )  # (N,3,H,W)

    # RGB -> XYZ
    if rgb2xyz is None:
        rgb2xyz = xp.array(
            [[0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]], dtype=xp.float32
        )
    # xyz[n,i,h,w] = Σ_j rgb2xyz[i,j] * lin_rgb[n,j,h,w]   ==   M @ rgb per pixel
    xyz = xp.einsum('ij, njhw -> nihw', rgb2xyz, lin_rgb)

    # Normalize for D65 white point
    if white is None:
        white = xp.array([0.95047, 1.0, 1.08883], dtype=xp.float32).reshape(1, 3, 1, 1)
    xyz_norm = xyz / white

    # Non‑linear transform to Lab
    delta = 0.008856
    power = xp.where(
        xyz_norm > delta,
        xp.power(xp.clip(xyz_norm, delta, None), 1/3.0),   # clamp to avoid numerical issues
        (7.787 * xyz_norm) + (4.0 / 29.0),
    )

    L = 116.0 * power[:, 1, :, :] - 16.0
    a = 500.0 * (power[:, 0, :, :] - power[:, 1, :, :])
    b = 200.0 * (power[:, 1, :, :] - power[:, 2, :, :])

    lab_nchw = xp.stack([L, a, b], axis=1)  # (N,3,H,W)
    lab_nhwc = lab_nchw.transpose(0, 2, 3, 1)  # (N,H,W,3)
    if rgb_nhwc.ndim == 3:
        lab_nhwc = lab_nhwc[0]
    return lab_nhwc


def _lab_to_rgb(lab_nhwc: np.ndarray, xp, clip: bool = True, white=None) -> np.ndarray:
    """
    Convert NHWC Lab to NHWC RGB (float32, [0,1]).
    """
    if lab_nhwc.ndim == 4:
        # N, H, W, C = lab_nhwc.shape
        lab_nchw = lab_nhwc.transpose(0, 3, 1, 2)  # (N,3,H,W)
    else:
        lab_nchw = lab_nhwc.transpose(2, 0, 1)[None, ...]
        # N = 1

    L = lab_nchw[:, 0, :, :]
    a = lab_nchw[:, 1, :, :]
    b = lab_nchw[:, 2, :, :]

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    fz = xp.clip(fz, 0.0, None)

    delta = 0.2068966
    fx_ = xp.where(fx > delta, xp.power(fx, 3.0), (fx - 4.0 / 29.0) / 7.787)
    fy_ = xp.where(fy > delta, xp.power(fy, 3.0), (fy - 4.0 / 29.0) / 7.787)
    fz_ = xp.where(fz > delta, xp.power(fz, 3.0), (fz - 4.0 / 29.0) / 7.787)

    xyz = xp.stack([fx_, fy_, fz_], axis=1)  # (N,3,H,W)

    # Denormalise for D65 white point
    if white is None:
        white = xp.array([0.95047, 1.0, 1.08883], dtype=xp.float32).reshape(1, 3, 1, 1)
    xyz_im = xyz * white

    x = xyz_im[:, 0, :, :]
    y = xyz_im[:, 1, :, :]
    z = xyz_im[:, 2, :, :]

    # Direct linear formulas
    r = 3.2404813432005266 * x + -1.5371515162713185 * y + -0.4985363261688878 * z
    g = -0.9692549499965682 * x + 1.8759900014898907 * y + 0.0415559265582928 * z
    b_lin = 0.0556466391351772 * x + -0.2040413383665112 * y + 1.0573110696453443 * z
    lin_rgb = xp.stack([r, g, b_lin], axis=1)  # (N,3,H,W)

    # Linear RGB -> sRGB
    thresh = 0.0031308
    srgb_nchw = xp.where(
        lin_rgb > thresh,
        1.055 * xp.power(xp.clip(lin_rgb, thresh, None), 1/2.4) - 0.055,
        12.92 * lin_rgb,
    )
    if clip:
        srgb_nchw = xp.clip(srgb_nchw, 0.0, 1.0)

    srgb_nhwc = srgb_nchw.transpose(0, 2, 3, 1)
    if lab_nhwc.ndim == 3:
        srgb_nhwc = srgb_nhwc[0]
    return srgb_nhwc


class ReinhardNormalizer(Stage):
    def __init__(
        self,
        lab_reference_mean: Tuple[float, float, float] = (68.94, 29.76, -18.97),
        lab_reference_std: Tuple[float, float, float] = (11.52, 13.42, 8.59),
        apply_modified_reinhard: bool = False,
    ) -> None:
        self.lab_reference_mean = lab_reference_mean
        self.lab_reference_std = lab_reference_std
        self.apply_modified_reinhard = apply_modified_reinhard

        self._xp: Any = None
        self._ref_mean: Any = None
        self._ref_std: Any = None
        self._rgb2xyz: Any = None
        self._white: Any = None

        self.log.info(f"ReinhardNormalizer initialized (modified={apply_modified_reinhard})")

    def validate(self) -> None:
        self.ctx.require_key("use_gpu")

    def __call__(self, it: Iterable[CollatedPatchBatch]) -> Iterable[CollatedPatchBatch]:
        for batch in it:
            if self._xp is None:
                self._xp = get_xp_backend(self.ctx["use_gpu"])
                self._ref_mean = self._xp.asarray(self.lab_reference_mean, dtype=self._xp.float32)
                self._ref_std = self._xp.asarray(self.lab_reference_std, dtype=self._xp.float32)
                self._rgb2xyz = self._xp.array(
                    [[0.412453, 0.357580, 0.180423],
                     [0.212671, 0.715160, 0.072169],
                     [0.019334, 0.119193, 0.950227]], dtype=self._xp.float32
                )
                self._white = self._xp.array(
                    [0.95047, 1.0, 1.08883], dtype=self._xp.float32
                ).reshape(1, 3, 1, 1)

            patches = batch.patches  # NHWC uint8
            if patches.ndim != 4 or patches.shape[-1] != 3:
                raise ValueError(f"Expected NHWC with 3 channels, got {patches.shape}")
            if patches.dtype != self._xp.uint8:
                patches = patches.astype(self._xp.uint8, copy=False)

            # Convert to float [0,1] and keep NHWC
            rgb_float = patches.astype(self._xp.float32) / 255.0  # (N, H, W, 3)

            # Convert to Lab (NHWC)
            lab = _rgb_to_lab(rgb_float, self._xp, rgb2xyz=self._rgb2xyz, white=self._white)  # (N, H, W, 3)

            # Compute per‑patch mean and std over spatial axes (H,W)
            lab_flat = lab.reshape(len(lab), -1, 3)
            mean_lab = lab_flat.mean(axis=1)   # (N,3)
            std_lab = lab_flat.std(axis=1, ddof=1)  # (N,3)

            if not self.apply_modified_reinhard:
                # Standard Reinhard
                lab_centered = lab - mean_lab[:, None, None, :]
                lab_scaled = lab_centered / (std_lab[:, None, None, :] + 1e-8)
                lab_transformed = (
                    lab_scaled * self._ref_std[None, None, None, :]
                    + self._ref_mean[None, None, None, :]
                )
            else:
                # Modified Reinhard (uses ddof=1 std as well)
                q = (self._ref_std[0] - std_lab[:, 0]) / (self._ref_std[0] + 1e-8)
                q_mul = self._xp.where(q > 0, 1.0 + q, 1.05)  # (N,)

                L_new = (
                    (lab[..., 0] - mean_lab[:, 0][:, None, None]) * q_mul[:, None, None]
                    + mean_lab[:, 0][:, None, None]
                )
                a_new = lab[..., 1].copy()
                b_new = lab[..., 2].copy()

                mask = q <= 0
                if self._xp.any(mask):
                    a_new[mask] = (
                        lab[mask, ..., 1] - mean_lab[mask, 1][:, None, None]
                    ) + self._ref_mean[1]
                    b_new[mask] = (
                        lab[mask, ..., 2] - mean_lab[mask, 2][:, None, None]
                    ) + self._ref_mean[2]

                lab_transformed = self._xp.stack([L_new, a_new, b_new], axis=-1)

            # Convert back to RGB (NHWC)
            rgb_norm_float = _lab_to_rgb(lab_transformed, self._xp, clip=True, white=self._white)

            # Back to uint8
            rgb_norm_uint8 = self._xp.round(rgb_norm_float * 255.0).astype(self._xp.uint8)
            batch.patches = rgb_norm_uint8

            self.log.debug(f"Normalized batch for wsi='{batch.wsi_id}' size={len(batch)}")
            yield batch