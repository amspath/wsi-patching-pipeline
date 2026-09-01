from __future__ import annotations

import numpy as np
import pytest

from wsi_patching.core import PatchExtractor, WSIGrid
from wsi_patching.writers import NumpyStreamWriter


def _stream_patches(slide_path: str, **wsi_grid_kwargs) -> list[np.ndarray]:
    """Run the streaming pipeline and return list of patch batches."""
    p = (
        WSIGrid(slides=[slide_path], use_gpu=False, **wsi_grid_kwargs)
        .then(PatchExtractor(tile_size=64, stride=64, max_batch_size=200))
        .to(NumpyStreamWriter(layout="NHWC"))
    )
    return [imgs for _, imgs, _, _ in p.stream()]


class TestResampleFallbackMode:
    """Integration tests for fallback_mode='resample'."""

    def test_patch_count_reflects_virtual_dims(self, pyramid_slide):
        """resample at ds=4 uses virtual 128×128 dims → 4 patches (2×2 grid of 64×64).

        Contrast with nearest at ds=4, which uses actual level-1 dims (256×256)
        → 16 patches (4×4 grid of 64×64).
        """
        # resample: rf = 4.0 / 2.0 = 2.0; virtual dims = round(256/2)×round(256/2) = 128×128
        resample_batches = _stream_patches(pyramid_slide, resolution=4.0, unit="downsample", fallback_mode="resample")
        resample_total = sum(b.shape[0] for b in resample_batches)

        # nearest: picks level 1 (ds=2.0, closest to 4.0); actual dims 256×256.
        # 2.0 is 50% off the request, so the deviation guard has to be disabled here.
        nearest_batches = _stream_patches(
            pyramid_slide, resolution=4.0, unit="downsample", fallback_mode="nearest", max_relative_deviation=None
        )
        nearest_total = sum(b.shape[0] for b in nearest_batches)

        assert resample_total == 4, f"expected 4 patches (2×2 virtual grid), got {resample_total}"
        assert nearest_total == 16, f"expected 16 patches (4×4 actual grid), got {nearest_total}"

    def test_patch_shape_is_tile_size(self, pyramid_slide):
        """Output patches must be exactly tile_size×tile_size regardless of resample factor."""
        batches = _stream_patches(pyramid_slide, resolution=4.0, unit="downsample", fallback_mode="resample")
        for batch in batches:
            assert batch.shape[1:] == (64, 64, 3), f"unexpected patch shape {batch.shape[1:]}"

    def test_patch_content_faithful_to_slide(self, pyramid_slide):
        """Top-left patch should be dominated by red pixels (level-1 top-left is all red)."""
        batches = _stream_patches(pyramid_slide, resolution=4.0, unit="downsample", fallback_mode="resample")
        all_patches = np.concatenate(batches, axis=0)  # (4, 64, 64, 3)

        # Patches are ordered by (y, x); top-left patch is index 0.
        top_left = all_patches[0]  # (64, 64, 3)
        mean_rgb = top_left.mean(axis=(0, 1))
        # Red channel dominant, close to the source [200, 50, 50]
        assert mean_rgb[0] > 150, f"expected red-dominant top-left patch, got mean RGB {mean_rgb}"
        assert mean_rgb[1] < 100
        assert mean_rgb[2] < 100

    def test_resample_factor_one_when_exact_level_exists(self, pyramid_slide):
        """When requested ds exactly matches a level, resample_factor should be 1.0
        and patch count should match the nearest/ceil result."""
        # ds=2.0 exactly matches level 1 → rf=1.0, virtual dims = actual dims = 256×256
        resample_batches = _stream_patches(pyramid_slide, resolution=2.0, unit="downsample", fallback_mode="resample")
        nearest_batches = _stream_patches(pyramid_slide, resolution=2.0, unit="downsample", fallback_mode="nearest")
        resample_total = sum(b.shape[0] for b in resample_batches)
        nearest_total = sum(b.shape[0] for b in nearest_batches)

        assert resample_total == nearest_total == 16  # 4×4 from 256×256


class TestResampleInterpolation:
    """Check that each interpolation option produces valid, non-trivial patches."""

    @pytest.mark.parametrize("interp", ["nearest", "linear", "cubic", "area", "lanczos"])
    def test_interpolation_mode_produces_valid_patches(self, pyramid_slide, interp):
        batches = _stream_patches(
            pyramid_slide, resolution=4.0, unit="downsample", fallback_mode="resample", resample_interpolation=interp
        )
        total = sum(b.shape[0] for b in batches)
        assert total == 4

        all_patches = np.concatenate(batches, axis=0)
        # Patches must not be all zeros — slide has non-zero colour data
        assert all_patches.max() > 0, f"interpolation={interp!r} produced all-zero patches"
        # All patches must have the correct shape
        assert all_patches.shape == (4, 64, 64, 3)
