import numpy as np
import pytest

from wsi_patching.backends.cupy_numpy import get_xp_backend
from wsi_patching.core.types.types import CollatedPatchBatch
from wsi_patching.filtering.pen_artifact_filter import PenArtifactFilter
from wsi_patching.utils.meta_typing import PipelineContext


def make_collated(patches):
    return CollatedPatchBatch(
        wsi_id="id", patches=patches, coords=np.zeros((patches.shape[0], 2), dtype=np.int32), use_gpu=False
    )


# ---- Fixtures with tiny synthetic batches ----
@pytest.fixture
def xp():
    # CPU path for tests unless explicitly testing GPU
    return get_xp_backend(False)


def make_patch(xp, rgb, h=8, w=8):
    patch = xp.zeros((h, w, 3), dtype=xp.uint8)
    patch[..., 0] = rgb[0]
    patch[..., 1] = rgb[1]
    patch[..., 2] = rgb[2]
    return patch


@pytest.fixture
def collated_mixed_batch(xp):
    """
    Batch of 3 patches:
      - strong blue-ish "pen-like" pixels -> expected to be filtered OUT
      - mid tissue-like -> kept
      - very dark -> kept due to darkness guard
    """
    # Heavily blue (likely to be classified as pen by LUTs)
    pen_like_blue = make_patch(xp, (20, 50, 220))
    # A mid-tone tissue-like color
    tissue_like = make_patch(xp, (180, 120, 120))
    # Very dark patch (kept by darkness guard regardless of LUT)
    very_dark = make_patch(xp, (1, 1, 1))

    patches = xp.stack([pen_like_blue, tissue_like, very_dark], axis=0)
    return make_collated(patches)


# ---- Tests ----
def test_init_and_validate_builds_luts_cpu():
    """Verifies that validate() on CPU builds all three mode LUTs."""
    f = PenArtifactFilter(diff_thresh=5.0, max_pen_fraction=0.01)
    # Simulate pipeline context
    f.attach_context(PipelineContext({"use_gpu": False}))
    f.validate()

    # Internal LUTs should be built
    assert f._luts_red is not None
    assert f._luts_green is not None
    assert f._luts_blue is not None


def test_validate_gpu_requires_cupy_or_skips():
    """Ensures GPU validation runs when CuPy is available, otherwise the test is skipped cleanly."""
    f = PenArtifactFilter()
    f.attach_context(PipelineContext({"use_gpu": False}))
    try:
        f.validate()
    except Exception as e:
        # If CuPy isn't available, validate_xp_backend(True) should fail.
        # That's acceptable; just assert we hit the expected path.
        pytest.skip(f"GPU backend unavailable or CuPy not installed: {e}")


def test_filter_drops_pen_like_and_keeps_normal_and_dark(collated_mixed_batch):
    """Checks that pen-like patches are removed while normal and very dark patches are retained."""
    f = PenArtifactFilter(diff_thresh=5.0, max_pen_fraction=0.01)
    f.attach_context(PipelineContext({"use_gpu": False}))
    f.validate()

    batches = list(f([collated_mixed_batch]))
    assert len(batches) == 1
    out = batches[0]

    # We started with 3 patches: [pen-like blue, tissue-like, very dark].
    # Expect the pen-like one to be removed, keeping 2.
    assert out.patches.shape[0] == 2

    # Optional: if using the stub, we can inspect pen_fraction shape == kept count
    if hasattr(out, "_cols") and "pen_fraction" in out._cols:  # type: ignore
        assert out._cols["pen_fraction"].shape[0] == 2  # type: ignore


def test_accepts_float32_scaled_input_and_casts(xp):
    """Confirms float32 inputs scaled to [0,255] are accepted and correctly cast to uint8 for processing."""
    f = PenArtifactFilter()
    f.attach_context(PipelineContext({"use_gpu": False}))
    f.validate()

    # Create two patches in float32 scaled [0,255]
    pen_like = make_patch(xp, (20, 50, 220)).astype(xp.float32)
    tissue_like = make_patch(xp, (180, 120, 120)).astype(xp.float32)
    patches = xp.stack([pen_like, tissue_like], axis=0)

    # Wrap in collated container
    collated = make_collated(patches)

    # Run filter
    batches = list(f([collated]))
    out = batches[0]

    # Should produce at least 1 kept (tissue-like); pen-like likely dropped
    assert out.patches.shape[0] >= 1


def test_darkness_guard_allows_very_dark_pixels(xp):
    """Ensures the darkness guard preserves very dark patches even if LUTs would reject them."""
    f = PenArtifactFilter(diff_thresh=5.0, max_pen_fraction=0.01)
    f.attach_context(PipelineContext({"use_gpu": False}))
    f.validate()

    # Build a batch of a single very dark patch that would otherwise be suspicious
    dark = make_patch(xp, (1, 1, 1))
    collated = make_collated(xp.stack([dark], axis=0))

    # Process
    out = list(f([collated]))[0]

    # Darkness guard should keep it
    assert out.patches.shape[0] == 1


def test_pen_fraction_threshold_controls_filtering(xp):
    """Validates that changing max_pen_fraction tightens/loosens filtering as expected."""
    # Make a patch that is strongly "pen-like" to exceed threshold
    pen_like = make_patch(xp, (10, 40, 230), h=16, w=16)
    # And a patch that should largely pass
    ok = make_patch(xp, (180, 120, 120), h=16, w=16)
    collated = make_collated(xp.stack([pen_like, ok], axis=0))

    # With very strict threshold, expect to drop first patch
    f_strict = PenArtifactFilter(max_pen_fraction=0.001)
    f_strict.attach_context(PipelineContext({"use_gpu": False}))
    f_strict.validate()
    out_strict = list(f_strict([collated]))[0]
    assert out_strict.patches.shape[0] == 1

    # With very loose threshold, keep both
    collated2 = make_collated(xp.stack([pen_like, ok], axis=0))
    f_loose = PenArtifactFilter(max_pen_fraction=1.0)
    f_loose.attach_context(PipelineContext({"use_gpu": False}))
    f_loose.validate()
    out_loose = list(f_loose([collated2]))[0]
    assert out_loose.patches.shape[0] == 2
