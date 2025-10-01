# tests/test_otsu_filter.py
import types

import numpy as np
import pytest

from wsi_patching.backends.cupy_numpy import ensure_cupy
from wsi_patching.filtering.otsu_filter import OtsuFilter
from wsi_patching.utils.meta_typing import PipelineContext
from wsi_patching.utils.types import CollatedPatchBatch


# -----------------------
# Fixtures for monkeypatching xp+backends
# -----------------------
@pytest.fixture
def np_xp():
    """Return a namespace that looks like numpy for xp backends (no CuPy required)."""
    ns = types.SimpleNamespace(float32=np.float32, ones=np.ones)
    return ns


@pytest.fixture
def patch_backends(monkeypatch, np_xp):
    """Monkeypatch backend helpers so we can control behavior and keep everything on NumPy."""
    # ensure_array_matches_use_gpu: just return input unchanged
    monkeypatch.setitem(globals(), "ensure_array_matches_use_gpu", lambda arr, use_gpu: arr)
    # ensure_numpy: identity for tests
    monkeypatch.setitem(globals(), "ensure_numpy", lambda x: np.asarray(x))
    # get_xp_backend: return our numpy-like namespace
    monkeypatch.setitem(globals(), "get_xp_backend", lambda use_gpu: np_xp)


# -----------------------
# Parameter validation tests
# -----------------------


@pytest.mark.parametrize("num_bins", [0, 1])
def test_init_rejects_small_num_bins(num_bins):
    with pytest.raises(ValueError, match="num_bins must be >= 2"):
        OtsuFilter(num_bins=num_bins)


@pytest.mark.parametrize("prec", ["float16", "fp32", "double"])
def test_init_rejects_bad_precision(prec):
    with pytest.raises(ValueError, match="float_precision must be 'float32' or 'float64'"):
        OtsuFilter(float_precision=prec)


@pytest.mark.parametrize("val", [-0.1, 1.1])
def test_init_rejects_bad_min_fraction(val):
    with pytest.raises(ValueError, match="min_tissue_fraction must be in \\[0, 1\\]"):
        OtsuFilter(min_tissue_fraction=val)


# -----------------------
# validate() should require ctx['use_gpu']
# -----------------------


def test_validate_requires_use_gpu_key():
    f = OtsuFilter()
    f.attach_context(PipelineContext({}))
    with pytest.raises(KeyError):
        f.validate()
    # now with key should pass and record the requirement
    f.attach_context(PipelineContext({"use_gpu": False}))
    f.validate()


# -----------------------
# Main call path: polarity + filtering + stats
# -----------------------


@pytest.mark.parametrize("tissue_is_darker", [True, False])
def test_polarity_and_stats(monkeypatch, patch_backends, tissue_is_darker, caplog):
    """
    Build a controlled gray image and thresholds so we know exactly which pixels are 'tissue'.
    We stub the private methods to avoid relying on real image ops.
    """
    # Create a batch of two small 2x2 RGB patches
    B, H, W, C = 2, 2, 2, 3
    patches = np.zeros((B, H, W, C), dtype=np.uint8)

    # Stub _rgb_to_gray_xp to return a predictable gray array in [0,1]
    # gray[0] = [[0.2, 0.8],[0.2,0.8]] ; gray[1] = [[0.6,0.6],[0.6,0.6]]
    gray = np.stack(
        [np.array([[0.2, 0.8], [0.2, 0.8]], dtype=np.float32), np.full((H, W), 0.6, dtype=np.float32)], axis=0
    )

    # thresholds per batch item
    thresholds = np.array([0.5, 0.6], dtype=np.float32)

    def stub_rgb_to_gray(self, imgs, xp):
        assert xp.array_equal(imgs, patches)  # got the same array object
        return gray

    def stub_otsu(self, gray_in, num_bins, xp):
        assert num_bins == 256
        assert xp.array_equal(gray_in, gray)
        return thresholds

    monkeypatch.setattr(OtsuFilter, "_rgb_to_gray_xp", stub_rgb_to_gray)
    monkeypatch.setattr(OtsuFilter, "_batched_otsu_thresholds_xp", stub_otsu)

    # Build filter
    filt = OtsuFilter(tissue_is_darker=tissue_is_darker, num_bins=256, min_tissue_fraction=0.0)
    filt.attach_context(PipelineContext({"use_gpu": False}))

    batch = CollatedPatchBatch(patches=patches.copy(), wsi_id="id", coords=np.zeros((B, 2)), meta_cols={})

    with caplog.at_level("INFO"):
        out_batches = list(filt([batch]))

    assert len(out_batches) == 1
    out = out_batches[0]

    # Check columns exist and have correct shapes
    assert "otsu_threshold" in out.meta_cols
    assert "tissue_fraction" in out.meta_cols
    assert "tissue_pixel_count" in out.meta_cols

    np.testing.assert_allclose(out.meta_cols["otsu_threshold"], thresholds)

    # Compute expected masks for each polarity
    if tissue_is_darker:
        # item0: gray < 0.5 -> pixels [0.2] true, [0.8] false -> 2/4 tissue
        # item1: gray < 0.6 -> all False (since equal is False) -> 0/4 tissue
        expected_counts = np.array([2, 0], dtype=np.int64)
    else:
        # item0: gray >= 0.5 -> pixels [0.8] true -> 2/4 tissue
        # item1: gray >= 0.6 -> all True (equal counts) -> 4/4 tissue
        expected_counts = np.array([2, 4], dtype=np.int64)

    np.testing.assert_array_equal(out.meta_cols["tissue_pixel_count"], expected_counts)
    expected_fracs = expected_counts / 4.0
    np.testing.assert_allclose(out.meta_cols["tissue_fraction"], expected_fracs.astype(np.float32))

    # No filtering when min_tissue_fraction == 0
    assert out.patches.shape[0] == 2

    # Log line captured
    assert any("batch_in=2 -> batch_out=2" in rec.message for rec in caplog.records)


def test_min_tissue_fraction_filters(monkeypatch, patch_backends):
    """
    Same controlled gray/thresholds as above, but set a min_tissue_fraction to filter out items.
    """
    B, H, W, C = 2, 2, 2, 3
    patches = np.zeros((B, H, W, C), dtype=np.uint8)

    gray = np.stack(
        [
            np.array([[0.2, 0.8], [0.2, 0.8]], dtype=np.float32),  # 2/4 tissue in both polarities (see below)
            np.full((H, W), 0.6, dtype=np.float32),  # polarity-dependent
        ],
        axis=0,
    )
    thresholds = np.array([0.5, 0.6], dtype=np.float32)

    monkeypatch.setattr(OtsuFilter, "_rgb_to_gray_xp", lambda self, imgs, xp=None: gray)
    monkeypatch.setattr(OtsuFilter, "_batched_otsu_thresholds_xp", lambda self, g, n, xp=None: thresholds)

    # We'll use tissue_is_darker=False so that item1 has 4/4 tissue; item0 has 2/4.
    filt = OtsuFilter(tissue_is_darker=False, min_tissue_fraction=0.75)
    filt.attach_context(PipelineContext({"use_gpu": False}))

    batch = CollatedPatchBatch(patches=patches.copy(), wsi_id="id", coords=np.zeros((B, 2)), meta_cols={})
    out = next(iter(filt([batch])))

    # Only the second item (4/4 tissue = 1.0) survives; first has 0.5 < 0.75
    assert out.patches.shape[0] == 1
    # The columns should be filtered as well
    np.testing.assert_array_equal(out.meta_cols["tissue_pixel_count"], np.array([4]))
    np.testing.assert_allclose(out.meta_cols["tissue_fraction"], np.array([1.0], dtype=np.float32))
    np.testing.assert_allclose(out.meta_cols["otsu_threshold"], np.array([0.6], dtype=np.float32))
