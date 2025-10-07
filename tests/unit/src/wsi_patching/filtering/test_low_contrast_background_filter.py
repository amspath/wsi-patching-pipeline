import numpy as np
import pytest

from wsi_patching.core.types.types import CollatedPatchBatch
from wsi_patching.filtering.low_contrast_background_filter import LowContrastBackgroundFilter
from wsi_patching.utils.meta_typing import PipelineContext


# -----------------------
# Fixtures for monkeypatching xp+backends
# -----------------------
class _XPRecorder:
    """
    A tiny NumPy-like 'xp' namespace that also records dtypes passed to asarray.
    Provides just what the filter needs: float32/float64, asarray, ones, full, array_equal.
    """

    def __init__(self):
        self.float32 = np.float32
        self.float64 = np.float64
        self._asarray_dtypes = []

    def asarray(self, x, dtype=None):
        self._asarray_dtypes.append(dtype)
        return np.asarray(x, dtype=dtype)

    def ones(self, *args, **kwargs):
        return np.ones(*args, **kwargs)

    def full(self, *args, **kwargs):
        return np.full(*args, **kwargs)

    def array_equal(self, a, b):
        return np.array_equal(a, b)


@pytest.fixture
def np_xp():
    return _XPRecorder()


@pytest.fixture
def patch_backends(monkeypatch, np_xp):
    """Monkeypatch backend helpers so we can control behavior and keep everything on NumPy."""
    # ensure_array_matches_use_gpu: just return input unchanged
    import wsi_patching.filtering.low_contrast_background_filter as mod

    monkeypatch.setattr(mod, "ensure_array_matches_use_gpu", lambda arr, use_gpu: arr, raising=True)
    # ensure_numpy: identity for tests
    monkeypatch.setattr(mod, "ensure_numpy", lambda x: np.asarray(x), raising=True)
    # get_xp_backend: return our numpy-like namespace
    monkeypatch.setattr(mod, "get_xp_backend", lambda use_gpu: np_xp, raising=True)


# -----------------------
# Parameter validation tests
# -----------------------
@pytest.mark.parametrize("val", [-0.1, 1.1])
def test_init_rejects_bad_range_threshold(val):
    with pytest.raises(ValueError, match="range_threshold must be in \\[0, 1\\]"):
        LowContrastBackgroundFilter(range_threshold=val)


@pytest.mark.parametrize("prec", ["float16", "fp32", "double"])
def test_init_rejects_bad_precision(prec):
    with pytest.raises(ValueError, match="float_precision must be 'float32' or 'float64'"):
        LowContrastBackgroundFilter(float_precision=prec)


# -----------------------
# validate() should require ctx['use_gpu']
# -----------------------
def test_validate_requires_use_gpu_key():
    f = LowContrastBackgroundFilter()
    f.attach_context(PipelineContext({}))
    with pytest.raises(KeyError):
        f.validate()
    # now with key should pass
    f.attach_context(PipelineContext({"use_gpu": False}))
    f.validate()


# -----------------------
# Core path: stats + filtering + logging
# -----------------------
def test_stats_and_filtering(monkeypatch, patch_backends, caplog):
    """
    Build controlled grayscale so we know exact min/max/range and which items survive.
    """
    # Create a batch of three 2x2 RGB patches (values don't matter; we'll stub grayscale)
    B, H, W, C = 3, 2, 2, 3
    patches = np.zeros((B, H, W, C), dtype=np.uint8)

    # Grayscale we want the stage to see:
    # item0: range = 0.3  -> keep (>= 0.2)
    # item1: range = 0.05 -> drop
    # item2: range = 1.0  -> keep
    gray = np.stack(
        [
            np.array([[0.1, 0.4], [0.2, 0.3]], dtype=np.float32),  # min=0.1, max=0.4, range=0.3
            np.array([[0.50, 0.55], [0.52, 0.53]], dtype=np.float32),  # range=0.05
            np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),  # range=1.0
        ],
        axis=0,
    )

    def stub_rgb_to_gray(self, imgs, xp):
        # Confirm patches came through unchanged, then return our gray
        assert xp.array_equal(imgs, patches)
        return gray

    monkeypatch.setattr(LowContrastBackgroundFilter, "_rgb_to_gray_xp", stub_rgb_to_gray)

    filt = LowContrastBackgroundFilter(range_threshold=0.2, float_precision="float32")
    filt.attach_context(PipelineContext({"use_gpu": False}))

    batch = CollatedPatchBatch(patches=patches.copy(), wsi_id="slideA", coords=np.zeros((B, 2)), use_gpu=False)

    with caplog.at_level("INFO"):
        out_batches = list(filt([batch]))

    # One yielded batch
    assert len(out_batches) == 1
    out = out_batches[0]

    # Check columns exist
    for k in ("gray_min", "gray_max", "gray_range", "range_threshold"):
        assert k in out.metadata.columns()

    # Expected stats BEFORE filtering would be:
    g_min = np.array([0.1, 0.50, 0.0], dtype=np.float32)
    g_max = np.array([0.4, 0.55, 1.0], dtype=np.float32)
    g_rng = g_max - g_min  # [0.3, 0.05, 1.0]
    keep = g_rng >= 0.2  # [T, F, T]

    # After filtering, only items 0 and 2 remain
    np.testing.assert_allclose(out.metadata.get("gray_min"), g_min[keep])
    np.testing.assert_allclose(out.metadata.get("gray_max"), g_max[keep])
    np.testing.assert_allclose(out.metadata.get("gray_range"), g_rng[keep])
    # range_threshold meta_col should be per-item, matching gray dtype
    np.testing.assert_allclose(out.metadata.get("range_threshold"), np.array([0.2, 0.2], dtype=np.float32))

    # Patches count reflects filtering
    assert out.patches.shape[0] == 2

    # Log captured and contains wsi + batch_out
    assert any("wsi=slideA" in rec.message and "batch_out=2" in rec.message for rec in caplog.records)


def test_all_filtered_yields_nothing(monkeypatch, patch_backends):
    """
    If every item is filtered out, the stage should not yield the batch at all.
    """
    B, H, W, C = 2, 2, 2, 3
    patches = np.zeros((B, H, W, C), dtype=np.uint8)

    # Both items have tiny range 0.01 -> both dropped if threshold=0.2
    gray = np.stack(
        [
            np.array([[0.50, 0.51], [0.50, 0.51]], dtype=np.float32),
            np.array([[0.60, 0.61], [0.60, 0.61]], dtype=np.float32),
        ],
        axis=0,
    )

    monkeypatch.setattr(LowContrastBackgroundFilter, "_rgb_to_gray_xp", lambda self, imgs, xp: gray)

    filt = LowContrastBackgroundFilter(range_threshold=0.2, float_precision="float32")
    filt.attach_context(PipelineContext({"use_gpu": False}))

    batch = CollatedPatchBatch(patches=patches.copy(), wsi_id="slideB", coords=np.zeros((B, 2)), use_gpu=False)

    out_batches = list(filt([batch]))
    assert out_batches == []  # nothing yielded


def test_float_precision_controls_threshold_dtype(monkeypatch, np_xp, patch_backends):
    """
    Verifies the dtype passed to xp.asarray(...) when building `thr`.
    """
    import wsi_patching.filtering.low_contrast_background_filter as mod

    B, H, W, C = 1, 2, 2, 3
    patches = np.zeros((B, H, W, C), dtype=np.uint8)
    gray = np.array([[[0.1, 0.4], [0.2, 0.3]]], dtype=np.float32)  # range 0.3

    monkeypatch.setattr(mod.LowContrastBackgroundFilter, "_rgb_to_gray_xp", lambda self, imgs, xp: gray, raising=True)

    # float64 path
    before = len(np_xp._asarray_dtypes)
    f64 = mod.LowContrastBackgroundFilter(range_threshold=0.2, float_precision="float64")
    f64.attach_context(PipelineContext({"use_gpu": False}))
    _ = list(f64([CollatedPatchBatch(patches=patches.copy(), wsi_id="id", coords=np.zeros((B, 2)), use_gpu=False)]))
    assert np_xp._asarray_dtypes[before] == np.float64

    # float32 path
    before = len(np_xp._asarray_dtypes)
    f32 = mod.LowContrastBackgroundFilter(range_threshold=0.2, float_precision="float32")
    f32.attach_context(PipelineContext({"use_gpu": False}))
    _ = list(f32([CollatedPatchBatch(patches=patches.copy(), wsi_id="id", coords=np.zeros((B, 2)), use_gpu=False)]))
    assert np_xp._asarray_dtypes[before] == np.float32
