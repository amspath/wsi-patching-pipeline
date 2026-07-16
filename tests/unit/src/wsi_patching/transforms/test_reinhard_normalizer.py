import numpy as np
import pytest

from wsi_patching.core.types.types import CollatedPatchBatch
from wsi_patching.transforms.reinhard_normalizer import ReinhardNormalizer, _lab_to_rgb, _rgb_to_lab
from wsi_patching.utils.meta_typing import PipelineContext


def fake_get_xp_backend(use_gpu):
    import numpy as xp

    return xp


def _make_batch(patches: np.ndarray) -> CollatedPatchBatch:
    return CollatedPatchBatch(
        patches=patches,
        wsi_id="test",
        coords=np.zeros((len(patches), 2), dtype=np.int64),
        use_gpu=False,
        wsi_dims=(1024, 1024),
    )


def _run_normalizer(norm: ReinhardNormalizer, batch: CollatedPatchBatch) -> CollatedPatchBatch:
    norm.attach_context(PipelineContext({"use_gpu": False}))
    (out,) = list(norm([batch]))
    return out


# ---- Tests ----


def test_validate_requires_use_gpu_key():
    norm = ReinhardNormalizer()
    norm.attach_context(PipelineContext({}))
    with pytest.raises(KeyError):
        norm.validate()

    norm.attach_context(PipelineContext({"use_gpu": False}))
    norm.validate()  # should not raise


def test_output_shape_and_dtype(monkeypatch):
    monkeypatch.setattr("wsi_patching.transforms.reinhard_normalizer.get_xp_backend", fake_get_xp_backend)

    patches = np.random.randint(0, 256, (4, 32, 32, 3), dtype=np.uint8)
    batch = _make_batch(patches)
    out = _run_normalizer(ReinhardNormalizer(), batch)

    assert out.patches.shape == patches.shape
    assert out.patches.dtype == np.uint8


def test_rgb_lab_roundtrip():
    rng = np.random.default_rng(0)
    rgb = rng.random((3, 16, 16, 3)).astype(np.float32)
    recovered = _lab_to_rgb(_rgb_to_lab(rgb, np), np, clip=False)
    np.testing.assert_allclose(recovered, rgb, atol=1e-4)


def test_standard_reinhard_shifts_stats_toward_reference(monkeypatch):
    monkeypatch.setattr("wsi_patching.transforms.reinhard_normalizer.get_xp_backend", fake_get_xp_backend)

    # Solid-color patch: std=0, so output Lab values collapse to ref_mean
    ref_mean = (50.0, 10.0, -10.0)
    ref_std = (8.0, 6.0, 5.0)
    patches = np.full((2, 32, 32, 3), fill_value=200, dtype=np.uint8)
    batch = _make_batch(patches)

    norm = ReinhardNormalizer(lab_reference_mean=ref_mean, lab_reference_std=ref_std)
    out = _run_normalizer(norm, batch)

    # Convert output back to Lab to check mean is close to ref_mean
    rgb_float = out.patches.astype(np.float32) / 255.0
    lab_out = _rgb_to_lab(rgb_float, np)
    lab_mean = lab_out.reshape(len(lab_out), -1, 3).mean(axis=1)  # (N, 3)

    expected = np.tile(np.array(ref_mean), (len(patches), 1))
    np.testing.assert_allclose(lab_mean, expected, atol=1.5)


def test_modified_reinhard_smoke(monkeypatch):
    monkeypatch.setattr("wsi_patching.transforms.reinhard_normalizer.get_xp_backend", fake_get_xp_backend)

    patches = np.random.randint(0, 256, (4, 32, 32, 3), dtype=np.uint8)
    batch = _make_batch(patches)
    out = _run_normalizer(ReinhardNormalizer(apply_modified_reinhard=True), batch)

    assert out.patches.shape == patches.shape
    assert out.patches.dtype == np.uint8
    assert int(out.patches.min()) >= 0
    assert int(out.patches.max()) <= 255
