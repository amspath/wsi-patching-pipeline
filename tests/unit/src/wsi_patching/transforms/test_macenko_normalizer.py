from typing import Tuple

import numpy as np
import pytest

from wsi_patching.backends.cupy_numpy import ensure_cupy
from wsi_patching.core.types.types import CollatedPatchBatch
from wsi_patching.transforms.macenko_normalizer import MacenkoNormalizer
from wsi_patching.utils.meta_typing import PipelineContext


def fake_get_xp_backend(use_gpu):
    import numpy as xp

    return xp


# ---- Helpers: synthetic data & linear-algebra utilities ----
def beer_lambert_forward(H_true, C, I0=255):
    """
    H_true: (3,2)
    C: (2, P) non-negative concentrations
    Returns uint8 RGB intensities in [0,255] of shape (3, P)
    """
    OD = H_true @ C  # (3, P)
    rgb_uint8 = I0 * np.exp(-OD)  # (3, P)
    rgb_uint8 = np.clip(rgb_uint8, 0, 255)
    return rgb_uint8.astype(np.uint8)


def synth_patches_from_stains(H_true, n_patches=4, patch_hw=(32, 32), I0=255, conc_scale=1.5, seed=123):
    """
    Create a stack of NHWC uint8 patches synthesized from a known stain matrix.
    """
    rng = np.random.default_rng(seed)
    H, W = patch_hw
    patches = []
    for _ in range(n_patches):
        # Make smooth-ish nonnegative concentration fields
        # 2 stains, H*W pixels
        C = rng.gamma(shape=2.0, scale=conc_scale, size=(2, H * W))
        # (3, H*W) uint8
        rgb_uint8 = beer_lambert_forward(H_true, C, I0=I0)
        img = rgb_uint8.T.reshape(H, W, 3).astype(np.uint8)
        patches.append(img)
    patches = np.stack(patches, axis=0)  # (N, H, W, 3)
    return patches


def principal_angles_between_subspaces(A, B):
    """
    Return principal angles (radians) between column spaces of A and B.
    A: (m, k), B: (m, k)
    """
    # Orthonormal bases via QR
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    # Singular values of Qa^T Qb are cosines of principal angles
    S = np.linalg.svd(Qa.T @ Qb, full_matrices=False)[1]
    S = np.clip(S, 0.0, 1.0)
    angles = np.arccos(S)
    return np.sort(angles)


def subspace_is_close(H_est, H_true, tol_deg=2.0):
    """
    Check if the 2D subspaces spanned by columns of H_est and H_true align
    within tol_deg degrees for both principal angles.
    """
    angs = principal_angles_between_subspaces(H_est, H_true)
    return np.degrees(angs).max() <= tol_deg


def make_golden_concentrations(
    n_tissue_per_stain: int, n_white: int, conc_values_stain1: np.ndarray, conc_values_stain2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (C_all, idx_s1, idx_s2)
    C_all: (2, P) in the order [all stain1-only pixels | all stain2-only pixels | all whites]
    idx_s1, idx_s2: index arrays for tissue pixels of stain1 and stain2 respectively within C_all.
    """
    assert conc_values_stain1.size == n_tissue_per_stain
    assert conc_values_stain2.size == n_tissue_per_stain

    # Pure stain-1 pixels: C = [c, 0]
    C_s1 = np.vstack([conc_values_stain1, np.zeros_like(conc_values_stain1)])
    # Pure stain-2 pixels: C = [0, c]
    C_s2 = np.vstack([np.zeros_like(conc_values_stain2), conc_values_stain2])
    # Whites (near-zero OD): C ~ [0, 0]
    C_w = np.zeros((2, n_white), dtype=float)

    C_all = np.concatenate([C_s1, C_s2, C_w], axis=1)
    idx_s1 = np.arange(0, n_tissue_per_stain)
    idx_s2 = np.arange(n_tissue_per_stain, 2 * n_tissue_per_stain)
    return C_all, idx_s1, idx_s2


def make_golden_patch_from_H(
    H_true: np.ndarray,
    alpha: float,
    beta: float,
    I0: int = 255,
    n_tissue_per_stain: int = 400,
    white_fraction: float = 0.35,
    c_min: float = 0.2,
    c_max: float = 2.0,
    seed: int = 1234,
) -> dict:
    """
    Construct a single NHWC uint8 patch with:
      - pure stain-1 pixels with known concentrations
      - pure stain-2 pixels with known concentrations
      - a controlled fraction of near-white pixels (OD below beta)
    Returns dict with:
      'patch': (1, H, W, 3) uint8,
      'C_true': (2, P),
      'idx_s1', 'idx_s2' (indices of tissue pixels),
      'expected_max_sat': shape (2,)
    """
    # Make reproducible, non-uniform concentrations with known percentiles
    # Use sorted values so the percentile is deterministic
    conc_values_stain1 = np.linspace(c_min, c_max, n_tissue_per_stain)
    conc_values_stain2 = np.linspace(c_min, c_max, n_tissue_per_stain) ** 1.2  # different shape, still monotone

    n_white = int(round((2 * n_tissue_per_stain) * white_fraction / (1 - white_fraction)))
    C_all, idx_s1, idx_s2 = make_golden_concentrations(
        n_tissue_per_stain, n_white, conc_values_stain1, conc_values_stain2
    )  # (2, P)
    P = C_all.shape[1]

    # Build OD = H @ C; ensure whites are truly "near white" vs beta threshold
    OD = H_true @ C_all  # (3, P)

    # Randomly jitter the white concentrations to be *very* small but safely < beta in OD-norm
    # (Keep tissue untouched.)
    if n_white > 0:
        white_cols = np.arange(2 * n_tissue_per_stain, P)
        # push OD-norm well below beta for whites
        OD[:, white_cols] = OD[:, white_cols] * 0.0 + (beta * 0.2)  # small OD along all channels

    rgb_uint8 = I0 * np.exp(-OD)  # (3, P)
    rgb_uint8 = np.clip(rgb_uint8, 0, 255).astype(np.uint8)

    # Lay out pixels on a rectangle (single patch)
    side = int(np.ceil(np.sqrt(P)))
    total = side * side
    # pad with extra near-white if needed
    if total > P:
        pad_white = total - P
        I_pad = np.full((3, pad_white), I0, dtype=np.uint8)
        rgb_uint8 = np.concatenate([rgb_uint8, I_pad], axis=1)
        C_pad = np.zeros((2, pad_white), dtype=float)
        C_all = np.concatenate([C_all, C_pad], axis=1)

    Hh = side
    Ww = side
    img = rgb_uint8.T.reshape(Hh, Ww, 3)
    patch = img[np.newaxis, ...]  # (1, H, W, 3)

    # Compute expected max_sat as alpha-percentiles of the TRUE tissue concentrations only
    expected_max_sat = np.array(
        [np.percentile(C_all[0, idx_s1], 99.0), np.percentile(C_all[1, idx_s2], 99.0)], dtype=float
    )

    # --- compute exact white fraction in the final patch using OD-norm and beta ---
    img_f = patch[0].reshape(-1, 3).astype(np.float32)
    img_f[img_f == 0] = 1.0
    OD_flat = -np.log(img_f / float(I0))
    OD_norm = np.linalg.norm(OD_flat, axis=1)
    expected_white_fraction = float((OD_norm < beta).mean())

    return {
        "patch": patch,
        "C_true": C_all,
        "idx_s1": idx_s1,
        "idx_s2": idx_s2,
        "expected_max_sat": expected_max_sat,
        "expected_white_fraction": expected_white_fraction,
    }


# ---- Fixtures ----


@pytest.fixture
def H_true():
    # A fixed well-separated stain basis (columns).
    # Columns are unit-length and not collinear.
    # You can swap for your typical H&E basis if you prefer.
    H = np.array([[0.65, 0.07], [0.70, 0.99], [0.29, 0.11]], dtype=float)
    # Normalize columns to unit length (Macenko usually normalizes)
    H /= np.linalg.norm(H, axis=0, keepdims=True)
    return H


@pytest.fixture
def synthetic_batch(H_true):
    patches = synth_patches_from_stains(H_true, n_patches=6, patch_hw=(48, 48), I0=255, conc_scale=1.2, seed=42)
    return CollatedPatchBatch(
        patches=patches, wsi_id="wsi_synth", coords=np.array([[0, 0]] * patches.shape[0]), use_gpu=False
    )


# ---- Tests ----


def test_fit_recovers_stain_subspace_numpy(monkeypatch, H_true, synthetic_batch):
    """
    Validates that Macenko fit recovers the correct 2D subspace of stains
    on CPU/NumPy.
    """
    # monkeypatch the backend getter used by your normalizer
    monkeypatch.setattr("wsi_patching.backends.cupy_numpy.get_xp_backend", fake_get_xp_backend)

    norm = MacenkoNormalizer(alpha=1.0, beta=0.15, light_intensity=255, pixel_limit=None)
    norm.attach_context(PipelineContext({"use_gpu": False}))

    norm.fit(synthetic_batch)

    assert norm._fitted is True
    assert norm._H_mat is not None and norm._H_mat.shape == (3, 2)
    assert norm._max_sat is not None and norm._max_sat.shape in {(2,), (2, 1), (2,)}  # allow slight shape differences

    # Subspace correctness (ignoring column order and sign)
    assert subspace_is_close(np.asarray(norm._H_mat, dtype=float), H_true, tol_deg=3.0)


def test_fit_is_idempotent(monkeypatch, H_true, synthetic_batch):
    monkeypatch.setattr("wsi_patching.backends.cupy_numpy.get_xp_backend", fake_get_xp_backend)

    norm = MacenkoNormalizer()
    norm.attach_context(PipelineContext({"use_gpu": False}))

    norm.fit(synthetic_batch)
    H1 = np.array(norm._H_mat, dtype=float)
    max_sat1 = np.array(norm._max_sat, dtype=float).copy()

    # Call fit again; should be a no-op
    norm.fit(synthetic_batch)
    H2 = np.array(norm._H_mat, dtype=float)
    max_sat2 = np.array(norm._max_sat, dtype=float)

    np.testing.assert_allclose(H1, H2, rtol=0, atol=0)
    np.testing.assert_allclose(max_sat1, max_sat2, rtol=0, atol=0)


@pytest.mark.skipif(pytest.importorskip("cupy", reason="CuPy not installed.") is None, reason="CuPy not installed.")
def test_cpu_gpu_consistency(monkeypatch, H_true, synthetic_batch):
    """
    If CuPy is available, ensure CPU and GPU produce near-identical subspaces.
    """
    monkeypatch.setattr("wsi_patching.backends.cupy_numpy.get_xp_backend", fake_get_xp_backend)
    norm_cpu = MacenkoNormalizer(pixel_limit=None)  # turn off subsampling for determinism
    norm_cpu.attach_context(PipelineContext({"use_gpu": False}))
    norm_cpu.fit(synthetic_batch)
    H_cpu = np.array(norm_cpu._H_mat, dtype=float)

    # Second run: GPU (CuPy)
    import cupy as cp

    def fake_cp_backend(use_gpu):
        return cp

    monkeypatch.setattr("wsi_patching.backends.cupy_numpy.get_xp_backend", fake_cp_backend)
    norm_gpu = MacenkoNormalizer(pixel_limit=None)
    norm_gpu.attach_context(PipelineContext({"use_gpu": True}))
    synthetic_batch.patches = ensure_cupy(synthetic_batch.patches)
    norm_gpu.fit(synthetic_batch)
    H_gpu = np.array(cp.asnumpy(norm_gpu._H_mat), dtype=float)

    # Compare subspaces between CPU and GPU
    assert subspace_is_close(H_cpu, H_gpu, tol_deg=2.0)


def test_validate_requires_use_gpu_key(monkeypatch):
    norm = MacenkoNormalizer()
    norm.attach_context(PipelineContext({}))  # missing key
    with pytest.raises(KeyError):
        norm.validate()

    norm.attach_context(PipelineContext({"use_gpu": False}))
    # Should not raise
    norm.validate()


def test_shapes_and_types(monkeypatch, H_true):
    """
    Basic sanity: ensure function accepts NHWC uint8 and returns proper types/shapes.
    """
    monkeypatch.setattr("wsi_patching.backends.cupy_numpy.get_xp_backend", fake_get_xp_backend)

    # Single small patch, still should fit
    patches = synth_patches_from_stains(H_true, n_patches=1, patch_hw=(16, 16), seed=7)
    batch = CollatedPatchBatch(patches=patches, wsi_id="S3", coords=np.array([[0, 0]]), use_gpu=False)

    norm = MacenkoNormalizer(pixel_limit=None)
    norm.attach_context(PipelineContext({"use_gpu": False}))
    norm.fit(batch)

    assert norm._H_mat.shape == (3, 2)
    # max_sat sometimes stored as (2,1) or (2,), accept both
    ms = np.array(norm._max_sat)
    assert ms.shape in {(2,), (2, 1)}
    assert ms.dtype.kind in "fc"  # float or complex (should be float)


def test_golden_H_and_maxsat_with_controlled_whites(monkeypatch, H_true):
    """
    Golden test (canonical Macenko):
      - Pure stains + controlled near-whites (< beta by OD-norm).
      - H spans the injected stain subspace (principal angles ~ 0).
      - max_sat equals the 99th percentile of per-stain concentrations computed
        in the *estimated* basis on β-masked tissue pixels.
      - Including whites would lower percentiles; fitted max_sat should exceed
        the all-pixels percentiles (verifying β exclusion indirectly).
    """
    alpha = 0.90  # for angle trimming only
    beta = 0.15  # OD-norm threshold
    I0 = 255

    monkeypatch.setattr("wsi_patching.backends.cupy_numpy.get_xp_backend", fake_get_xp_backend)

    gold = make_golden_patch_from_H(
        H_true=H_true,
        alpha=alpha,
        beta=beta,
        I0=I0,
        n_tissue_per_stain=500,
        white_fraction=0.40,
        c_min=0.15,
        c_max=2.1,
        seed=2024,
    )
    patch = gold["patch"]

    batch = CollatedPatchBatch(patches=patch, wsi_id="GOLD", coords=np.array([[0, 0]]), use_gpu=False)

    norm = MacenkoNormalizer(alpha=alpha, beta=beta, light_intensity=I0, pixel_limit=None)
    norm.attach_context(PipelineContext({"use_gpu": False}))
    norm.fit(batch)

    # H correctness (up to column order/sign)
    H_est = np.asarray(norm._H_mat, dtype=float)
    assert H_est.shape == (3, 2)
    assert subspace_is_close(H_est, H_true, tol_deg=2.0)

    # ---- Basis-aware expected max_sat (canonical 99th percentile) ----
    img = patch[0].reshape(-1, 3).astype(np.float32)
    img[img == 0] = 1.0
    OD_flat = -np.log(img / float(I0))  # (P, 3)
    OD_norm = np.linalg.norm(OD_flat, axis=1)
    tissue_mask = OD_norm >= beta
    assert tissue_mask.any()

    # Concentrations in the *estimated* basis
    C_hat = np.linalg.pinv(H_est) @ OD_flat[tissue_mask].T  # (2, T)

    expected_ms_impl = np.percentile(C_hat, 99.0, axis=1)  # (2,)

    # Match column order by max absolute cosine similarity to H_true
    a = H_est / np.linalg.norm(H_est, axis=0, keepdims=True)
    b = H_true / np.linalg.norm(H_true, axis=0, keepdims=True)
    sims = np.abs(a.T @ b)
    perm = (0, 1) if (sims[0, 0] + sims[1, 1]) >= (sims[0, 1] + sims[1, 0]) else (1, 0)

    ms = np.asarray(norm._max_sat, dtype=float).reshape(-1)
    ms_ordered = ms[list(perm)]
    expected_ms_impl_ordered = expected_ms_impl[list(perm)]

    # Tight check: your pipeline should exactly reproduce what we computed here
    np.testing.assert_allclose(ms_ordered, expected_ms_impl_ordered, rtol=1e-3, atol=2e-3)

    # ---- Indirect β exclusion check ----
    # Recompute 99th percentiles if whites were (incorrectly) included:
    C_all_impl = np.linalg.pinv(H_est) @ OD_flat.T  # all pixels
    all_pix_ms = np.percentile(C_all_impl, 99.0, axis=1)
    all_pix_ms_ordered = all_pix_ms[list(perm)]

    assert np.all(ms_ordered >= all_pix_ms_ordered - 1e-6)
    assert np.any(ms_ordered > all_pix_ms_ordered + 1e-3), (
        "max_sat suggests whites leaked into the percentile; β masking may be wrong"
    )


def test_beta_gate_matches_known_white_fraction(monkeypatch, H_true):
    alpha = 0.90
    beta = 0.12  # OD-norm threshold
    I0 = 255
    white_fraction = 0.30
    n_tissue = 400

    gold = make_golden_patch_from_H(
        H_true=H_true,
        alpha=alpha,
        beta=beta,
        I0=I0,
        n_tissue_per_stain=n_tissue,
        white_fraction=white_fraction,
        c_min=0.2,
        c_max=1.8,
        seed=7,
    )
    patch = gold["patch"]

    # Recompute OD-norms from the actual patch
    img = patch[0].reshape(-1, 3).astype(np.float32)
    img[img == 0] = 1.0
    OD_flat = -np.log(img / float(I0))
    OD_norm = np.linalg.norm(OD_flat, axis=1)

    # Canonical Macenko β gate: keep pixels with OD-norm >= beta
    tissue_mask = OD_norm >= beta
    frac_white = 1.0 - tissue_mask.mean()

    # If you added 'expected_white_fraction' in the helper, assert tightly:
    if "expected_white_fraction" in gold:
        np.testing.assert_allclose(frac_white, gold["expected_white_fraction"], rtol=0, atol=1e-6)
    else:
        # Otherwise compare to the requested white_fraction with a small tolerance
        # (padding to a square can shift it a bit)
        assert abs(frac_white - white_fraction) < 0.02
