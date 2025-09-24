import numpy as np
import pytest

import wsi_patching.backends.cupy_numpy as mod


# --- CuPy stub to avoid the real dependency -----------------
class DummyCuPyArray:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    # Mimic cupy.ndarray.get() -> numpy.ndarray
    def get(self):
        return np.array(self._arr)


class DummyCP:
    ndarray = DummyCuPyArray

    @staticmethod
    def asarray(x):
        return DummyCuPyArray(x)


def enable_dummy_cupy(monkeypatch):
    """Flip the module into 'CuPy available' mode using our stub."""
    monkeypatch.setattr(mod, "_cupy_available", True, raising=True)
    monkeypatch.setattr(mod, "cp", DummyCP, raising=False)


# ------------------- tests -------------------
def test_validate_xp_backend_raises_when_gpu_without_cupy(monkeypatch):
    monkeypatch.setattr(mod, "_cupy_available", False, raising=True)
    with pytest.raises(ImportError):
        mod.validate_xp_backend(use_gpu=True)

    # CPU path never raises
    mod.validate_xp_backend(use_gpu=False)


def test_ensure_numpy_with_numpy_passthrough():
    a = np.arange(6).reshape(2, 3)
    out = mod.ensure_numpy(a)
    assert out is a  # same object


def test_ensure_numpy_with_cupy_like(monkeypatch):
    enable_dummy_cupy(monkeypatch)
    cp_arr = DummyCuPyArray([[1, 2], [3, 4]])
    out = mod.ensure_numpy(cp_arr)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, np.array([[1, 2], [3, 4]]))


def test_ensure_numpy_rejects_other_types(monkeypatch):
    # even if cupy is enabled, non-ndarray/list should fail
    enable_dummy_cupy(monkeypatch)
    with pytest.raises(TypeError):
        mod.ensure_numpy([1, 2, 3])  # list is not supported


def test_ensure_cupy_raises_if_cupy_unavailable(monkeypatch):
    monkeypatch.setattr(mod, "_cupy_available", False, raising=True)
    with pytest.raises(ImportError):
        mod.ensure_cupy(np.zeros((2, 2), dtype=np.uint8))


def test_ensure_cupy_with_numpy_to_cupy(monkeypatch):
    enable_dummy_cupy(monkeypatch)
    a = np.arange(4).reshape(2, 2)
    out = mod.ensure_cupy(a)
    assert isinstance(out, DummyCuPyArray)
    np.testing.assert_array_equal(out.get(), a)


def test_ensure_cupy_with_cupy_passthrough(monkeypatch):
    enable_dummy_cupy(monkeypatch)
    cp_arr = DummyCuPyArray([9, 8, 7])
    out = mod.ensure_cupy(cp_arr)
    assert out is cp_arr


def test_get_xp_backend_returns_np_or_cp(monkeypatch):
    # CPU -> numpy
    monkeypatch.setattr(mod, "_cupy_available", False, raising=True)
    assert mod.get_xp_backend(use_gpu=False) is np

    # GPU -> cp when available
    enable_dummy_cupy(monkeypatch)
    assert mod.get_xp_backend(use_gpu=True) is DummyCP

    # GPU -> error when unavailable
    monkeypatch.setattr(mod, "_cupy_available", False, raising=True)
    with pytest.raises(ImportError):
        mod.get_xp_backend(use_gpu=True)
