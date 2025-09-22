try:
    import cupy as cp

    _cupy_available = True
except ImportError:
    _cupy_available = False

from typing import Union

import numpy as np


def ensure_numpy(arr: Union[np.ndarray, "cp.ndarray"]) -> np.ndarray:
    """Ensure the input array is a NumPy ndarray. If it's a CuPy array, convert it to NumPy."""
    if _cupy_available and isinstance(arr, cp.ndarray):
        return arr.get()
    elif isinstance(arr, np.ndarray):
        return arr
    else:
        raise TypeError(f"Input must be a numpy.ndarray or cupy.ndarray, got {type(arr)}")


def ensure_cupy(arr: Union[np.ndarray, "cp.ndarray"]) -> "cp.ndarray":
    """Ensure the input array is a CuPy ndarray. If it's a NumPy array, convert it to CuPy."""
    if not _cupy_available:
        raise ImportError("CuPy is not available. Please install CuPy to use this function.")
    if isinstance(arr, np.ndarray):
        return cp.asarray(arr)
    elif isinstance(arr, cp.ndarray):
        return arr
    else:
        raise TypeError(f"Input must be a numpy.ndarray or cupy.ndarray, got {type(arr)}")


def get_xp_backend(use_gpu: bool):
    """Return either numpy or cupy to use as 'xp.asarray', etc."""
    if use_gpu:
        if not _cupy_available:
            raise ImportError("CuPy is not available. Please install CuPy to use GPU backend.")
        return cp
    else:
        return np
