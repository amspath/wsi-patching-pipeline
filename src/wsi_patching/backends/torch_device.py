from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np


def get_torch_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    elif use_gpu and torch.backends.mps.is_available():
        return torch.device("mps")
    elif use_gpu:
        raise RuntimeError("Requested GPU device, but no CUDA or MPS backend is available.")
    else:
        return torch.device("cpu")


def to_torch_from_xp(array: "np.ndarray | cp.ndarray", device: torch.device, dtype=torch.float32) -> torch.Tensor:
    """Convert a NumPy or CuPy array to a PyTorch tensor on the specified device, with a specified dtype."""

    if isinstance(array, "np.ndarray"):
        # NumPy -> torch
        return torch.as_tensor(array, device=device, dtype=dtype)
    else:
        # Cupy -> torch
        return torch.from_dlpack(array.toDlpack(), device=device, dtype=dtype)
