from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np
    import torch


def get_torch_device(use_gpu: bool) -> torch.device:
    try:
        import torch
    except ImportError:
        raise ImportError(
            "torch is required to use GPU device selection. Install it with: pip install wsi-patching[gpu]"
        )
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    elif use_gpu and torch.backends.mps.is_available():
        return torch.device("mps")
    elif use_gpu:
        raise RuntimeError("Requested GPU device, but no CUDA or MPS backend is available.")
    else:
        return torch.device("cpu")


def to_torch_from_xp(
    array: np.ndarray | cp.ndarray, device: torch.device, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Convert a NumPy or CuPy array to a PyTorch tensor on the specified device, with a specified dtype."""
    try:
        import torch
    except ImportError:
        raise ImportError(
            "torch is required to convert arrays to tensors. Install it with: pip install wsi-patching[gpu]"
        )
    if dtype is None:
        dtype = torch.float32
    import numpy as np

    if isinstance(array, np.ndarray):
        return torch.as_tensor(array, device=device, dtype=dtype)
    else:
        return torch.from_dlpack(array.toDlpack()).to(device=device, dtype=dtype)
