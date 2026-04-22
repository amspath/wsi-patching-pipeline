from contextlib import ExitStack
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from wsi_patching.backends.torch_device import get_torch_device


def _patch_cuda_mps(cuda_avail: bool, mps_avail: bool):
    """Context manager that patches the availability checks directly."""
    stack = ExitStack()
    stack.enter_context(patch("torch.cuda.is_available", return_value=cuda_avail))
    stack.enter_context(patch("torch.backends.mps.is_available", return_value=mps_avail))
    return stack


def test_returns_cpu_when_use_gpu_false():
    assert get_torch_device(use_gpu=False).type == "cpu"


def test_prefers_cuda_when_available():
    with _patch_cuda_mps(cuda_avail=True, mps_avail=True):
        dev = get_torch_device(use_gpu=True)
        assert isinstance(dev, torch.device)
        assert dev.type == "cuda"  # CUDA preferred when both available


def test_returns_mps_when_no_cuda_but_mps_available():
    with _patch_cuda_mps(cuda_avail=False, mps_avail=True):
        dev = get_torch_device(use_gpu=True)
        assert dev.type == "mps"


def test_raises_when_gpu_requested_but_none_available():
    with _patch_cuda_mps(cuda_avail=False, mps_avail=False):
        with pytest.raises(RuntimeError, match="no CUDA or MPS backend is available"):
            get_torch_device(use_gpu=True)
