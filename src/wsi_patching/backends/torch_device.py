import torch


def get_torch_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    elif use_gpu and torch.backends.mps.is_available():
        return torch.device("mps")
    elif use_gpu:
        raise RuntimeError("Requested GPU device, but no CUDA or MPS backend is available.")
    else:
        return torch.device("cpu")
