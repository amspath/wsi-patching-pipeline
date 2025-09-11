from collections.abc import Iterable

import torch

from wsi_patching.core import Stage
from wsi_patching.typing import Sample


class GPUOps(Stage):
    """
    Micro-batched GPU ops. The GPU process is responsible for *collecting* batches
    from producers (size/timeout). This stage expects a batch and returns a batch.
    Default kernel is a no-op (copy to device, back to host).
    """

    placement = "gpu"
    function = "gpu_ops"

    def __init__(self, device: int = 0, batch_size: int = 200, batch_timeout_ms: int = 75):
        self.device = device
        self.batch_size = int(batch_size)
        self.batch_timeout_ms = int(batch_timeout_ms)

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        # This stage is only executed inside the GPU process loop (see gpu_process_main),
        # where we already coalesce input items into batches of the configured size/timeout.
        # Here we simply perform the "GPU work" on the batch and yield the same batch.
        if torch is None:
            # No torch installed: just pass batches through unchanged
            for item in it:
                yield item
            return

        device = torch.device(f"cuda:{self.device}") if torch.cuda.is_available() else torch.device("cpu")

        for item in it:
            # Expect item to be {"batch": [Sample, ...]}
            batch = item.get("batch", [])
            if not batch:
                continue
            # Stack into tensor (N,H,W,C) -> (N,C,H,W)
            patches = [torch.as_tensor(s["patch"]) for s in batch]
            x = torch.stack(patches, dim=0)  # (N,H,W,C)
            if x.ndim == 3:
                x = x.unsqueeze(-1)
            x = x.permute(0, 3, 1, 2).contiguous()  # (N,C,H,W)
            x = x.to(device, non_blocking=True)

            # ===== TODO: your real kernels here =====
            # e.g., stain normalization, color space, normalization, etc.
            # For now: simple pass-through (identity)
            y = x
            # ========================================

            # Move back to CPU as (N,H,W,C), write back into samples
            y_cpu = y.detach().to("cpu")
            y_cpu = y_cpu.permute(0, 2, 3, 1).contiguous()
            for i, s in enumerate(batch):
                s["patch"] = y_cpu[i].numpy()  # replace with processed patch
            yield {"batch": batch}
