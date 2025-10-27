from typing import TYPE_CHECKING, Iterable, Literal, Optional, Tuple, Union

import numpy as np
import torch

from wsi_patching.core.types.types import CollatedPatchBatch
from wsi_patching.writers.stream_writers.stream_writer_base import StreamWriterBase

if TYPE_CHECKING:
    import cupy as cp


class TorchStreamWriter(StreamWriterBase):
    """Returns the collated patches as numpy arrays in a streaming fashion."""

    def __init__(
        self,
        layout: Literal["NCHW", "NHWC"] = "NCHW",
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.layout = layout
        self.dtype = dtype
        self.device = device

    def validate(self) -> None:
        self.ctx.require_key("use_gpu")

        if self.device is None:
            self.device = torch.device("cuda" if self._ctx["use_gpu"] else "cpu")

    def stream(self, batch: Iterable[CollatedPatchBatch]) -> Iterable[Tuple[str, torch.Tensor, torch.Tensor, dict]]:
        self.log.info(f"Received batch from wsi: {batch.wsi_id} size: {len(batch.patches)}")

        # coords -> torch.long on target device
        coords_t = torch.as_tensor(batch.coords, dtype=torch.long, device=self.device)

        # patches -> torch.float32 on target device
        images_t = self._to_tensor(batch.patches, device=self.device, dtype=self.dtype)

        # expect BHWC; permute if user requested NCHW (default)
        if self.layout == "NCHW":
            images_t = images_t.permute(0, 3, 1, 2).contiguous()

        # basic length check
        if images_t.shape[0] != coords_t.shape[0]:
            raise ValueError(
                f"Batch length mismatch: images N={images_t.shape[0]} vs coords N={coords_t.shape[0]} "
                f"(wsi_id={batch.wsi_id})"
            )

        # accumulate
        yield batch.wsi_id, images_t, coords_t, batch.metadata.get_all_row_wise()

    def _to_tensor(
        self, arr: Union[np.ndarray, "cp.ndarray"], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Convert numpy/cupy BCHW array to torch.Tensor on desired device & dtype (eager)."""
        # move/cast in one go
        return torch.as_tensor(arr).to(device=device, dtype=dtype, non_blocking=(device.type in ("cuda", "mps")))
