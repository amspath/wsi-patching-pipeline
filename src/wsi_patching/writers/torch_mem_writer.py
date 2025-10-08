from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch

from wsi_patching.backends.torch_device import get_torch_device
from wsi_patching.core.types.types import CollatedPatchBatch
from wsi_patching.writers.writer_base import WriterBase

if TYPE_CHECKING:
    import cupy as cp


class InMemoryPatchDataset(torch.utils.data.Dataset):
    """In-memory dataset"""

    def __init__(
        self,
        images: torch.Tensor,
        coords: torch.Tensor,
        wsi_ids: List[str],
        metadata: List[Dict[str, Any]],
        layout: Literal["NCHW", "NHWC"] = "NCHW",
    ) -> None:
        assert images.ndim == 4, f"images must be 4D tensor, got {images.shape}"
        assert coords.ndim == 2 and coords.shape[1] == 2, f"coords must be [N,2], got {coords.shape}"
        assert images.shape[0] == coords.shape[0] == len(wsi_ids), "length mismatch among images/coords/wsi_ids"
        self.images = images
        self.coords = coords
        self.wsi_ids = wsi_ids
        self.metadata = metadata
        self.layout = layout

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "image": self.images[idx],
            "coord": self.coords[idx],
            "wsi_id": self.wsi_ids[idx],
            "meta": self.metadata[idx],
        }


class TorchMemoryWriter(WriterBase):
    """
    Collects CollatedPatchBatch (assumed [B, C, H, W]) and builds an in-memory torch Dataset.
    Eagerly tensorizes to float32 on CPU/GPU (based on ctx['use_gpu']).
    """

    def __init__(self, layout: Literal["NCHW", "NHWC"] = "NCHW", dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.layout = layout
        self.dtype = dtype

        # Accumulators
        self._images_chunks: List[torch.Tensor] = []
        self._coords_chunks: List[torch.Tensor] = []
        self._wsi_ids: List[str] = []
        self._metadata: List[Dict[str, Any]] = []

        self._dataset: Optional[InMemoryPatchDataset] = None
        self._device: Optional[torch.device] = None

        self.log.info("Initialized. NOTE: Memory heavy — stores all patches as float32 tensors in RAM/VRAM.")

    def validate(self) -> None:
        self.ctx.require_key("use_gpu")

    # --- WriterBase hooks ---
    def open(self) -> None:
        self.log.info("Opening...")
        self._device = get_torch_device(self.ctx["use_gpu"])
        self.log.info("device=%s, layout=%s, dtype=%s", self._device, self.layout, self.dtype)

    def write(self, batch: CollatedPatchBatch) -> None:
        self.log.info(f"Received batch from wsi: {batch.wsi_id} size: {len(batch.patches)}")

        # coords -> torch.long on target device
        coords_t = torch.as_tensor(batch.coords, dtype=torch.long, device=self._device)

        # patches -> torch.float32 on target device
        images_t = self._to_tensor(batch.patches, device=self._device, dtype=self.dtype)

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
        self._images_chunks.append(images_t)
        self._coords_chunks.append(coords_t)
        self._metadata.extend(batch.metadata.get_all_row_wise())
        self._wsi_ids.extend([batch.wsi_id] * images_t.shape[0])

    def close(self) -> None:
        self.log.info("Closing ...")
        if self._dataset is not None:
            return  # already finalized

        if not self._images_chunks:
            # empty dataset
            dev = self._device if self._device is not None else torch.device("cpu")
            empty_imgs = torch.empty((0, 1, 1, 1), dtype=self.dtype, device=dev)
            empty_coords = torch.empty((0, 2), dtype=torch.long, device=dev)
            self._dataset = InMemoryPatchDataset(empty_imgs, empty_coords, [], [], layout=self.layout)
            self.log.info("Closed with empty dataset.")
            return

        images = torch.cat(self._images_chunks, dim=0)
        coords = torch.cat(self._coords_chunks, dim=0)

        # free chunk lists
        self._images_chunks.clear()
        self._coords_chunks.clear()

        self._dataset = InMemoryPatchDataset(
            images=images, coords=coords, wsi_ids=self._wsi_ids, metadata=self._metadata, layout=self.layout
        )
        self.log.info(
            f"Closed. Final dataset: N={len(self._dataset)}, "
            f"shape={tuple(self._dataset.images.shape)}, "
            f"device={self._dataset.images.device}, layout={self.layout}"
        )

    def get_output(self) -> torch.utils.data.Dataset:
        if self._dataset is None:
            self.close()
        return self._dataset

    # --- helpers ---
    def _to_tensor(
        self, arr: Union[np.ndarray, "cp.ndarray"], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Convert numpy/cupy BCHW array to torch.Tensor on desired device & dtype (eager)."""
        # move/cast in one go
        return torch.as_tensor(arr).to(device=device, dtype=dtype, non_blocking=(device.type in ("cuda", "mps")))
