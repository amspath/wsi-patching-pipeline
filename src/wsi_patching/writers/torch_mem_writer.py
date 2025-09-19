import logging
from typing import Any, Dict, List, Literal, Optional, Union

import cupy as cp  # optional
import numpy as np
import torch

from wsi_patching.core.pipeline import WriterBase
from wsi_patching.utils.types import CollatedPatchBatch


class InMemoryPatchDataset(torch.utils.data.Dataset):
    """In-memory dataset"""

    def __init__(
        self, images: torch.Tensor, coords: torch.Tensor, wsi_ids: List[str], layout: Literal["NCHW", "NHWC"] = "NCHW"
    ) -> None:
        assert images.ndim == 4, f"images must be 4D tensor, got {images.shape}"
        assert coords.ndim == 2 and coords.shape[1] == 2, f"coords must be [N,2], got {coords.shape}"
        assert images.shape[0] == coords.shape[0] == len(wsi_ids), "length mismatch among images/coords/wsi_ids"
        self.images = images
        self.coords = coords
        self.wsi_ids = wsi_ids
        self.layout = layout

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"image": self.images[idx], "coord": self.coords[idx], "wsi_id": self.wsi_ids[idx]}


class TorchMemoryWriter(WriterBase):
    """
    Collects CollatedPatchBatch (assumed [B, C, H, W]) and builds an in-memory torch Dataset.
    Eagerly tensorizes to float32 on CPU/GPU (based on ctx['use_gpu']).
    """

    input_type = CollatedPatchBatch

    def __init__(self, layout: Literal["NCHW", "NHWC"] = "NCHW", dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.layout = layout
        self.dtype = dtype

        # Accumulators
        self._images_chunks: List[torch.Tensor] = []
        self._coords_chunks: List[torch.Tensor] = []
        self._wsi_ids: List[str] = []

        self._dataset: Optional[InMemoryPatchDataset] = None
        self._device: Optional[torch.device] = None

        logging.info(
            "[TorchMemoryWriter] Initialized. NOTE: Memory heavy — stores all patches as float32 tensors in RAM/VRAM."
        )

    def validate(self) -> None:
        self.ctx.require_key("use_gpu")

    # --- WriterBase hooks ---
    def open(self) -> None:
        logging.info("TorchMemoryWriter opening...")
        use_gpu = self.ctx["use_gpu"]
        self._device = torch.device("cuda") if use_gpu and torch.cuda.is_available() else torch.device("cpu")
        logging.info("TorchMemoryWriter device=%s, layout=%s, dtype=%s", self._device, self.layout, self.dtype)

    def write(self, batch: CollatedPatchBatch) -> None:
        logging.info(f"[writer] Received batch from wsi: {batch.wsi_id} size: {len(batch.patches)}")

        if batch is None:
            return

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
        self._wsi_ids.extend([batch.wsi_id] * images_t.shape[0])

    def close(self) -> None:
        if self._dataset is not None:
            return  # already finalized

        if not self._images_chunks:
            # empty dataset
            dev = self._device if self._device is not None else torch.device("cpu")
            empty_imgs = torch.empty((0, 1, 1, 1), dtype=self.dtype, device=dev)
            empty_coords = torch.empty((0, 2), dtype=torch.long, device=dev)
            self._dataset = InMemoryPatchDataset(empty_imgs, empty_coords, [], layout=self.layout)
            logging.info("TorchMemoryWriter closed with empty dataset.")
            return

        images = torch.cat(self._images_chunks, dim=0)
        coords = torch.cat(self._coords_chunks, dim=0)

        # free chunk lists
        self._images_chunks.clear()
        self._coords_chunks.clear()

        self._dataset = InMemoryPatchDataset(images=images, coords=coords, wsi_ids=self._wsi_ids, layout=self.layout)
        logging.info(
            "TorchMemoryWriter closed. Final dataset: N=%d, shape=%s, device=%s, layout=%s",
            len(self._dataset),
            tuple(self._dataset.images.shape),
            self._dataset.images.device,
            self.layout,
        )

    def get_output(self) -> torch.utils.data.Dataset:
        if self._dataset is None:
            self.close()
        return self._dataset  # type: ignore[return-value]

    # --- helpers ---
    def _to_tensor(
        self,
        arr: Union[np.ndarray, "cp.ndarray"],  # type: ignore[name-defined]
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert numpy/cupy BCHW array to torch.Tensor on desired device & dtype (eager)."""
        if isinstance(arr, cp.ndarray):
            arr = cp.asnumpy(arr)

        if isinstance(arr, np.ndarray):
            t = torch.from_numpy(arr)
        else:
            # last resort: let torch try to wrap (will raise if unsupported)
            t = torch.as_tensor(arr)

        # move/cast in one go
        return t.to(device=device, dtype=dtype, non_blocking=(device.type == "cuda"))
