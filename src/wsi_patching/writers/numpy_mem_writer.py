from typing import List, Literal, Tuple

import numpy as np

from wsi_patching.backends.cupy_numpy import ensure_numpy
from wsi_patching.core.types.types import CollatedPatchBatch
from wsi_patching.writers.writer_base import WriterBase


class NumpyMemoryWriter(WriterBase):
    """
    Collects CollatedPatchBatch (assumed BCHW) and builds an in-memory NumPy dataset.
    Eagerly copies to float32 NumPy arrays; no torch involved.
    """

    def __init__(self, layout: Literal["NCHW", "NHWC"] = "NCHW", dtype: np.dtype = np.float32) -> None:
        super().__init__()
        self.layout = layout
        self.dtype = dtype

        self._images_chunks: List[np.ndarray] = []
        self._coords_chunks: List[np.ndarray] = []
        self.meta = []

        self.wsi_ids: List[str] = []

        self.final_images, self.final_coords = None, None

        self.log.info("Initialized. NOTE: Memory heavy — stores all patches as float32 NumPy arrays in RAM.")

    # --- WriterBase hooks ---
    def open(self) -> None:
        self.log.info("Opening... layout=%s dtype=%s", self.layout, self.dtype)

    def write(self, sample: CollatedPatchBatch) -> None:
        self.log.info(f"Received batch from wsi: {sample.wsi_id} size: {len(sample.patches)}")

        # coords -> np.int64
        coords_np = np.asarray(sample.coords, dtype=np.int64)

        # patches -> np.float (from numpy or cupy)
        images_np = ensure_numpy(sample.patches)
        images_np = np.asarray(images_np, dtype=self.dtype)

        # store as requested layout (assume input is BHWC)
        if self.layout == "NCHW":
            # BHWC -> BCHW
            images_np = np.transpose(images_np, (0, 3, 1, 2))

        # accumulate
        self._images_chunks.append(images_np)
        self._coords_chunks.append(coords_np)
        self.meta.extend(sample.metadata.get_all_row_wise())
        self.wsi_ids.extend([sample.wsi_id] * images_np.shape[0])

    def close(self) -> None:
        if self.final_images is not None:
            return

        if not self._images_chunks:
            self.final_images = np.empty((0, 1, 1, 1), dtype=self.dtype)
            self.final_coords = np.empty((0, 2), dtype=np.int64)
            self.log.info("Closed with empty dataset.")
            return

        self.final_images = np.concatenate(self._images_chunks, axis=0)
        self.final_coords = np.concatenate(self._coords_chunks, axis=0)

        # free chunks
        self._images_chunks.clear()
        self._coords_chunks.clear()

        self.log.info(
            "Closed. Final dataset: N=%d, shape=%s, layout=%s, dtype=%s",
            len(self.final_images),
            tuple(self.final_images.shape),
            self.layout,
            self.final_images.dtype,
        )

    def get_output(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[dict]]:
        if self.final_images is None:
            self.close()
        assert self.final_images is not None
        return self.wsi_ids, self.final_images, self.final_coords, self.meta
