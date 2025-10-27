from typing import Iterable, Literal, Tuple

import numpy as np

from wsi_patching.backends.cupy_numpy import ensure_numpy
from wsi_patching.core.types.types import CollatedPatchBatch
from wsi_patching.writers.stream_writers.stream_writer_base import StreamWriterBase


class NumpyStreamWriter(StreamWriterBase):
    """
    Returns the collated patches as numpy arrays in a streaming fashion.
    """

    def __init__(self, layout: Literal["NCHW", "NHWC"] = "NCHW", dtype: np.dtype = np.float32) -> None:
        super().__init__()
        self.layout = layout
        self.dtype = dtype

    def stream(self, batch: CollatedPatchBatch) -> Iterable[Tuple[str, np.ndarray, np.ndarray, dict]]:
        self.log.info(f"Received batch from wsi: {batch.wsi_id} size: {len(batch.patches)}")

        # coords -> np.int64
        coords_np = np.asarray(batch.coords, dtype=np.int64)

        # patches -> np.float (from numpy or cupy)
        images_np = ensure_numpy(batch.patches)
        images_np = np.asarray(images_np, dtype=self.dtype)

        # store as requested layout (assume input is BHWC)
        if self.layout == "NCHW":
            # BHWC -> BCHW
            images_np = np.transpose(images_np, (0, 3, 1, 2))

        # yield
        yield batch.wsi_id, images_np, coords_np, batch.metadata.get_all_row_wise()
