import io
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List

import numpy as np
from PIL import Image

from wsi_patching.backends.cupy_numpy import ensure_numpy
from wsi_patching.core.pipeline import Stage
from wsi_patching.core.types.types import CollatedPatchBatch, EncodedCollatedPatchBatch


class PNGEncoder(Stage):
    """
    Encodes patches to PNG bytes and flattens batches into single-sample items ready for the writer.
    Output items contain: "__key__", "sample_bytes", "json_bytes"

    Speedups vs. baseline:
    - Threaded encoding across patches (libpng releases GIL).
    - Lower compress_level (1) for big throughput gains.
    - Use frombuffer for 'L'/'RGB'/'RGBA' to avoid an extra copy.
    - Ensure contiguous uint8 to keep the fast path.
    """

    def __init__(self, compress_level: int = 1, threads: int | None = None):
        # compress_level: 0 (no compression) .. 9 (max). 1–3 is a good speed/size tradeoff.
        self.compress_level = int(compress_level)
        self.threads = threads or os.cpu_count() or 8

    @staticmethod
    def _to_uint8_numpy(patch) -> np.ndarray:
        # Move from CuPy to NumPy if needed, then ensure contiguous uint8.
        arr = ensure_numpy(patch)
        if arr.dtype != np.uint8:
            # clip + astype in C (fast)
            arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
        # some GPU→CPU transfers yield non-contiguous; enforce C-order
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        return arr

    def _encode_one(self, patch) -> bytes:
        arr = self._to_uint8_numpy(patch)
        h, w = arr.shape[:2]

        # Try zero-copy Image.frombuffer for common modes
        im = None
        if arr.ndim == 2:
            # Grayscale
            im = Image.frombuffer("L", (w, h), arr, "raw", "L", 0, 1)
        elif arr.ndim == 3 and arr.shape[2] in (3, 4):
            mode = "RGB" if arr.shape[2] == 3 else "RGBA"
            im = Image.frombuffer(mode, (w, h), arr, "raw", mode, 0, 1)
        else:
            # Fallback if unusual shape
            im = Image.fromarray(arr)

        bio = io.BytesIO()
        # Avoid optimize=True (slower), set only compress_level
        im.save(bio, format="PNG", compress_level=self.compress_level)
        return bio.getvalue()

    def __call__(self, it: Iterable[CollatedPatchBatch]) -> Iterable[EncodedCollatedPatchBatch]:
        # Single thread pool reused across all batches in this call
        with ThreadPoolExecutor(max_workers=self.threads) as ex:
            for batch in it:
                # ex.map preserves order → aligns with batch.coords/indexing
                encoded: List[bytes] = list(ex.map(self._encode_one, batch.patches))
                yield EncodedCollatedPatchBatch.from_collated_patch_batch(
                    batch, encoding="PNG", encoded_patches=encoded
                )
