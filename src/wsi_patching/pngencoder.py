import io
import time
from typing import Iterable

import cupy as cp
from PIL import Image

from wsi_patching.core.pipeline import Stage
from wsi_patching.utils.profiling import get_current_profiler
from wsi_patching.utils.types import CollatedPatchBatch, Patch


class PNGEncoder(Stage):
    """
    Encodes patches to PNG bytes and flattens batches into single-sample items ready for the writer.
    Output items contain: "__key__", "sample_bytes", "json_bytes"

    Input is assumed to be uint8 convertable (i.e. between 0-255); no checks are performed.
    """

    def __call__(self, it: Iterable[CollatedPatchBatch]) -> Iterable[Patch]:
        prof = get_current_profiler()
        for batch in it:
            for idx in range(len(batch.coords)):
                t0 = time.perf_counter()
                key = f"{batch.wsi_id}-{batch.coords[idx][0]}-{batch.coords[idx][1]}"

                # If not uint8, convert now
                if batch.patches.dtype != "uint8":
                    patch_uint8 = (batch.patches[idx].clip(0, 255)).astype("uint8")
                else:
                    patch_uint8 = batch.patches[idx]

                if isinstance(patch_uint8, cp.ndarray):
                    patch_uint8 = patch_uint8.get()

                # PNG
                bio = io.BytesIO()
                Image.fromarray(patch_uint8).save(bio, format="PNG")
                png_bytes = bio.getvalue()
                # JSON
                meta = {"wsi_id": batch.wsi_id, "coord": batch.coords[idx], **batch.meta}

                dt = time.perf_counter() - t0
                if prof is not None:
                    prof.add_time("PNGEncoder.isolated", dt, yielded=True)
                yield Patch(key=key, patch=png_bytes, meta=meta)
