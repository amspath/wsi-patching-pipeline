import io
import time
from typing import Iterable

from PIL import Image

from wsi_patching.core.pipeline import Stage
from wsi_patching.utils.profiling import get_current_profiler
from wsi_patching.utils.types import CollatedPatchBatch, EncodedPatch


class PNGEncoder(Stage):
    """
    Encodes patches to PNG bytes and flattens batches into single-sample items ready for the writer.
    Output items contain: "__key__", "sample_bytes", "json_bytes"
    """

    def __call__(self, it: Iterable[CollatedPatchBatch]) -> Iterable[EncodedPatch]:
        prof = get_current_profiler()
        for batch in it:
            for idx in range(len(batch.coords)):
                t0 = time.perf_counter()
                key = f"{batch.wsi_id}-{batch.coords[idx][0]}-{batch.coords[idx][1]}"
                # PNG
                bio = io.BytesIO()
                Image.fromarray(batch.patches[idx]).save(bio, format="PNG")
                png_bytes = bio.getvalue()
                # JSON
                meta = {"wsi_id": batch.wsi_id, "coord": batch.coords[idx], **batch.meta}

                dt = time.perf_counter() - t0
                if prof is not None:
                    prof.add_time("PNGEncoder.isolated", dt, yielded=True)
                yield EncodedPatch(key=key, patch_bytes=png_bytes, json_dict=meta)
