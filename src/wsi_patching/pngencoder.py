import io
import time
from typing import Iterable

from PIL import Image

from wsi_patching.core.pipeline import Stage
from wsi_patching.utils.profiling import get_current_profiler
from wsi_patching.utils.types import EncodedPatch, PatchBatch


class PNGEncoder(Stage):
    """
    Encodes patches to PNG bytes and flattens batches into single-sample items ready for the writer.
    Output items contain: "__key__", "sample_bytes", "json_bytes"
    """

    def validate(self) -> None:
        self.ctx.require_key("level")

    def __call__(self, it: Iterable[PatchBatch]) -> Iterable[EncodedPatch]:
        prof = get_current_profiler()
        for batch in it:
            for sample in batch.samples:
                t0 = time.perf_counter()
                key = f"{sample.wsi_id}-{sample.coord[0]}-{sample.coord[1]}"
                # PNG
                bio = io.BytesIO()
                Image.fromarray(sample.patch).save(bio, format="PNG")
                png_bytes = bio.getvalue()
                # JSON
                meta = {"wsi_id": sample.wsi_id, "coord": sample.coord, **sample.meta}

                dt = time.perf_counter() - t0
                if prof is not None:
                    prof.add_time("PNGEncoder.isolated", dt, yielded=True)
                yield EncodedPatch(key=key, patch_bytes=png_bytes, json_dict=meta)
