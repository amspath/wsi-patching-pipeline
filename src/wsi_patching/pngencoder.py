import io
import time
from typing import Iterable, List, Optional

from PIL import Image

from wsi_patching.pipeline import Sample, Stage
from wsi_patching.profiling import get_current_profiler


class PNGEncoder(Stage):
    """
    Encodes patches to PNG bytes and flattens batches into single-sample items ready for the writer.
    Output items contain: "__key__", "png_bytes", "json_bytes"
    """

    def validate(self) -> None:
        self.ctx.require_key("level")

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        prof = get_current_profiler()  # may be None if profiling is disabled
        for item in it:
            # Handle batched items
            if isinstance(item, dict) and "batch" in item:
                batch: List[Sample] = item["batch"]
                for s in batch:
                    t0 = time.perf_counter()
                    out = self._encode_one(s)
                    dt = time.perf_counter() - t0
                    if prof is not None and out is not None:
                        # record isolated encode time only (no upstream waiting)
                        prof.add_time("PNGEncoder.isolated", dt, yielded=True)
                    if out is not None:
                        yield out
                continue

            # Single item path
            if isinstance(item, dict) and item.get("type") == "sample":
                t0 = time.perf_counter()
                out = self._encode_one(item)
                dt = time.perf_counter() - t0
                if prof is not None and out is not None:
                    prof.add_time("PNGEncoder.isolated", dt, yielded=True)
                if out is not None:
                    yield out

    def _encode_one(self, s: Sample) -> Optional[Sample]:
        patch = s.get("patch")
        key = f"{s['wsi_id']}-{s['coord'][0]}-{s['coord'][1]}-L{self.ctx['level']}"

        # Encode to PNG
        buf = io.BytesIO()
        Image.fromarray(patch, mode="RGB").save(buf, format="PNG")
        png_bytes = buf.getvalue()

        # Build json sidecar (exclude heavy fields)
        meta = {k: v for k, v in s.items() if k not in ("patch",)}

        return {"__key__": key, "png_bytes": png_bytes, "json_bytes": meta}
