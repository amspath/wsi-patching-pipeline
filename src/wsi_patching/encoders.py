import io
from collections.abc import Iterable

import imageio.v3 as iio

from wsi_patching.core import Stage
from wsi_patching.typing import Sample


def ensure_png_bytes(img_arr) -> bytes:
    if iio is None:
        raise RuntimeError("imageio.v3 not available. Install imageio to encode PNG.")
    buf = io.BytesIO()
    # Do not specify colors/styles; keep defaults for speed.
    iio.imwrite(buf, img_arr, extension=".png")
    return buf.getvalue()


class PNGEncoder(Stage):
    """
    Encode each sample in the batch to PNG bytes (fast path).
    Emits individual encoded samples (unbatched) to the writer queue.
    """

    placement = "gpu"

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        for item in it:
            batch = item.get("batch", [])
            for s in batch:
                png_bytes = ensure_png_bytes(s["patch"])
                meta = {k: v for k, v in s.items() if k != "patch"}
                yield {**meta, "png": png_bytes}
