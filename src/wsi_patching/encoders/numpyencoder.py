from typing import Iterable

from wsi_patching.backends.cupy_numpy import ensure_numpy
from wsi_patching.core.pipeline import Stage
from wsi_patching.core.types.types import CollatedPatchBatch, Patch


class NumpyEncoder(Stage):
    """
    Transforms CollatedPatchBatch items into individual Patch items with numpy array bytes.
    Output items contain: "__key__", "sample_bytes", "json_bytes"

    Input is assumed to be uint8 convertable (i.e. between 0-255); no checks are performed.
    """

    def __call__(self, it: Iterable[CollatedPatchBatch]) -> Iterable[Patch]:
        for batch in it:
            for idx in range(len(batch.coords)):
                wsi_id, coord, patch, meta = batch.get(idx)
                key = f"{wsi_id}-{coord[0]}-{coord[1]}"

                # If not uint8, convert now
                if patch.dtype != "uint8":
                    patch_uint8 = (patch.clip(0, 255)).astype("uint8")
                else:
                    patch_uint8 = patch

                patch_uint8 = ensure_numpy(patch_uint8)

                # JSON
                meta["coord"] = coord
                yield Patch(key=key, patch=patch_uint8, meta=meta)
