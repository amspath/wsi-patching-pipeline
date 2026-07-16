from typing import Iterable

import numpy as np

from wsi_patching.core.pipeline import Stage
from wsi_patching.core.types.types import CollatedPatchBatch


class RemoveEdgeTiles(Stage):
    """
    Drop tiles that lie within ``depth`` tiles of the whole-slide image border.

    A tile at top-left coordinate ``(x, y)`` (in the patched-resolution pixel space)
    is kept iff::

        depth * tile_size <= x < W - depth * tile_size
        depth * tile_size <= y < H - depth * tile_size

    where ``(W, H)`` are the WSI dimensions at the patched resolution, read from
    ``batch.wsi_dims``; ``tile_size`` is read from the pipeline context. This makes the
    filter correct regardless of batch/region layout, unlike inferring the slide size
    from a single batch's coordinates.

    Parameters
    ----------
    depth : int, default 1
        Number of tile rings to remove along every border. ``depth=0`` is a no-op.
    """

    def __init__(self, *, depth: int = 1):
        if depth < 0:
            raise ValueError("depth must be >= 0")
        self.depth = int(depth)

    def validate(self) -> None:
        self.ctx.require_key("tile_size")

    def __call__(self, it: Iterable[CollatedPatchBatch]) -> Iterable[CollatedPatchBatch]:
        tile_size = int(self.ctx["tile_size"])
        margin = self.depth * tile_size

        for batch in it:
            if self.depth == 0:
                yield batch
                continue

            wsi_width, wsi_height = batch.wsi_dims
            coords = batch.coords  # (N, 2) int, top-left (x, y) at patched resolution

            keep_mask = (
                (coords[:, 0] >= margin)
                & (coords[:, 0] < wsi_width - margin)
                & (coords[:, 1] >= margin)
                & (coords[:, 1] < wsi_height - margin)
            )

            in_sz = len(batch.patches)
            batch.filter_on_mask(np.asarray(keep_mask, dtype=np.bool_))
            self.log.info(
                f"wsi={batch.wsi_id} batch_in={in_sz} batch_out={len(batch.patches)} "
                f"(depth={self.depth}, tile_size={tile_size})"
            )

            if len(batch.patches) == 0:
                continue

            yield batch
