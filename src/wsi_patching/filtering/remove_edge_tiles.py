from typing import Iterable

import numpy as np

from wsi_patching.core.pipeline import Stage
from wsi_patching.core.types.types import CollatedPatchBatch
from wsi_patching.utils.audit import Knob


def _edge_distance(coords: np.ndarray, wsi_dims: tuple, tile_size: int) -> np.ndarray:
    """How many full tile rings a tile sits away from the nearest slide border.

    Defined so that this stage's keep condition is exactly `edge_distance >= depth`.
    Each of the four clauses of the keep condition is an integer upper bound on
    `depth`, so their conjunction is the minimum of the four bounds:

        x >= depth*ts        <=>  depth <= x // ts
        x <  W - depth*ts    <=>  depth <= ceil((W - x)/ts) - 1   (and same for y/H)
    """
    W, H = int(wsi_dims[0]), int(wsi_dims[1])
    x, y = coords[:, 0].astype(np.int64), coords[:, 1].astype(np.int64)
    ts = int(tile_size)
    return np.minimum.reduce(
        [x // ts, y // ts, -(-(W - x) // ts) - 1, -(-(H - y) // ts) - 1]  # -(-a//b) is ceil(a/b)
    )


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

    # `edge_distance` is pure geometry -- independent of `depth` -- so the audit
    # report can re-decide keep/drop for any depth without re-running.
    AUDIT_KNOBS = (Knob("depth", "edge_distance", ">=", permissive=0.0, integer=True),)

    def __init__(self, *, depth: int = 1):
        if depth < 0:
            raise ValueError("depth must be >= 0")
        self.depth = int(depth)

    def validate(self) -> None:
        self.ctx.require_key("tile_size")

    def __call__(self, it: Iterable[CollatedPatchBatch]) -> Iterable[CollatedPatchBatch]:
        tile_size = int(self.ctx["tile_size"])

        for batch in it:
            # Attach the metric for every row (the audit needs it even at depth=0).
            edge_distance = _edge_distance(batch.coords, batch.wsi_dims, tile_size)
            batch.add_meta_column("edge_distance", edge_distance)

            if self.depth == 0:
                yield batch
                continue

            keep_mask = edge_distance >= self.depth

            in_sz = len(batch.patches)
            batch.filter_on_mask(np.asarray(keep_mask, dtype=np.bool_))
            self.log.info(
                f"wsi={batch.wsi_id} batch_in={in_sz} batch_out={len(batch.patches)} "
                f"(depth={self.depth}, tile_size={tile_size})"
            )

            if len(batch.patches) == 0:
                continue

            yield batch
