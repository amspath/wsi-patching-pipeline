from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

from wsi_patching.backends.cupy_numpy import ensure_array_matches_use_gpu, get_xp_backend

if TYPE_CHECKING:
    import cupy as cp
import numpy as np

Box = Tuple[int, int, int, int]


@dataclass(frozen=True)
class SlideBase:
    wsi_id: str
    wsi_path: str
    dims: Tuple[int, int]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Slide(SlideBase):
    pass


@dataclass(frozen=True)
class SlideWithROIs(SlideBase):
    rois: List[Any] = field(default_factory=list)  # ROI instances


@dataclass(frozen=True)
class TilePlan:
    wsi_id: str
    wsi_path: str
    dims: Tuple[int, int]
    roi_index: int
    roi_bounds: Box
    tiles: List[Tuple[int, int]]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegionTask:
    wsi_id: str
    wsi_path: str
    region: Box
    tiles: List[Tuple[int, int]]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Patch:
    key: str
    patch: object
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EndOfStream:
    pass


@dataclass(frozen=True)
class EndOfQueue:
    pass


@dataclass
class CollatedPatchBatch:
    """A batch of patches from a single WSI, with coordinates and optional metadata.

    Optimized for fast addition of columns in the meta data and in-place filtering.

    Example usage:
    ```
    batch = CollatedPatchBatch(wsi_id, coords_np, patches_np, {"area": areas_np})
    batch.add_col("probs", probs_np)                       # (N, C)
    mask = (batch.meta_cols["area"] > 2500)                # boolean np.ndarray
    batch.filter(mask)                                     # in-place
    wsi_id, coord, patch, meta = batch.get(0)              # get first sample
    ```
    """

    wsi_id: str
    coords: np.ndarray  # (N, 2)
    patches: Union[np.ndarray, "cp.ndarray"]  # (N, ...)

    # Meta columns, each entry in the dict is an array of length N
    meta_cols: Dict[str, np.ndarray]  # each first dim == N

    def add_col(self, name: str, values: np.ndarray) -> None:
        if values.shape[0] != self.coords.shape[0]:
            raise ValueError("add_col: first dimension must equal number of rows")
        self.meta_cols[name] = values

    def filter(self, mask: np.ndarray, use_gpu: bool) -> None:
        """
        In-place compaction. `mask` is a boolean array (same backend) of length N.
        """
        xp = get_xp_backend(use_gpu=use_gpu)

        if mask.dtype != xp.bool_:
            raise TypeError("filter: mask must be boolean (np.bool_ / cp.bool_)")
        if mask.shape[0] != self.coords.shape[0]:
            raise ValueError("filter: mask length mismatch")

        # Vectorized compaction (one pass per column)
        self.coords = np.compress(mask, self.coords, axis=0)
        self.patches = xp.compress(ensure_array_matches_use_gpu(mask, use_gpu), self.patches, axis=0)
        for k, v in self.meta_cols.items():
            self.meta_cols[k] = np.compress(mask, v, axis=0)

    def get(self, i: int):
        """
        Return a single sample:
        coords: shape (2,)
        patch:  shape (...)
        meta:   dict[str, Any] with per-column row values
        """
        n = self.coords.shape[0]
        if i < -n or i >= n:
            raise IndexError(f"index {i} out of range for N={n}")
        coord = self.coords[i]
        patch = self.patches[i]
        meta = {k: _to_jsonable(v[i]) for k, v in self.meta_cols.items()}
        return self.wsi_id, coord, patch, meta


def _to_jsonable(x: Any):
    # Fast path for NumPy stuff
    if isinstance(x, np.generic):
        return x.item()
    # numpy arrays
    if isinstance(x, np.ndarray):
        return x.tolist()
    # Already JSON-friendly? return as-is
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    # Fallback: stringify (or raise if you prefer)
    return str(x)
