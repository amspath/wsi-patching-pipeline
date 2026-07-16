from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from wsi_patching.backends.cupy_numpy import ensure_array_matches_use_gpu, is_cupy
from wsi_patching.core.types.util_types import MetaColStorage
from wsi_patching.utils.audit import get_current_audit

if TYPE_CHECKING:
    import cupy as cp
import numpy as np

# Bounding box: (x, y, w, h)
Box = Tuple[int, int, int, int]


@dataclass(frozen=True)
class SlideBase:
    wsi_id: str
    wsi_path: str
    dims: Tuple[int, int]
    level: int
    downsample: float = 1.0
    resample_factor: float = 1.0
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
    level: int
    roi_index: int
    roi_bounds: Box
    tiles: List[Tuple[int, int]]
    downsample: float = 1.0
    resample_factor: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegionTask:
    wsi_id: str
    wsi_path: str
    wsi_dims: Tuple[int, int]
    level: int
    region: Box
    tiles: List[Tuple[int, int]]
    downsample: float = 1.0
    resample_factor: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


class CollatedPatchBatch:
    wsi_id: str
    coords: np.ndarray  # (N, 2)
    patches: Union[np.ndarray, "cp.ndarray"]  # (N, ...)
    metadata: MetaColStorage
    wsi_dims: Tuple[int, int]  # (W, H) of the whole slide at patched resolution

    def __init__(
        self,
        wsi_id: str,
        coords: np.ndarray,
        patches: Union[np.ndarray, "cp.ndarray"],
        use_gpu: bool,
        wsi_dims: Tuple[int, int],
        metadata: Optional[MetaColStorage] = None,
    ):
        self.wsi_id = wsi_id
        self.coords = coords
        self.patches = patches
        self.wsi_dims = wsi_dims

        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = MetaColStorage(length=coords.shape[0])

        if use_gpu and not is_cupy(self.patches):
            raise TypeError("patches must be a cupy.ndarray when use_gpu is True")

        if self.patches.shape[0] != coords.shape[0]:
            raise ValueError("coords and patches must have same first dimension")

    def __len__(self) -> int:
        return self.coords.shape[0]

    def add_meta_column(self, name: str, values: Any) -> None:
        self.metadata.add_column(name, values)

    def drop_meta_column(self, name: str) -> None:
        self.metadata.drop_column(name)

    def filter_on_mask(self, bool_mask: np.ndarray) -> None:
        """In-place selection of rows using boolean mask.

        Drops data from all fields where mask is False.
        """
        n = self.coords.shape[0]
        if bool_mask.dtype != np.bool_:
            raise TypeError("Boolean mask must be of type np.bool_")
        if bool_mask.shape[0] != n:
            raise ValueError("Boolean mask length mismatch")

        # Audit hook: snapshot dropped rows' coords + metadata (the "why") before
        # they're discarded. No-op unless a run has auditing active.
        rec = get_current_audit()
        if rec is not None and not bool(bool_mask.all()):
            drop_mask = ~bool_mask
            rec.record_meta(self.coords[drop_mask], self.metadata.take(drop_mask).get_all_row_wise())

        self.coords = self.coords[bool_mask]
        self.patches = self.patches[ensure_array_matches_use_gpu(bool_mask, is_cupy(self.patches))]
        self.metadata = self.metadata.take(bool_mask)

    def get(self, idx: int):
        """
        Return a single sample:
        wsi_id: str
        coords: shape (2,)
        patch:  shape (...)
        meta:   dict[str, Any] with per-column row values
        """
        if not (0 <= idx < self.coords.shape[0]):
            raise IndexError("Index out of range")

        return self.wsi_id, self.coords[idx], self.patches[idx], self.metadata.get_sample(idx)


class EncodedCollatedPatchBatch(CollatedPatchBatch):
    encoding: str
    encoded_patches: Any

    @classmethod
    def from_collated_patch_batch(cls, batch: CollatedPatchBatch, encoding: str, encoded_patches: Any):
        obj = cls(
            wsi_id=batch.wsi_id,
            coords=batch.coords,
            patches=batch.patches,
            use_gpu=is_cupy(batch.patches),
            wsi_dims=batch.wsi_dims,
            metadata=batch.metadata,
        )
        obj.encoding = encoding
        obj.encoded_patches = encoded_patches
        return obj
