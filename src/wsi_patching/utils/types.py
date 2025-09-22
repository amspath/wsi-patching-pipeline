from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

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


@dataclass
class PatchSample:
    wsi_id: str
    coord: Tuple[int, int]
    patch: np.ndarray  # H x W x C uint8
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollatedPatchBatch:
    wsi_id: str
    coords: List[Tuple[int, int]]
    patches: Union[np.ndarray, "cp.ndarray"]  # np if use_gpu=False else cp.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Patch:
    key: str
    patch: object
    meta: dict


@dataclass(frozen=True)
class EndOfStream:
    pass


@dataclass(frozen=True)
class EndOfQueue:
    pass
