import logging
from typing import Optional, Tuple, Union

try:
    from cucim import CuImage
    from cucim.clara._cucim import CuImage as CuImageType

    _cucim_available = True
except ImportError:
    _cucim_available = False

import numpy as np
from openslide import OpenSlide

logger = logging.getLogger(__name__)
_raised_warning = False


def validate_slide_backend(use_gpu: bool) -> None:
    if use_gpu and not _cucim_available:
        raise ImportError("cuCIM is not available. Please install cuCIM to use GPU backend.")


def _open_slide(path: str, use_gpu: bool) -> Union["CuImage", "OpenSlide"]:
    if _cucim_available and use_gpu:
        if path.lower().endswith((".tiff", ".tif", ".svs")):
            return CuImage(str(path))
        else:
            global _raised_warning
            if not _raised_warning:
                logger.warning(
                    f"cuCIM only supports tiff and svs, and does not support the file format '{path.split('.')[-1]}'. "
                    "Falling back to OpenSlide."
                )
                _raised_warning = True
    return OpenSlide(str(path))


def get_dimensions_for_level(path: str, level: int, use_gpu: bool) -> Tuple[int, int]:
    slide = _open_slide(path, use_gpu)
    if _cucim_available and isinstance(slide, CuImageType):
        W, H = slide.resolutions["level_dimensions"][level]
    else:
        W, H = slide.level_dimensions[level]

    slide.close()
    return int(W), int(H)


def read_region(
    path: str, x: int, y: int, w: int, h: int, level: int, use_gpu: bool, num_workers_cucim: int = 8
) -> Optional[np.ndarray]:
    """
    Return requested region at the given level.
    """
    img = _open_slide(path, use_gpu)
    if _cucim_available and isinstance(img, CuImageType):
        # Use cuCIM
        region = img.read_region(location=(x, y), size=(w, h), level=level, num_workers=num_workers_cucim)
    else:
        # Use OpenSlide
        region = img.read_region((x, y), level, (w, h)).convert("RGB")

    img.close()
    return np.asarray(region)
