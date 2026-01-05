import logging
from typing import Literal, Optional, Tuple, Union

import numpy as np
from isyntax import ISyntax
from openslide import OpenSlide

try:
    from cucim import CuImage
    from cucim.clara._cucim import CuImage as CuImageType

    _cucim_available = True
except ImportError:
    _cucim_available = False


logger = logging.getLogger(__name__)


def validate_slide_backend(use_gpu: bool) -> None:
    if use_gpu and not _cucim_available:
        raise ImportError("cuCIM is not available. Please install cuCIM to use GPU backend.")


def _open_slide(path: str, use_gpu: bool) -> Union["CuImage", "OpenSlide"]:
    if _cucim_available and use_gpu and path.lower().endswith((".tiff", ".tif", ".svs")):
        return CuImage(str(path))
    if path.lower().endswith(".isyntax"):
        return ISyntax.open(str(path))

    return OpenSlide(str(path))


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
    elif isinstance(img, ISyntax):
        # Use ISyntax
        region = img.read_region(x, y, w, h, level)[:, :, :3]  # discard alpha channel
    else:
        # Use OpenSlide
        region = img.read_region((x, y), level, (w, h)).convert("RGB")

    img.close()
    return np.asarray(region)


def get_dimensions_for_level(path: str, level: int) -> Tuple[int, int]:
    """
    Returns:
        W: int
        H: int
    """
    slide = _open_slide(path, use_gpu=False)
    try:
        assert 0 <= level < slide.level_count, f"Level {level} exceeds maximum level {slide.level_count - 1}."
        W, H = slide.level_dimensions[level]
        return int(W), int(H)
    finally:
        slide.close()


def get_level_for_resolution(
    path: str,
    resolution: float,
    unit: Literal["level", "mpp", "downsample"],
    fallback_mode: Literal["nearest", "floor", "ceil", "error"],
) -> int:
    """
    Determine which pyramid level to use for a given resolution specification.

    Args:
        path: Path to the WSI.
        resolution: Requested resolution value.
            - If unit == "level": pyramid level index (0, 1, 2, ...)
            - If unit == "mpp": microns per pixel.
            - If unit == "downsample": Downsample factor relative to level 0.
        unit: Resolution unit ("level", "mpp", or "downsample").
        fallback_mode:
            - "nearest": pick level whose value is closest to 'resolution'.
            - "floor":   pick coarsest level with value >= 'resolution'
                         (if none, use coarsest level).
            - "ceil":    pick finest level with value <= 'resolution'
                         (if none, use finest level, i.e. level 0).
            - "error":   require (almost) exact match, otherwise raise ValueError.

    Returns:
        level_idx: int
    """
    slide = _open_slide(path, use_gpu=False)

    try:
        # --- Trivial case: unit == "level" ---------------------------------
        if unit == "level":
            if not float(resolution).is_integer():
                raise ValueError("When unit is 'level', resolution must be an integer.")
            level_idx = int(resolution)
            if level_idx < 0:
                raise ValueError("Level must be non-negative.")
            if level_idx >= slide.level_count:
                raise ValueError(f"Requested level {level_idx} exceeds maximum level {slide.level_count - 1}.")
            return level_idx

        # --- Compute per-level values for mpp / downsample --------------------
        level_downsamples = list(slide.level_downsamples)  # e.g. [1.0, 2.0, 4.0, ...]
        requested = float(resolution)

        if unit == "mpp":
            if isinstance(slide, ISyntax):
                mpp0 = slide.mpp_x
            else:
                mpp0 = slide.properties.get("openslide.mpp-x")
                if mpp0 is None:
                    raise ValueError("Slide does not expose 'openslide.mpp-x' mpp metadata.")
            mpp0 = float(mpp0)
            # Effective mpp per level
            values = [mpp0 * float(ds) for ds in level_downsamples]

        elif unit == "downsample":
            # Downsample factor vs level 0
            values = [float(ds) for ds in level_downsamples]

        else:
            raise ValueError(f"Unknown unit: {unit}")

        # --- Helper: nearest index -----------------------------------------
        def nearest_idx() -> int:
            return min(range(len(values)), key=lambda i: abs(values[i] - requested))

        # --- Select level according to fallback_mode ------------------------
        if fallback_mode == "nearest":
            level_idx = nearest_idx()

        elif fallback_mode == "floor":
            # coarsest level with value >= requested (larger value = coarser)
            eligible = [i for i, v in enumerate(values) if v >= requested]
            if eligible:
                level_idx = min(eligible, key=lambda i: values[i])
            else:
                # if none >= requested, use coarsest (last) level
                level_idx = len(values) - 1

        elif fallback_mode == "ceil":
            # finest level with value <= requested
            eligible = [i for i, v in enumerate(values) if v <= requested]
            if eligible:
                level_idx = max(eligible, key=lambda i: values[i])
            else:
                # if none <= requested, use finest (level 0)
                level_idx = 0

        elif fallback_mode == "error":
            tol = 1e-6
            matches = [i for i, v in enumerate(values) if abs(v - requested) <= tol * max(1.0, abs(requested))]
            if not matches:
                raise ValueError(f"No exact {unit} match for requested {requested}; available values: {values}")
            level_idx = matches[0]

        else:
            raise ValueError(f"Unknown fallback_mode: {fallback_mode}")

        return level_idx

    finally:
        slide.close()
