import logging
import threading
from contextlib import contextmanager
from typing import List, Literal, Tuple

import numpy as np
from fastslide import FastSlide

logger = logging.getLogger(__name__)

_thread_local = threading.local()


def _new_slide(path: str) -> FastSlide:
    return FastSlide.from_file_path(str(path))


@contextmanager
def _open_slide(path: str):
    slide = _new_slide(path)
    try:
        yield slide
    finally:
        if hasattr(slide, "close"):
            slide.close()


def _get_cached_slide(path: str):
    """Return a cached open FastSlide handle for the current thread."""
    cache = getattr(_thread_local, "slide_cache", None)
    if cache is None:
        _thread_local.slide_cache = {}
        cache = _thread_local.slide_cache
    if path not in cache:
        cache[path] = _new_slide(path)
    return cache[path]


def close_cached_slides() -> None:
    """Close and evict all cached slide handles for the current thread.

    Must be called by the producer thread after finishing a slide to prevent
    resource leaks. Safe to call even if no slides are cached.
    """
    cache = getattr(_thread_local, "slide_cache", {})
    for path, slide in list(cache.items()):
        try:
            if hasattr(slide, "close"):
                slide.close()
        except Exception:
            logger.warning("Failed to close cached slide for path %s", path)
    _thread_local.slide_cache = {}


def _get_mpp0(slide) -> float:
    mpp = getattr(slide, "mpp", None)
    if mpp is not None and mpp[0] is not None:
        return float(mpp[0])

    properties = getattr(slide, "properties", None)
    if properties is not None:
        mpp_x = properties.get("openslide.mpp-x")
        if mpp_x is not None:
            return float(mpp_x)

    raise ValueError("Could not retrieve mpp metadata from slide.mpp or openslide.mpp-x property.")


def read_region(path: str, x: int, y: int, w: int, h: int, level: int) -> np.ndarray:
    """
    Return the requested region at the given pyramid level as a NumPy array.

    Args:
        path: Path to the WSI file.
        x: Left edge of the region in level-0 coordinates.
        y: Top edge of the region in level-0 coordinates.
        w: Width of the region in pixels at the requested level.
        h: Height of the region in pixels at the requested level.
        level: Pyramid level to read from.

    Returns:
        NumPy array of shape (h, w, 3), dtype uint8.
    """
    slide = _get_cached_slide(path)
    x_native, y_native = slide.convert_level0_to_level_native(x, y, level)
    region = slide.read_region((x_native, y_native), level, (w, h))
    return np.asarray(region.numpy())


def get_dimensions_for_level(path: str, level: int) -> Tuple[int, int]:
    """
    Returns:
        W: int
        H: int
    """
    with _open_slide(path) as slide:
        assert 0 <= level < slide.level_count, f"Level {level} exceeds maximum level {slide.level_count - 1}."
        W, H = slide.level_dimensions[level]
        return int(W), int(H)


def get_level_downsamples(path: str) -> List[float]:
    """Return the list of downsample factors relative to level 0 for each pyramid level."""
    with _open_slide(path) as slide:
        return [float(ds) for ds in slide.level_downsamples]


def get_resample_factor(path: str, resolution: float, unit: Literal["mpp", "downsample"], selected_level: int) -> float:
    """Return resample_factor = requested_value / level_value.

    Factor > 1.0 means the selected level is finer than requested: read more
    level pixels, then scale down to the requested size.

    Args:
        path: Path to the WSI file.
        resolution: Requested resolution value (mpp or downsample).
        unit: "mpp" or "downsample".
        selected_level: The pyramid level that was selected for resampling.

    Returns:
        resample_factor (float > 0); equals 1.0 when level is an exact match.
    """
    with _open_slide(path) as slide:
        level_downsamples = [float(ds) for ds in slide.level_downsamples]
        level_ds = level_downsamples[selected_level]

        if unit == "mpp":
            mpp0 = _get_mpp0(slide)
            level_value = mpp0 * level_ds
        elif unit == "downsample":
            level_value = level_ds
        else:
            raise ValueError(f"Unknown unit: {unit}")

        return float(resolution) if level_value == 0 else float(resolution) / level_value


def get_level_for_resolution(
    path: str,
    resolution: float,
    unit: Literal["level", "mpp", "downsample"],
    fallback_mode: Literal["nearest", "floor", "ceil", "error", "resample"],
    max_relative_deviation: float | None = None,
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
            - "nearest":  pick level whose value is closest to 'resolution';
                          see 'max_relative_deviation'.
            - "floor":    pick coarsest level with value >= 'resolution'
                          (if none, use coarsest level).
            - "ceil":     pick finest level with value <= 'resolution'
                          (if none, use finest level, i.e. level 0).
            - "resample": same selection as "ceil"; caller must then resample
                          the read pixels to the exact requested resolution.
                          Raises ValueError when unit == "level".
        max_relative_deviation: Maximum allowed |level_value - resolution| / resolution
            for fallback_mode="nearest". Raises ValueError when the closest level lies
            outside that band. None disables the check. Ignored by the other modes.

    Returns:
        level_idx: int
    """
    with _open_slide(path) as slide:
        # --- Guard: resample is not meaningful for discrete level indices ---
        if fallback_mode == "resample" and unit == "level":
            raise ValueError("fallback_mode='resample' is not meaningful for unit='level'.")

        if max_relative_deviation is not None and max_relative_deviation < 0:
            raise ValueError(f"max_relative_deviation must be non-negative, got {max_relative_deviation}.")

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
        level_downsamples = [float(ds) for ds in slide.level_downsamples]  # e.g. [1.0, 2.0, 4.0, ...]
        requested = float(resolution)

        if unit == "mpp":
            mpp0 = _get_mpp0(slide)
            # Effective mpp per level
            values = [mpp0 * ds for ds in level_downsamples]

        elif unit == "downsample":
            # Downsample factor vs level 0
            values = level_downsamples

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

        elif fallback_mode == "resample":
            # Same selection as "ceil": finest level with value <= requested
            eligible = [i for i, v in enumerate(values) if v <= requested]
            if eligible:
                level_idx = max(eligible, key=lambda i: values[i])
            else:
                level_idx = 0

        elif fallback_mode == "error":
            tol = 1e-6
            matches = [i for i, v in enumerate(values) if abs(v - requested) <= tol * max(1.0, abs(requested))]
            if not matches:
                raise ValueError(f"No exact {unit} match for requested {requested}; available values: {values}")
            level_idx = matches[0]

        else:
            raise ValueError(f"Unknown fallback_mode: {fallback_mode}")

        if fallback_mode == "nearest" and max_relative_deviation is not None:
            deviation = abs(values[level_idx] - requested) / requested
            if deviation > max_relative_deviation:
                raise ValueError(
                    f"Nearest level {level_idx} has {unit} {values[level_idx]}, deviating "
                    f"{deviation:.2%} from the requested {requested} "
                    f"(max_relative_deviation={max_relative_deviation:.2%}). "
                    f"Available values: {values}. Widen max_relative_deviation, or pass None to disable."
                )

        return level_idx
