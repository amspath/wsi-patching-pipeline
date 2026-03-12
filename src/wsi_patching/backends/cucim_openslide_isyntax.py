import logging
from typing import List, Literal, Optional, Tuple

import fastslide
import numpy as np

logger = logging.getLogger(__name__)


def validate_slide_backend(use_gpu: bool) -> None:
    """Validate that the slide reading backend is available.

    With fastslide, slide reading is always performed on the CPU regardless of
    the ``use_gpu`` flag.  The ``use_gpu`` flag only controls CuPy array usage
    for patch processing, which is validated separately by ``validate_xp_backend``.
    """
    pass  # fastslide is a required dependency; always available


def _open_slide(path: str) -> fastslide.FastSlide:
    return fastslide.FastSlide.from_file_path(str(path))


def read_region(
    path: str,
    x: int,
    y: int,
    w: int,
    h: int,
    level: int,
    use_gpu: bool = False,
    num_workers_cucim: int = 8,
) -> Optional[np.ndarray]:
    """
    Return the requested region at the given pyramid level as a NumPy array.

    Args:
        path: Path to the WSI file.
        x: Left edge of the region in level-0 coordinates.
        y: Top edge of the region in level-0 coordinates.
        w: Width of the region in pixels at the requested level.
        h: Height of the region in pixels at the requested level.
        level: Pyramid level to read from.
        use_gpu: Ignored. Kept for API compatibility; slide reading is always
            performed on the CPU via fastslide.
        num_workers_cucim: Ignored. Kept for API compatibility.

    Returns:
        NumPy array of shape (h, w, 3), dtype uint8.
    """
    with _open_slide(path) as slide:
        # fastslide uses level-native coordinates; convert from level-0 coords.
        x_native, y_native = slide.convert_level0_to_level_native(x, y, level)
        region = slide.read_region(location=(x_native, y_native), level=level, size=(w, h))
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


def get_level_for_resolution(
    path: str,
    resolution: float,
    unit: Literal["level", "mpp", "downsample"],
    fallback_mode: Literal["nearest", "floor", "ceil", "error", "resample"],
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
            - "resample": pick the finest level whose resolution is at least as
                          sharp as 'resolution' (same as "ceil"), and resample the
                          read region to match the requested resolution exactly.
                          Use get_resample_factor() to obtain the scaling factor.

    Returns:
        level_idx: int
    """
    with _open_slide(path) as slide:
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
            mpp = slide.mpp
            if mpp is None or mpp[0] is None:
                raise ValueError("Slide does not expose mpp metadata.")
            mpp0 = float(mpp[0])
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

        elif fallback_mode == "error":
            tol = 1e-6
            matches = [i for i, v in enumerate(values) if abs(v - requested) <= tol * max(1.0, abs(requested))]
            if not matches:
                raise ValueError(f"No exact {unit} match for requested {requested}; available values: {values}")
            level_idx = matches[0]

        elif fallback_mode == "resample":
            # Use the finest available level that is at least as sharp as requested
            # (same level selection as "ceil"), then resample to hit the exact resolution.
            eligible = [i for i, v in enumerate(values) if v <= requested]
            if eligible:
                level_idx = max(eligible, key=lambda i: values[i])
            else:
                # All available levels are coarser than requested; use the finest.
                level_idx = 0

        else:
            raise ValueError(f"Unknown fallback_mode: {fallback_mode}")

        return level_idx


def get_resample_factor(
    path: str,
    level: int,
    resolution: float,
    unit: Literal["mpp", "downsample"],
) -> float:
    """
    Compute the resampling factor needed to convert a region read at *level* to
    the exact requested resolution.

    A factor > 1.0 means the selected level is sharper (finer) than the requested
    resolution, so the read region will be larger and must be downsampled.
    A factor of 1.0 means no resampling is required (exact match).

    Args:
        path: Path to the WSI.
        level: The pyramid level at which the region will be read (the "read level",
               as returned by get_level_for_resolution with fallback_mode="resample").
        resolution: Requested resolution value (in the same unit).
        unit: Resolution unit ("mpp" or "downsample").

    Returns:
        resample_factor: float >= 1.0
    """
    with _open_slide(path) as slide:
        level_downsamples = [float(ds) for ds in slide.level_downsamples]
        actual_downsample = level_downsamples[level]

        if unit == "mpp":
            mpp = slide.mpp
            if mpp is None or mpp[0] is None:
                raise ValueError(f"Slide '{path}' does not expose mpp metadata.")
            mpp0 = float(mpp[0])
            actual_value = mpp0 * actual_downsample
            requested_value = float(resolution)
        elif unit == "downsample":
            actual_value = actual_downsample
            requested_value = float(resolution)
        else:
            raise ValueError(f"get_resample_factor does not support unit='{unit}'. Use 'mpp' or 'downsample'.")

        if actual_value <= 0:
            raise ValueError(f"Actual resolution value at level {level} is non-positive: {actual_value}")

        # factor = requested / actual: >1 means read level is finer (more pixels) than requested.
        # We clamp to 1.0 to avoid accidental upsampling when values are nearly equal.
        factor = requested_value / actual_value
        return max(1.0, factor)
