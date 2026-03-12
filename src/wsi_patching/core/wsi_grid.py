from pathlib import Path
from typing import Any, Iterable, List, Literal

from wsi_patching.backends.cucim_openslide_isyntax import (
    get_dimensions_for_level,
    get_level_downsamples,
    get_level_for_resolution,
    get_resample_factor,
    validate_slide_backend,
)
from wsi_patching.backends.cupy_numpy import validate_xp_backend
from wsi_patching.backends.torch_device import get_torch_device
from wsi_patching.core.pipeline import PipelineContext, Stage
from wsi_patching.core.types.types import Slide


class WSIGrid(Stage):
    """
    Minimal source that yields one 'slide' sample per input slide.
    (MVP simplified: we do not enumerate *all tiles* here; Regionize will do per-ROI tiling.)
    """

    def __init__(
        self,
        slides: List[str],
        use_gpu: bool,
        resolution: float,
        unit: Literal["level", "mpp", "downsample"],
        fallback_mode: Literal["nearest", "floor", "ceil", "error", "resample"] = "error",
    ):
        """
        Initializes the WSIGrid stage, the starting point of a WSI patching pipeline.

        Args:
            slides: List of file paths to whole slide images (WSIs).
            use_gpu: Whether to use GPU-accelerated backends when possible (e.g., cuCIM, CuPy).
            resolution: Desired resolution for patch extraction.
            unit: Unit of the resolution ("level", "mpp", or "downsample").
            fallback_mode: Strategy for selecting resolution if exact match is unavailable (default is "error").
                Options are ("nearest", "floor", "ceil", "error", "resample").
                - "resample": read from the finest available level that is at least as sharp as requested,
                  then resample the region to the exact requested resolution. Tile coordinates and patch
                  dimensions are expressed in the virtual requested-resolution space.
        """
        self.slides = list(slides)
        self.use_gpu = use_gpu
        self.resolution = resolution
        self.unit = unit
        self.fallback_mode = fallback_mode

    def export_context(self, ctx: "PipelineContext") -> None:
        # Seed/override global grid parameters for other stages to read.
        ctx["resolution"] = self.resolution
        ctx["unit"] = self.unit
        ctx["fallback_mode"] = self.fallback_mode
        ctx["use_gpu"] = self.use_gpu

    def validate(self) -> None:
        validate_slide_backend(self.use_gpu)
        validate_xp_backend(self.use_gpu)
        get_torch_device(self.use_gpu)

    def for_slide(self, slide_path: str) -> "Stage":
        return WSIGrid(
            slides=[slide_path],
            use_gpu=self.use_gpu,
            resolution=self.resolution,
            unit=self.unit,
            fallback_mode=self.fallback_mode,
        )

    def __call__(self, it: Iterable[Any]) -> Iterable[Slide]:
        for path in self.slides:
            if not self.check_slide_exists(path):
                self.log.error(f"Slide path does not exist: {path}, skipping...")
                raise FileNotFoundError(f"Slide path does not exist: {path}")

            wsi_id = Path(path).stem
            selected_level = get_level_for_resolution(path, self.resolution, self.unit, self.fallback_mode)
            downsample = get_level_downsamples(path)[selected_level]

            if self.fallback_mode == "resample" and self.unit in ("mpp", "downsample"):
                resample_factor = get_resample_factor(path, selected_level, self.resolution, self.unit)
                # Compute virtual (requested-resolution) dimensions from the actual level's dimensions.
                actual_W, actual_H = get_dimensions_for_level(path, selected_level)
                W = round(actual_W / resample_factor)
                H = round(actual_H / resample_factor)
                # Virtual downsample: level-0 pixels per virtual pixel.
                virtual_downsample = downsample * resample_factor
            else:
                resample_factor = 1.0
                W, H = get_dimensions_for_level(path, selected_level)
                virtual_downsample = downsample

            self.log.info(f"Starting on Slide {wsi_id}")
            yield Slide(
                wsi_id=wsi_id,
                wsi_path=path,
                dims=(W, H),
                level=selected_level,
                downsample=virtual_downsample,
                resample_factor=resample_factor,
                meta={
                    "slide.wsi_id": wsi_id,
                    "slide.requested_resolution": self.resolution,
                    "slide.requested_unit": self.unit,
                    "slide.requested_fallback_mode": self.fallback_mode,
                    "slide.selected_level": selected_level,
                    "slide.resample_factor": resample_factor,
                    "slide.path": path,
                },
            )

    def check_slide_exists(self, path: str) -> bool:
        return Path(path).exists()
