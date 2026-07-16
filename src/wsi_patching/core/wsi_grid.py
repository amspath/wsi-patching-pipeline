from pathlib import Path
from typing import Any, Iterable, List, Literal, cast

from wsi_patching.backends.cupy_numpy import validate_xp_backend
from wsi_patching.backends.fastslide import (
    get_dimensions_for_level,
    get_level_downsamples,
    get_level_for_resolution,
    get_resample_factor,
)
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
        resolution: float,
        unit: Literal["level", "mpp", "downsample"],
        use_gpu: bool = False,
        fallback_mode: Literal["nearest", "floor", "ceil", "error", "resample"] = "error",
        resample_interpolation: Literal["nearest", "linear", "cubic", "area", "lanczos"] = "lanczos",
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
                - "resample": selects the finest level with resolution <= requested, reads extra pixels,
                  then downsamples to the exact target resolution using cv2.resize. Not valid for unit="level".
            resample_interpolation: Interpolation method used when fallback_mode="resample".
                Options are ("nearest", "linear", "cubic", "area", "lanczos"). Default is "lanczos".
        """
        self.slides = list(slides)
        self.use_gpu = use_gpu
        self.resolution = resolution
        self.unit = unit
        self.fallback_mode = fallback_mode
        self.resample_interpolation = resample_interpolation

    def export_context(self, ctx: "PipelineContext") -> None:
        # Seed/override global grid parameters for other stages to read.
        ctx["resolution"] = self.resolution
        ctx["unit"] = self.unit
        ctx["fallback_mode"] = self.fallback_mode
        ctx["use_gpu"] = self.use_gpu
        ctx["resample_interpolation"] = self.resample_interpolation

    def validate(self) -> None:
        validate_xp_backend(self.use_gpu)

    def for_slide(self, slide_path: str) -> "Stage":
        return WSIGrid(
            slides=[slide_path],
            use_gpu=self.use_gpu,
            resolution=self.resolution,
            unit=self.unit,
            fallback_mode=self.fallback_mode,
            resample_interpolation=self.resample_interpolation,
        )

    def __call__(self, it: Iterable[Any]) -> Iterable[Slide]:
        for path in self.slides:
            if not self.check_slide_exists(path):
                self.log.error(f"Slide path does not exist: {path}, skipping...")
                raise FileNotFoundError(f"Slide path does not exist: {path}")

            wsi_id = Path(path).stem
            selected_level = get_level_for_resolution(path, self.resolution, self.unit, self.fallback_mode)
            W, H = get_dimensions_for_level(path, selected_level)
            level_downsamples = get_level_downsamples(path)
            downsample = level_downsamples[selected_level]

            if self.fallback_mode == "resample":
                rf = get_resample_factor(
                    path, self.resolution, cast(Literal["mpp", "downsample"], self.unit), selected_level
                )
                W_virtual = round(W / rf)
                H_virtual = round(H / rf)
                virtual_ds = downsample * rf
                slide_dims = (W_virtual, H_virtual)
                slide_downsample = virtual_ds
                resample_factor = rf
            else:
                slide_dims = (W, H)
                slide_downsample = downsample
                resample_factor = 1.0

            self.log.info(f"Starting on Slide {wsi_id}")
            yield Slide(
                wsi_id=wsi_id,
                wsi_path=path,
                dims=slide_dims,
                level=selected_level,
                downsample=slide_downsample,
                resample_factor=resample_factor,
                meta={
                    "slide.wsi_id": wsi_id,
                    "slide.requested_resolution": self.resolution,
                    "slide.requested_unit": self.unit,
                    "slide.requested_fallback_mode": self.fallback_mode,
                    "slide.selected_level": selected_level,
                    "slide.downsample": slide_downsample,
                    "slide.path": path,
                },
            )

    def check_slide_exists(self, path: str) -> bool:
        return Path(path).exists()
