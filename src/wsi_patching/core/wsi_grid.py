from pathlib import Path
from typing import Any, Iterable, List, Literal

from wsi_patching.backends.cucim_openslide_isyntax import (
    get_virtual_slide_dims,
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
    ):
        """
        Initializes the WSIGrid stage, the starting point of a WSI patching pipeline.

        Args:
            slides: List of file paths to whole slide images (WSIs).
            use_gpu: Whether to use GPU-accelerated backends when possible (e.g., cuCIM, CuPy).
            resolution: Desired resolution for patch extraction.
            unit: Unit of the resolution ("level", "mpp", or "downsample").

        Note:
            WSIGrid computes virtual (target-resolution) slide dimensions and exports
            ``resolution`` and ``unit`` to the pipeline context. The actual pyramid level
            used for reading is selected by ``RegionReadAndBatch`` (or ``PatchExtractor``)
            using its own ``fallback_mode`` parameter.
        """
        self.slides = list(slides)
        self.use_gpu = use_gpu
        self.resolution = resolution
        self.unit = unit

    def export_context(self, ctx: "PipelineContext") -> None:
        # Seed/override global grid parameters for other stages to read.
        ctx["resolution"] = self.resolution
        ctx["unit"] = self.unit
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
        )

    def __call__(self, it: Iterable[Any]) -> Iterable[Slide]:
        for path in self.slides:
            if not self.check_slide_exists(path):
                self.log.error(f"Slide path does not exist: {path}, skipping...")
                raise FileNotFoundError(f"Slide path does not exist: {path}")

            wsi_id = Path(path).stem
            virtual_W, virtual_H, virtual_ds = get_virtual_slide_dims(path, self.resolution, self.unit)

            self.log.info(f"Starting on Slide {wsi_id}")
            yield Slide(
                wsi_id=wsi_id,
                wsi_path=path,
                dims=(virtual_W, virtual_H),
                level=0,
                downsample=virtual_ds,
                resample_factor=1.0,
                meta={
                    "slide.wsi_id": wsi_id,
                    "slide.requested_resolution": self.resolution,
                    "slide.requested_unit": self.unit,
                    "slide.path": path,
                },
            )

    def check_slide_exists(self, path: str) -> bool:
        return Path(path).exists()
