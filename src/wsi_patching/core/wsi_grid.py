import logging
from pathlib import Path
from typing import Any, Iterable, List, Tuple

from cucim import CuImage

from wsi_patching.core.pipeline import PipelineContext, Stage
from wsi_patching.utils.types import Slide


class WSIGrid(Stage):
    """
    Minimal source that yields one 'slide' sample per input slide.
    (MVP simplified: we do not enumerate *all tiles* here; Regionize will do per-ROI tiling.)
    """

    def __init__(self, slides: List[str], tile_size: int, stride: int, use_gpu: bool, level: int = 0):
        self.slides = list(slides)
        self.tile_size = tile_size
        self.stride = stride
        self.use_gpu = use_gpu
        self.level = level

    def export_context(self, ctx: "PipelineContext") -> None:
        # Seed/override global grid parameters for other stages to read.
        ctx["tile_size"] = self.tile_size
        ctx["stride"] = self.stride
        ctx["level"] = self.level
        ctx["use_gpu"] = self.use_gpu

    def for_slide(self, slide_path: str) -> "Stage":
        return WSIGrid(
            slides=[slide_path], tile_size=self.tile_size, stride=self.stride, use_gpu=self.use_gpu, level=self.level
        )

    def __call__(self, _it: Iterable[Any]) -> Iterable[Slide]:
        for path in self.slides:
            wsi_id = Path(path).stem
            W, H = self._get_level_dims(path, self.level)
            logging.info(f"Starting on Slide {wsi_id}")
            yield Slide(wsi_id=wsi_id, wsi_path=path, dims=(W, H), meta={"backend": "cucim"})

    def _get_level_dims(self, path: str, level: int) -> Tuple[int, int]:
        img = CuImage(path)
        W, H = img.resolutions["level_dimensions"][level]
        return int(W), int(H)
