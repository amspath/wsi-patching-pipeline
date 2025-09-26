from dataclasses import dataclass
from typing import Dict, List, Tuple

from wsi_patching.regions_of_interest.rois import ROI, BoxROI
from wsi_patching.utils.types import SlideBase


class ROIProvider:
    def for_slide(self, slide: SlideBase) -> List[ROI]:
        raise NotImplementedError


@dataclass
class RectROIProvider(ROIProvider):
    rois: Dict[str, List[Tuple[int, int, int, int]]]

    def for_slide(self, slide: SlideBase) -> List[ROI]:
        wsi_id = slide.wsi_id
        W, H = slide.dims
        out: List[ROI] = []
        for x, y, w, h in self.rois.get(wsi_id, []):
            if x < 0 or y < 0 or (x + w) > W or (y + h) > H:
                raise ValueError(f"ROI {(x, y, w, h)} for slide {wsi_id} lies outside {(W, H)}")
            out.append(BoxROI(x, y, w, h))
        return out


class WholeSlideProvider(ROIProvider):
    def for_slide(self, slide: SlideBase) -> List[ROI]:
        W, H = slide.dims
        return [BoxROI(0, 0, int(W), int(H))]
