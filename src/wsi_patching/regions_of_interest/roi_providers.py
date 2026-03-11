import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from wsi_patching.backends.cucim_openslide_isyntax import get_level_downsamples
from wsi_patching.core.types.types import SlideBase
from wsi_patching.regions_of_interest.rois import ROI, BoxROI


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


@dataclass
class RectROIfromXMLProvider(ROIProvider):
    rois: Dict[str, str]
    annotation_group: str = "roi"
    annotation_level: Optional[int] = None

    def for_slide(self, slide: SlideBase) -> List[ROI]:
        out: List[ROI] = []

        xml_file = self.rois[slide.wsi_id]
        tree = ET.parse(xml_file)
        root = tree.getroot()
        W, H = slide.dims

        if self.annotation_level is not None and self.annotation_level != slide.level:
            downsamples = get_level_downsamples(slide.wsi_path)
            scale = downsamples[self.annotation_level] / downsamples[slide.level]
        else:
            scale = 1.0

        for ann in root.findall(f".//Annotation[@PartOfGroup='{self.annotation_group}']"):
            if ann.get("Type") != "Rectangle":
                continue

            coords = ann.find("Coordinates")
            if coords is None:
                continue

            xs = [float(coord.attrib["X"]) * scale for coord in coords.findall("Coordinate")]
            ys = [float(coord.attrib["Y"]) * scale for coord in coords.findall("Coordinate")]

            x_min, y_min = min(xs), min(ys)
            x_max, y_max = max(xs), max(ys)
            if x_min < 0 or y_min < 0 or x_max > W or y_max > H:
                raise ValueError(
                    f"ROI {(x_min, y_min, x_max - x_min, y_max - y_min)} for slide {slide.wsi_id} lies outside {(W, H)}"
                )
            out.append(BoxROI(int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)))

        return out


class WholeSlideProvider(ROIProvider):
    def for_slide(self, slide: SlideBase) -> List[ROI]:
        W, H = slide.dims
        return [BoxROI(0, 0, int(W), int(H))]
