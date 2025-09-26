from dataclasses import dataclass
import logging
from typing import Dict, List, Tuple

import xml.etree.ElementTree as ET

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


@dataclass
class RectROIfromXMLProvider(ROIProvider):
    rois: Dict[str, str]

    def for_slide(self, slide: SlideBase) -> List[ROI]:
        out: List[ROI] = []

        xml_file = self.rois[slide.wsi_id]
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for ann in root.findall(".//Annotation[@PartOfGroup='roi']"):
            if ann.get("Type") != "Rectangle":
                continue

            coords = ann.find("Coordinates")
            if coords is None:
                continue

            xs = [float(coord.attrib["X"]) for coord in coords.findall("Coordinate")]
            ys = [float(coord.attrib["Y"]) for coord in coords.findall("Coordinate")]

            x_min, y_min = min(xs), min(ys)
            x_max, y_max = max(xs), max(ys)
            out.append(BoxROI(int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)))

        return out


class WholeSlideProvider(ROIProvider):
    def for_slide(self, slide: SlideBase) -> List[ROI]:
        W, H = slide.dims
        return [BoxROI(0, 0, int(W), int(H))]
