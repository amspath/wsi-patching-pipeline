import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from wsi_patching.core.pipeline import Stage
from wsi_patching.utils.types import Slide, SlideBase, SlideWithROIs

Box = Tuple[int, int, int, int]


class ROI:
    def bounds(self) -> Box: ...
    def contains_point(self, x: float, y: float) -> bool: ...


@dataclass
class BoxROI(ROI):
    x: int
    y: int
    w: int
    h: int

    def bounds(self) -> Box:
        return (self.x, self.y, self.w, self.h)

    def contains_point(self, x: float, y: float) -> bool:
        return (self.x <= x < self.x + self.w) and (self.y <= y < self.y + self.h)


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


@dataclass
class RectAreaROI:
    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def subdivide(self, max_size: int, tile_size: int, stride: int) -> List["RectAreaROI"]:
        """
        Split this rectangle into smaller rectangles if width or height > max_size.
        Splits are aligned to stride so that tiles stay consistent.
        """
        sub_rois: List[RectAreaROI] = []
        x_end = self.x + self.w
        y_end = self.y + self.h

        for yy in range(self.y, y_end, max_size):
            for xx in range(self.x, x_end, max_size):
                ww = min(max_size, x_end - xx)
                hh = min(max_size, y_end - yy)

                # Align to stride boundaries: ensure splits produce valid tile starts
                # (optional: round ww/hh up so they cover full tiles)
                aligned_w = (ww // stride) * stride
                aligned_h = (hh // stride) * stride
                if aligned_w < tile_size or aligned_h < tile_size:
                    continue

                sub_rois.append(RectAreaROI(xx, yy, aligned_w, aligned_h))

        return sub_rois


class AttachROIs(Stage):
    """Attach a list[ROI] to each slide using one or more providers."""

    def __init__(self, providers: List[ROIProvider], preclip_to_slide: bool = True):
        self.providers = list(providers)
        self.preclip = bool(preclip_to_slide)

    def __call__(self, it: Iterable[Slide]) -> Iterable[SlideWithROIs]:
        for s in it:
            all_rois = []
            for prov in self.providers:
                try:
                    all_rois.extend(prov.for_slide(s))
                except Exception as e:
                    logging.info(f"[AttachROIs] {type(prov).__name__} failed: {e}")

            if not all_rois:
                all_rois = WholeSlideProvider().for_slide(s)
                logging.warning(f"No ROIs found for slide {s.wsi_id}; using whole slide.")

            yield SlideWithROIs(**s.__dict__, rois=all_rois)
