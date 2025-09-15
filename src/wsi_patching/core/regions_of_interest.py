import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from wsi_patching.core.pipeline import Sample, Stage

Box = Tuple[int, int, int, int]  # (x, y, w, h) in level-0 pixels


class ROI:
    """Geometry-agnostic region of interest in level-0 coordinates."""

    def bounds(self) -> Box:
        raise NotImplementedError

    def contains_point(self, x: float, y: float) -> bool:
        """Return True if the (x,y) center lies in ROI. Used by center-in-ROI selection."""
        raise NotImplementedError


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
    """Source of ROIs for a slide."""

    def for_slide(self, slide: Sample) -> List[ROI]:
        raise NotImplementedError


@dataclass
class RectROIProvider(ROIProvider):
    """Compatibility provider using a dict: {wsi_id: [(x,y,w,h), ...]}.
    Raises ValueError if any ROI lies outside the slide bounds."""

    rois: Dict[str, List[Tuple[int, int, int, int]]]

    def for_slide(self, slide: Sample) -> List[ROI]:
        wsi_id = slide["wsi_id"]
        W, H = slide["dims"]
        out: List[ROI] = []
        for tpl in self.rois.get(wsi_id, []):
            x, y, w, h = tpl
            if x < 0 or y < 0 or (x + w) > W or (y + h) > H:
                raise ValueError(f"ROI {tpl} for slide {wsi_id} lies outside slide dimensions {(W, H)}")
            out.append(BoxROI(x, y, w, h))
        return out


class WholeSlideProvider(ROIProvider):
    """Provides a single ROI covering the full slide extent."""

    def for_slide(self, slide: Sample) -> List[ROI]:
        W, H = slide["dims"]
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

    def __init__(self, providers: List[ROIProvider], default_whole_slide: bool = True, preclip_to_slide: bool = True):
        self.providers = list(providers)
        self.default_whole_slide = bool(default_whole_slide)
        self.preclip = bool(preclip_to_slide)

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        for s in it:
            if s.get("type") != "slide":
                continue
            all_rois: List[ROI] = []
            for prov in self.providers:
                try:
                    rois = prov.for_slide(s)
                except Exception as e:
                    logging.info(f"[AttachROIs] provider {type(prov).__name__} failed: {e}")
                    rois = []
                all_rois.extend(rois)

            if not all_rois and self.default_whole_slide:
                all_rois.extend(WholeSlideProvider().for_slide(s))

            s2 = dict(s)
            s2["type"] = "roi_list"
            s2["rois"] = all_rois
            yield s2
