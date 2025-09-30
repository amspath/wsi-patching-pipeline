from typing import Iterable, List

from wsi_patching.core.pipeline import Stage
from wsi_patching.regions_of_interest.roi_providers import ROIProvider, WholeSlideProvider
from wsi_patching.utils.types import Slide, SlideWithROIs


class AttachROIs(Stage):
    """Add Regions of Interest (ROIs) to each slide using one or more ROI providers."""

    def __init__(self, providers: List[ROIProvider], preclip_to_slide: bool = True):
        self.providers = list(providers)
        self.preclip = bool(preclip_to_slide)

    def __call__(self, it: Iterable[Slide]) -> Iterable[SlideWithROIs]:
        for s in it:
            all_rois = []
            for prov in self.providers:
                try:
                    slide_rois = prov.for_slide(s)
                    all_rois.extend(slide_rois)
                    self.log.info(f"{type(prov).__name__} found {len(slide_rois)} ROIs for slide {s.wsi_id}")
                except Exception as e:
                    self.log.info(f"{type(prov).__name__} failed: {e}")

            if not all_rois:
                all_rois = WholeSlideProvider().for_slide(s)
                self.log.warning(f"No ROIs found for slide {s.wsi_id}; using whole slide.")

            yield SlideWithROIs(**s.__dict__, rois=all_rois)
