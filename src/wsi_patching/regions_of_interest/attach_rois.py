from typing import Iterable, List, Literal

from wsi_patching.core.pipeline import Stage
from wsi_patching.core.types.types import Slide, SlideWithROIs
from wsi_patching.regions_of_interest.roi_providers import ROIProvider, WholeSlideProvider


class AttachROIs(Stage):
    """Add Regions of Interest (ROIs) to each slide using one or more ROI providers."""

    def __init__(self, providers: List[ROIProvider], on_empty: Literal["error", "whole_slide"] = "error") -> None:
        """
        Args:
            providers: List of ROIProvider instances to use for obtaining ROIs.
            on_empty: Behavior when no ROIs are found for a slide. Default is "error".
                Options:
                - "error": raise an error.
                - "whole_slide": use the whole slide as a single ROI.
        """
        self.providers = list(providers)
        self.on_empty = on_empty

        if self.on_empty not in ("error", "whole_slide"):
            raise ValueError("on_empty must be 'error' or 'whole_slide'")

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
                if self.on_empty == "error":
                    raise ValueError(f"No ROIs found for slide {s.wsi_id}")

                all_rois = WholeSlideProvider().for_slide(s)
                self.log.warning(f"No ROIs found for slide {s.wsi_id}; using whole slide.")

            yield SlideWithROIs(**s.__dict__, rois=all_rois)
