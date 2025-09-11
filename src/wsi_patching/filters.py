from typing import Dict, Iterable, List, Optional

from wsi_patching.core import Stage
from wsi_patching.typing import Rect, Sample


class FilterByROI(Stage):
    """
    Attach ROI rectangles (level coords) per WSI.
    roi_by_wsi: { wsi_id: [(x,y,w,h), ...], ... }
    If no ROI provided, leave it empty -> Regionize will chunk full slide.
    """

    placement = "producer"

    def __init__(self, roi_by_wsi: Optional[Dict[str, List[Rect]]] = None):
        self.roi_by_wsi = roi_by_wsi or {}

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        for s in it:
            rois = self.roi_by_wsi.get(s["wsi_id"], [])
            s = dict(s)
            s["roi_rects"] = list(rois)
            yield s
