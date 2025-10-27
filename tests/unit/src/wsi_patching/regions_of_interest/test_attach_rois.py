from dataclasses import dataclass
from typing import Tuple

from wsi_patching.regions_of_interest.attach_rois import AttachROIs
from wsi_patching.regions_of_interest.roi_providers import ROIProvider
from wsi_patching.regions_of_interest.rois import BoxROI


@dataclass
class SlideStub:
    wsi_id: str
    wsi_path: str
    dims: Tuple[int, int]  # (W, H)
    meta: dict


def test_attach_rois_combines_providers_and_defaults_with_warning(caplog):
    slideA = SlideStub("A", "/a", (100, 100), {})
    slideB = SlideStub("B", "/b", (50, 40), {})

    class StaticProv(ROIProvider):
        def for_slide(self, slide):
            if slide.wsi_id == "A":
                return [BoxROI(0, 0, 10, 10), BoxROI(10, 10, 20, 20)]
            return []

    class FailingProv(ROIProvider):
        def for_slide(self, slide):
            raise RuntimeError("bang")

    attach = AttachROIs([FailingProv(), StaticProv()], on_empty="whole_slide")

    caplog.set_level("INFO")
    out = list(attach(iter([slideA, slideB])))

    # For A: two from StaticProv (FailingProv logs and is ignored)
    a = out[0]
    assert a.wsi_id == "A"
    assert hasattr(a, "rois")
    assert len(a.rois) == 2
    assert isinstance(a.rois[0], BoxROI)

    # For B: StaticProv returns none -> WholeSlideProvider fallback with warning
    b = out[1]
    assert b.wsi_id == "B"
    assert len(b.rois) == 1
    assert b.rois[0].bounds() == (0, 0, 50, 40)
    # log messages: provider failure + fallback warning
    assert "failed:" in caplog.text
    assert "No ROIs found for slide B; using whole slide." in caplog.text
