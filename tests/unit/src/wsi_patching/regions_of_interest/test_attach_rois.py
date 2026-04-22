from dataclasses import dataclass

import pytest

from wsi_patching.core.types.types import Slide
from wsi_patching.regions_of_interest.attach_rois import AttachROIs
from wsi_patching.regions_of_interest.roi_providers import RectROIProvider, ROIProvider
from wsi_patching.regions_of_interest.rois import BoxROI


@dataclass(frozen=True)
class SlideStub(Slide):
    pass


def test_attach_rois_combines_providers_and_defaults_with_warning(caplog):
    slideA = SlideStub("A", "/a", (100, 100), 0)
    slideB = SlideStub("B", "/b", (50, 40), 0)

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
    assert "No valid ROIs found for slide B; using whole slide." in caplog.text


def test_attach_rois_out_of_bounds_roi_raises_actual_provider_exception(caplog):
    slide = SlideStub("A", "/a", (100, 100), 0)

    prov = RectROIProvider(rois={"A": [(90, 90, 20, 20)]})  # out of bounds
    attach = AttachROIs([prov], on_empty="error")

    caplog.set_level("INFO")

    # Now we expect the original provider error, not "No valid ROIs found..."
    with pytest.raises(ValueError, match=r"lies outside"):
        list(attach(iter([slide])))

    assert "RectROIProvider failed:" in caplog.text
    assert "lies outside" in caplog.text


def test_attach_rois_multiple_provider_failures_aggregates_and_sets_cause(caplog):
    slide = SlideStub("A", "/a", (100, 100), 0)

    class BoomProv(ROIProvider):
        def for_slide(self, slide):
            raise RuntimeError("bang")

    oob = RectROIProvider(rois={"A": [(90, 90, 20, 20)]})
    attach = AttachROIs([BoomProv(), oob], on_empty="error")

    caplog.set_level("INFO")

    with pytest.raises(ValueError) as excinfo:
        list(attach(iter([slide])))

    # Aggregated message contains both failures
    msg = str(excinfo.value)
    assert "No valid ROIs found for slide A" in msg
    assert "RuntimeError: bang" in msg
    assert "ValueError:" in msg and "lies outside" in msg

    # Cause should be set (from provider_errors[0])
    assert excinfo.value.__cause__ is not None

    assert "failed:" in caplog.text
