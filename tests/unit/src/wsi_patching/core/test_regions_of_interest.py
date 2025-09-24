from dataclasses import dataclass

import pytest

from wsi_patching.core.regions_of_interest import (
    AttachROIs,
    BoxROI,
    RectAreaROI,
    RectROIProvider,
    ROIProvider,
    WholeSlideProvider,
)


# Minimal stand-in for Slide/SlideBase used by providers
@dataclass
class SlideStub:
    wsi_id: str
    wsi_path: str
    dims: tuple[int, int]  # (W, H)
    meta: dict


def test_boxroi_bounds_and_contains():
    r = BoxROI(10, 20, 30, 40)
    assert r.bounds() == (10, 20, 30, 40)
    # inside
    assert r.contains_point(10, 20) is True
    assert r.contains_point(39.9, 59.9) is True
    # edges: x+w and y+h are excluded
    assert r.contains_point(40, 60) is False
    # outside
    assert r.contains_point(9, 20) is False
    assert r.contains_point(10, 19) is False


def test_rect_roi_provider_ok_and_out_of_bounds():
    slide = SlideStub("S", "/p", (100, 80), {})
    prov = RectROIProvider(rois={"S": [(0, 0, 50, 50), (50, 30, 50, 50)]})
    rois = prov.for_slide(slide)
    assert len(rois) == 2
    assert isinstance(rois[0], BoxROI)
    assert rois[0].bounds() == (0, 0, 50, 50)

    # out of bounds: (x+w) beyond W, or (y+h) beyond H -> ValueError
    bad = RectROIProvider(rois={"S": [(60, 0, 50, 10)]})
    with pytest.raises(ValueError):
        bad.for_slide(slide)
    bad2 = RectROIProvider(rois={"S": [(0, 70, 10, 20)]})
    with pytest.raises(ValueError):
        bad2.for_slide(slide)


def test_whole_slide_provider():
    slide = SlideStub("S", "/p", (123, 77), {})
    rois = WholeSlideProvider().for_slide(slide)
    assert len(rois) == 1
    assert rois[0].bounds() == (0, 0, 123, 77)


def test_rect_area_roi_subdivide_aligned_and_skips_too_small():
    # Area 0..200 x 0..150; max_size=64 => grid chunks; stride=16; tile_size=32 (skip <32)
    area = RectAreaROI(0, 0, 200, 150)
    subs = area.subdivide(max_size=64, tile_size=32, stride=16)
    assert subs, "should produce some sub-ROIs"

    for r in subs:
        x, y, w, h = r.as_tuple()
        # each sub-ROI stays within the original
        assert 0 <= x < 200 and 0 <= y < 150
        assert x + w <= 200 and y + h <= 150
        # width/height aligned to stride and >= tile_size
        assert w % 16 == 0 and h % 16 == 0
        assert w >= 32 and h >= 32

    # Case where aligned size would be < tile_size -> result list empty
    tiny = RectAreaROI(0, 0, 20, 20)
    subs2 = tiny.subdivide(max_size=64, tile_size=32, stride=16)
    assert subs2 == []


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

    attach = AttachROIs([FailingProv(), StaticProv()], preclip_to_slide=True)

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
