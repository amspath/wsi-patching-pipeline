from dataclasses import dataclass
from typing import Tuple

from wsi_patching.regions_of_interest.rois import BoxROI, RectAreaROI


# Minimal stand-in for Slide/SlideBase used by providers
@dataclass
class SlideStub:
    wsi_id: str
    wsi_path: str
    dims: Tuple[int, int]  # (W, H)
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
