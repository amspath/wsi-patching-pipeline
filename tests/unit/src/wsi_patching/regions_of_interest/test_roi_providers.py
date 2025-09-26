from dataclasses import dataclass
from typing import Tuple

import pytest

from wsi_patching.regions_of_interest.roi_providers import RectROIProvider, WholeSlideProvider
from wsi_patching.regions_of_interest.rois import BoxROI


@dataclass
class SlideStub:
    wsi_id: str
    wsi_path: str
    dims: Tuple[int, int]  # (W, H)
    meta: dict


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
