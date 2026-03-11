import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Tuple
from unittest.mock import patch

import pytest

from wsi_patching.regions_of_interest.roi_providers import RectROIProvider, RectROIfromXMLProvider, WholeSlideProvider
from wsi_patching.regions_of_interest.rois import BoxROI


@dataclass
class SlideStub:
    wsi_id: str
    wsi_path: str
    dims: Tuple[int, int]  # (W, H)
    meta: dict
    level: int = 0


def _write_xml(tmp_path, group: str, x1: float, y1: float, x2: float, y2: float) -> str:
    """Write a minimal Aperio-style XML with one Rectangle annotation."""
    xml = textwrap.dedent(f"""\
        <Annotations>
          <Annotation PartOfGroup="{group}" Type="Rectangle">
            <Coordinates>
              <Coordinate X="{x1}" Y="{y1}" />
              <Coordinate X="{x2}" Y="{y1}" />
              <Coordinate X="{x2}" Y="{y2}" />
              <Coordinate X="{x1}" Y="{y2}" />
            </Coordinates>
          </Annotation>
        </Annotations>
    """)
    p = tmp_path / "ann.xml"
    p.write_text(xml)
    return str(p)


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


# --- RectROIfromXMLProvider tests ---


def test_xml_provider_default_group(tmp_path):
    """Default annotation_group='roi', no scaling → correct BoxROI returned."""
    xml = _write_xml(tmp_path, "roi", 10, 20, 60, 70)
    slide = SlideStub("S", "/p", (200, 200), {}, level=0)
    prov = RectROIfromXMLProvider(rois={"S": xml})
    rois = prov.for_slide(slide)
    assert len(rois) == 1
    assert rois[0].bounds() == (10, 20, 50, 50)


def test_xml_provider_custom_group(tmp_path):
    """Only annotations matching annotation_group are collected."""
    xml = textwrap.dedent("""\
        <Annotations>
          <Annotation PartOfGroup="tissue" Type="Rectangle">
            <Coordinates>
              <Coordinate X="0" Y="0" />
              <Coordinate X="10" Y="0" />
              <Coordinate X="10" Y="10" />
              <Coordinate X="0" Y="10" />
            </Coordinates>
          </Annotation>
          <Annotation PartOfGroup="roi" Type="Rectangle">
            <Coordinates>
              <Coordinate X="5" Y="5" />
              <Coordinate X="15" Y="5" />
              <Coordinate X="15" Y="15" />
              <Coordinate X="5" Y="15" />
            </Coordinates>
          </Annotation>
        </Annotations>
    """)
    p = tmp_path / "ann.xml"
    p.write_text(xml)
    slide = SlideStub("S", "/p", (200, 200), {}, level=0)

    prov_tissue = RectROIfromXMLProvider(rois={"S": str(p)}, annotation_group="tissue")
    assert prov_tissue.for_slide(slide)[0].bounds() == (0, 0, 10, 10)

    prov_roi = RectROIfromXMLProvider(rois={"S": str(p)}, annotation_group="roi")
    assert prov_roi.for_slide(slide)[0].bounds() == (5, 5, 10, 10)


def test_xml_provider_same_level_no_backend_call(tmp_path):
    """annotation_level == slide.level → get_level_downsamples is NOT called."""
    xml = _write_xml(tmp_path, "roi", 0, 0, 50, 50)
    slide = SlideStub("S", "/p", (200, 200), {}, level=1)
    prov = RectROIfromXMLProvider(rois={"S": xml}, annotation_level=1)

    _backend = "wsi_patching.regions_of_interest.roi_providers.get_level_downsamples"
    with patch(_backend) as mock_ds:
        rois = prov.for_slide(slide)
        mock_ds.assert_not_called()

    assert rois[0].bounds() == (0, 0, 50, 50)


def test_xml_provider_annotation_level_scaling(tmp_path):
    """Annotations at level 0 scaled to level 1 (ds ratio 2.0/1.0 → scale 0.5)."""
    # Annotation drawn at level 0: (100, 200) → (300, 400) → w=200, h=200
    xml = _write_xml(tmp_path, "roi", 100, 200, 300, 400)
    # slide.level=1, dims at level 1 are smaller
    slide = SlideStub("S", "/p", (500, 500), {}, level=1)
    prov = RectROIfromXMLProvider(rois={"S": xml}, annotation_level=0)

    _backend = "wsi_patching.regions_of_interest.roi_providers.get_level_downsamples"
    with patch(_backend, return_value=[1.0, 2.0, 4.0]):
        rois = prov.for_slide(slide)

    # scale = ds[0] / ds[1] = 1.0 / 2.0 = 0.5
    assert rois[0].bounds() == (50, 100, 100, 100)


def test_xml_provider_out_of_bounds_after_scaling(tmp_path):
    """ValueError when scaled ROI exceeds slide dims."""
    # Annotation at level 0 covering (0,0)→(400,400); after scale=0.5 → (0,0,200,200)
    # But slide dims are only (100, 100) at level 1 → out of bounds
    xml = _write_xml(tmp_path, "roi", 0, 0, 400, 400)
    slide = SlideStub("S", "/p", (100, 100), {}, level=1)
    prov = RectROIfromXMLProvider(rois={"S": xml}, annotation_level=0)

    _backend = "wsi_patching.regions_of_interest.roi_providers.get_level_downsamples"
    with patch(_backend, return_value=[1.0, 2.0]):
        with pytest.raises(ValueError):
            prov.for_slide(slide)


def _write_polygon_bbox_xml(tmp_path, group: str, x1: float, y1: float, x2: float, y2: float, filename: str = "ann.xml") -> str:
    """Write a minimal XML with one Polygon annotation that forms a 4-point axis-aligned bounding box."""
    xml = textwrap.dedent(f"""\
        <Annotations>
          <Annotation PartOfGroup="{group}" Type="Polygon">
            <Coordinates>
              <Coordinate X="{x1}" Y="{y1}" />
              <Coordinate X="{x2}" Y="{y1}" />
              <Coordinate X="{x2}" Y="{y2}" />
              <Coordinate X="{x1}" Y="{y2}" />
            </Coordinates>
          </Annotation>
        </Annotations>
    """)
    p = tmp_path / filename
    p.write_text(xml)
    return str(p)


def test_xml_provider_polygon_valid_bbox(tmp_path):
    """4-point Polygon annotation that is a valid axis-aligned bounding box is accepted."""
    xml = _write_polygon_bbox_xml(tmp_path, "roi", 10, 20, 60, 70)
    slide = SlideStub("S", "/p", (200, 200), {}, level=0)
    prov = RectROIfromXMLProvider(rois={"S": xml})
    rois = prov.for_slide(slide)
    assert len(rois) == 1
    assert rois[0].bounds() == (10, 20, 50, 50)


def test_xml_provider_polygon_too_many_points(tmp_path):
    """Polygon with more than 4 points raises ValueError."""
    xml = textwrap.dedent("""\
        <Annotations>
          <Annotation PartOfGroup="roi" Type="Polygon">
            <Coordinates>
              <Coordinate X="0" Y="0" />
              <Coordinate X="10" Y="0" />
              <Coordinate X="15" Y="5" />
              <Coordinate X="10" Y="10" />
              <Coordinate X="0" Y="10" />
            </Coordinates>
          </Annotation>
        </Annotations>
    """)
    p = tmp_path / "ann.xml"
    p.write_text(xml)
    slide = SlideStub("S", "/p", (200, 200), {}, level=0)
    prov = RectROIfromXMLProvider(rois={"S": str(p)})
    with pytest.raises(ValueError, match="5 point"):
        prov.for_slide(slide)


def test_xml_provider_polygon_not_axis_aligned(tmp_path):
    """4-point Polygon that is not axis-aligned (not a bounding box) raises ValueError."""
    xml = textwrap.dedent("""\
        <Annotations>
          <Annotation PartOfGroup="roi" Type="Polygon">
            <Coordinates>
              <Coordinate X="0" Y="5" />
              <Coordinate X="5" Y="0" />
              <Coordinate X="10" Y="5" />
              <Coordinate X="5" Y="10" />
            </Coordinates>
          </Annotation>
        </Annotations>
    """)
    p = tmp_path / "ann.xml"
    p.write_text(xml)
    slide = SlideStub("S", "/p", (200, 200), {}, level=0)
    prov = RectROIfromXMLProvider(rois={"S": str(p)})
    with pytest.raises(ValueError, match="axis-aligned"):
        prov.for_slide(slide)


def test_xml_provider_polygon_mixed_with_rectangle(tmp_path):
    """Both Rectangle and Polygon annotations in the same file are processed."""
    xml = textwrap.dedent("""\
        <Annotations>
          <Annotation PartOfGroup="roi" Type="Rectangle">
            <Coordinates>
              <Coordinate X="0" Y="0" />
              <Coordinate X="10" Y="0" />
              <Coordinate X="10" Y="10" />
              <Coordinate X="0" Y="10" />
            </Coordinates>
          </Annotation>
          <Annotation PartOfGroup="roi" Type="Polygon">
            <Coordinates>
              <Coordinate X="20" Y="20" />
              <Coordinate X="40" Y="20" />
              <Coordinate X="40" Y="50" />
              <Coordinate X="20" Y="50" />
            </Coordinates>
          </Annotation>
        </Annotations>
    """)
    p = tmp_path / "ann.xml"
    p.write_text(xml)
    slide = SlideStub("S", "/p", (200, 200), {}, level=0)
    prov = RectROIfromXMLProvider(rois={"S": str(p)})
    rois = prov.for_slide(slide)
    assert len(rois) == 2
    assert rois[0].bounds() == (0, 0, 10, 10)
    assert rois[1].bounds() == (20, 20, 20, 30)


def test_xml_provider_polygon_other_type_skipped(tmp_path):
    """Annotation types other than Rectangle and Polygon are silently skipped."""
    xml = textwrap.dedent("""\
        <Annotations>
          <Annotation PartOfGroup="roi" Type="Spline">
            <Coordinates>
              <Coordinate X="0" Y="0" />
              <Coordinate X="10" Y="5" />
              <Coordinate X="20" Y="0" />
            </Coordinates>
          </Annotation>
          <Annotation PartOfGroup="roi" Type="Rectangle">
            <Coordinates>
              <Coordinate X="5" Y="5" />
              <Coordinate X="15" Y="5" />
              <Coordinate X="15" Y="15" />
              <Coordinate X="5" Y="15" />
            </Coordinates>
          </Annotation>
        </Annotations>
    """)
    p = tmp_path / "ann.xml"
    p.write_text(xml)
    slide = SlideStub("S", "/p", (200, 200), {}, level=0)
    prov = RectROIfromXMLProvider(rois={"S": str(p)})
    rois = prov.for_slide(slide)
    assert len(rois) == 1
    assert rois[0].bounds() == (5, 5, 10, 10)
