from __future__ import annotations

import textwrap

import numpy as np
import pytest
import tifffile


@pytest.fixture(scope="session")
def synthetic_slide(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("slides")
    path = tmp / "test_slide.tif"
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[:256, :256] = [200, 50, 50]  # red top-left
    img[:256, 256:] = [50, 200, 50]  # green top-right
    img[256:, :256] = [50, 50, 200]  # blue bottom-left
    img[256:, 256:] = [200, 200, 50]  # yellow bottom-right
    tifffile.imwrite(str(path), img, photometric="rgb", tile=(256, 256), compression="deflate")
    return str(path), "test_slide"


@pytest.fixture(scope="session")
def synthetic_xml_annotation(tmp_path_factory, synthetic_slide):
    _, wsi_id = synthetic_slide
    tmp = tmp_path_factory.mktemp("annotations")
    xml_path = tmp / f"{wsi_id}.xml"
    xml_content = textwrap.dedent("""\
        <ASAP_Annotations>
          <Annotations>
            <Annotation PartOfGroup="roi" Type="Rectangle">
              <Coordinates>
                <Coordinate X="0" Y="0" />
                <Coordinate X="256" Y="256" />
              </Coordinates>
            </Annotation>
          </Annotations>
        </ASAP_Annotations>
    """)
    xml_path.write_text(xml_content)
    return str(xml_path), wsi_id


@pytest.fixture(scope="module")
def pyramid_slide(tmp_path_factory):
    """Two-level pyramid TIFF.

    Level 0: 512×512 (downsample=1.0) — four colour quadrants.
    Level 1: 256×256 (downsample=2.0) — same quadrants, half resolution.

    No MPP metadata, so tests use unit="downsample".
    """
    tmp = tmp_path_factory.mktemp("pyramid_slides")
    path = tmp / "pyramid.tif"

    img_l0 = np.zeros((512, 512, 3), dtype=np.uint8)
    img_l0[:256, :256] = [200, 50, 50]  # red   top-left
    img_l0[:256, 256:] = [50, 200, 50]  # green top-right
    img_l0[256:, :256] = [50, 50, 200]  # blue  bottom-left
    img_l0[256:, 256:] = [200, 200, 50]  # yellow bottom-right

    img_l1 = img_l0[::2, ::2]  # 256×256 — simple 2× subsample

    with tifffile.TiffWriter(str(path)) as tif:
        opts = dict(photometric="rgb", compression="deflate")
        tif.write(img_l0, tile=(256, 256), subfiletype=0, **opts)
        tif.write(img_l1, tile=(256, 256), subfiletype=1, **opts)

    return str(path)
