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
    img[:256, :256] = [200, 50, 50]   # red top-left
    img[:256, 256:] = [50, 200, 50]   # green top-right
    img[256:, :256] = [50, 50, 200]   # blue bottom-left
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
