from __future__ import annotations

from wsi_patching.core import PatchExtractor, WSIGrid
from wsi_patching.regions_of_interest import AttachROIs, RectROIProvider
from wsi_patching.regions_of_interest.roi_providers import RectROIfromXMLProvider
from wsi_patching.writers import NumpyStreamWriter


def test_rect_roi_limits_patches(synthetic_slide):
    """RectROIProvider covering top half → 8 patches (4×2)."""
    path, wsi_id = synthetic_slide
    p = (
        WSIGrid(slides=[path], resolution=0, unit="level", use_gpu=False)
        .then(AttachROIs(providers=[RectROIProvider(rois={wsi_id: [(0, 0, 512, 256)]})]))
        .then(PatchExtractor(tile_size=128, stride=128, max_batch_size=100))
        .to(NumpyStreamWriter(layout="NHWC"))
    )
    patches_list = [imgs for _, imgs, _, _ in p.stream(cpu_processes=1)]
    total = sum(b.shape[0] for b in patches_list)
    assert total == 8  # 4×2 grid in 512×256 ROI


def test_xml_roi_limits_patches(synthetic_slide, synthetic_xml_annotation):
    """RectROIfromXMLProvider with 256×256 ROI → 4 patches (2×2)."""
    path, wsi_id = synthetic_slide
    xml_path, _ = synthetic_xml_annotation
    p = (
        WSIGrid(slides=[path], resolution=0, unit="level", use_gpu=False)
        .then(AttachROIs(providers=[RectROIfromXMLProvider(rois={wsi_id: xml_path})]))
        .then(PatchExtractor(tile_size=128, stride=128, max_batch_size=100))
        .to(NumpyStreamWriter(layout="NHWC"))
    )
    patches_list = [imgs for _, imgs, _, _ in p.stream(cpu_processes=1)]
    total = sum(b.shape[0] for b in patches_list)
    assert total == 4  # XML ROI is 256×256 → 2×2 grid
