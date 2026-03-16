from __future__ import annotations

from wsi_patching.core import PatchExtractor, WSIGrid
from wsi_patching.writers import NumpyStreamWriter


def test_whole_slide_patch_extraction(synthetic_slide):
    """Pipeline without ROI: extract all 16 patches from 512x512 at 128x128."""
    path, _ = synthetic_slide
    p = (
        WSIGrid(slides=[path], resolution=0, unit="level", use_gpu=False)
        .then(PatchExtractor(tile_size=128, stride=128, max_batch_size=100))
        .to(NumpyStreamWriter(layout="NHWC"))
    )
    patches_list = [imgs for _, imgs, _, _ in p.stream(cpu_processes=1)]
    total = sum(b.shape[0] for b in patches_list)
    assert total == 16  # 4×4 from 512/128
    assert patches_list[0].shape[1:] == (128, 128, 3)


def test_multi_slide_streaming(synthetic_slide):
    """Two slides passed as a list — both are processed."""
    path, _ = synthetic_slide
    p = (
        WSIGrid(slides=[path, path], resolution=0, unit="level", use_gpu=False)
        .then(PatchExtractor(tile_size=128, stride=128, max_batch_size=100))
        .to(NumpyStreamWriter(layout="NHWC"))
    )
    wsi_ids = [wid for wid, _, _, _ in p.stream(cpu_processes=1)]
    assert len(wsi_ids) == 2
