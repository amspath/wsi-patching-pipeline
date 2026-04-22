"""
End-to-end pipeline benchmarks: WSIGrid → PatchExtractor → NumpyStreamWriter.

Measures total wall time to drain all patches from a 2048×2048 synthetic TIFF.

  128 px / stride 128  →  256 patches
  224 px / stride 224  →   81 patches
"""
from __future__ import annotations

import pytest

from wsi_patching.core import PatchExtractor, WSIGrid
from wsi_patching.writers import NumpyStreamWriter


def _run_pipeline(slide_path: str, tile_size: int, stride: int) -> int:
    p = (
        WSIGrid(slides=[slide_path], resolution=0, unit="level", use_gpu=False)
        .then(PatchExtractor(tile_size=tile_size, stride=stride, max_batch_size=256))
        .to(NumpyStreamWriter(layout="NHWC"))
    )
    total = sum(imgs.shape[0] for _, imgs, _, _ in p.stream(num_workers=1))
    return total


@pytest.mark.parametrize("tile_size,stride,expected", [(128, 128, 256), (224, 224, 100)])
def test_pipeline_throughput(benchmark, synthetic_slide_path, tile_size, stride, expected):
    result = {}

    def run():
        result["total"] = _run_pipeline(synthetic_slide_path, tile_size, stride)

    benchmark.pedantic(run, warmup_rounds=2, rounds=25, iterations=3)
    assert result["total"] == expected
