from __future__ import annotations

import numpy as np
import pytest
import tifffile

from wsi_patching.core.types.types import CollatedPatchBatch


@pytest.fixture(scope="session")
def synthetic_slide_path(tmp_path_factory):
    """2048×2048 RGB TIFF — large enough for meaningful patch counts."""
    tmp = tmp_path_factory.mktemp("bench_slides")
    path = tmp / "bench_slide.tif"
    img = np.zeros((2048, 2048, 3), dtype=np.uint8)
    img[:1024, :1024] = [200, 50, 50]
    img[:1024, 1024:] = [50, 200, 50]
    img[1024:, :1024] = [50, 50, 200]
    img[1024:, 1024:] = [200, 200, 50]
    tifffile.imwrite(str(path), img, photometric="rgb", tile=(256, 256), compression="deflate")
    return str(path)


@pytest.fixture
def make_patch_batch():
    """Factory: returns a CollatedPatchBatch with random uint8 patches."""

    def _make(n: int, tile_size: int = 224) -> CollatedPatchBatch:
        rng = np.random.default_rng(42)
        patches = rng.integers(0, 256, size=(n, tile_size, tile_size, 3), dtype=np.uint8)
        coords = np.stack([np.arange(n), np.zeros(n, dtype=np.int64)], axis=1).astype(np.int64)
        return CollatedPatchBatch(wsi_id="bench", coords=coords, patches=patches, use_gpu=False)

    return _make
