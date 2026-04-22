"""
Benchmarks for PNGEncoder.

- Single-patch encoding: raw PIL encode speed (compress_level=1).
- Batch encoding: ThreadPoolExecutor throughput for batches of 32 and 64 patches.
"""

from __future__ import annotations

import numpy as np
import pytest

from wsi_patching.encoders import PNGEncoder


@pytest.fixture(scope="module")
def encoder():
    return PNGEncoder(compress_level=1)


@pytest.fixture(scope="module")
def single_patch():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)


def test_encode_single_patch(benchmark, encoder, single_patch):
    def _run():
        return encoder._encode_one(single_patch)

    result = benchmark.pedantic(_run, warmup_rounds=3, rounds=50, iterations=5)
    assert len(result) > 0


@pytest.mark.parametrize("batch_size", [32, 64])
def test_encode_batch(benchmark, encoder, make_patch_batch, batch_size):
    batch = make_patch_batch(batch_size, tile_size=224)

    def _run():
        return list(encoder([batch]))

    results = benchmark.pedantic(_run, warmup_rounds=3, rounds=50, iterations=5)
    assert len(results) == 1
    assert len(results[0].encoded_patches) == batch_size
