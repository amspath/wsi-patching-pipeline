"""
Benchmarks for TilePlanner._generate_tiles_vectorized — pure NumPy, no I/O.

A 10 000×10 000 virtual bounding box is used so tile counts are large:
  224 px / stride 224  →  ~2 000 tiles
  128 px / stride 128  →  ~6 100 tiles
  128 px / stride  64  →  ~24 000 tiles  (overlapping)
"""

from __future__ import annotations

import pytest

from wsi_patching.core.chunking_and_batching import TilePlanner
from wsi_patching.regions_of_interest.rois import BoxROI

SIDE = 10_000
ROI = BoxROI(x=0, y=0, w=SIDE, h=SIDE)


def _plan(tile_size: int, stride: int, mode: str = "any_overlap") -> list:
    planner = TilePlanner(tile_size=tile_size, stride=stride, tile_selection_mode=mode)  # type: ignore
    return planner._generate_tiles_vectorized(ROI, 0, 0, SIDE, SIDE, SIDE, SIDE, tile_size, stride)  # type: ignore[arg-type]


@pytest.mark.parametrize("tile_size,stride", [(224, 224), (128, 128), (128, 64)])
def test_tile_planner_any_overlap(benchmark, tile_size, stride):
    def _run():
        return _plan(tile_size, stride, "any_overlap")

    tiles = benchmark.pedantic(_run, warmup_rounds=3, rounds=100, iterations=20)
    assert len(tiles) > 0


@pytest.mark.parametrize("tile_size,stride", [(224, 224), (128, 128), (128, 64)])
def test_tile_planner_full_inside_bounds(benchmark, tile_size, stride):
    def _run():
        return _plan(tile_size, stride, "full_inside_bounds")

    tiles = benchmark.pedantic(_run, warmup_rounds=3, rounds=100, iterations=20)
    assert len(tiles) > 0
