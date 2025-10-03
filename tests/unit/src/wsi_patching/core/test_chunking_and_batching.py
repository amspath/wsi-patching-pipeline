from dataclasses import dataclass
from typing import List, Tuple
from unittest.mock import patch

import numpy as np
import pytest

from wsi_patching.core.chunking_and_batching import ReadWindowChunker, RegionReadAndBatch, TilePlanner
from wsi_patching.regions_of_interest.rois import BoxROI
from wsi_patching.utils.meta_typing import PipelineContext


# ------------------- tiny stubs for types -------------------
@dataclass
class SlideStub:
    wsi_id: str
    wsi_path: str
    dims: Tuple[int, int]  # (W, H)
    meta: dict


@dataclass
class SlideWithROIsStub(SlideStub):
    rois: list


@dataclass
class TilePlanStub:
    wsi_id: str
    wsi_path: str
    dims: Tuple[int, int]
    roi_index: int
    roi_bounds: Tuple[int, int, int, int]
    tiles: List[Tuple[int, int]]
    meta: dict


@dataclass
class RegionTaskStub:
    wsi_id: str
    wsi_path: str
    region: Tuple[int, int, int, int]  # (x, y, w, h)
    tiles: List[Tuple[int, int]]
    meta: dict


def fake_read_region(path, x, y, w, h, level, use_gpu, num_workers_cucim):
    # Return an HxWx3 array filled with unique value (for sanity checks if desired)
    return np.full((h, w, 3), fill_value=11, dtype=np.uint8)


# ------------------- TilePlanner -------------------
def test_tileplanner_whole_slide_no_rois_generates_tiles():
    slide = SlideStub("S", "/s", (64, 64), {})
    tp = TilePlanner(tile_selection_mode="full_inside_bounds", tile_size=16, stride=16)
    # seed context
    tp.attach_context(PipelineContext({"tile_size": 16, "stride": 16, "level": 0}))
    tp.validate()

    plans = list(tp(iter([slide])))
    assert len(plans) == 1
    plan = plans[0]
    # 64x64 with 16 stride & tile -> grid at (0,0),(16,0),(32,0),(48,0) x (0,16,32,48) => 16 tiles
    assert len(plan.tiles) == 16
    assert plan.roi_bounds == (0, 0, 64, 64)
    # a couple of specific coordinates
    assert (0, 0) in plan.tiles and (48, 48) in plan.tiles


def test_tileplanner_center_mode_accepts_boundary_tiles():
    # ROI that starts at (8,8) sized 9x9; tile_size 16
    # full_inside_bounds would reject (0,0) tile, but center (8,8) lies inside -> accept in center_in_roi.
    slide = SlideWithROIsStub("S", "/s", (40, 40), {}, rois=[BoxROI(8, 8, 9, 9)])
    tp = TilePlanner(tile_size=16, stride=16, tile_selection_mode="center_in_roi")
    tp.attach_context(PipelineContext({"tile_size": 16, "stride": 16, "level": 0}))
    tp.validate()

    plans = list(tp(iter([slide])))
    assert len(plans) == 1
    plan = plans[0]
    assert (8, 8) in plan.tiles  # accepted by center rule
    # few tiles overall due to tiny ROI
    assert len(plan.tiles) >= 1


def test_tileplanner_warns_when_no_tiles(caplog):
    # Tiny slide 15x15 with tile_size 16 -> no tiles
    slide = SlideStub("S", "/s", (15, 15), {})
    tp = TilePlanner(tile_size=16, stride=16, tile_selection_mode="full_inside_bounds")
    tp.attach_context(PipelineContext({"tile_size": 16, "stride": 16, "level": 0}))
    tp.validate()

    caplog.set_level("WARNING")
    plans = list(tp(iter([slide])))
    # No emitted TilePlan (because there were no tiles)
    assert plans == []
    assert "No tiles found for slide S ROI 0" in caplog.text


# ------------------- ReadWindowChunker -------------------
def test_readwindowchunker_validate_defaults_and_guards(caplog):
    r = ReadWindowChunker(max_window_size=None)
    # seed context
    r.attach_context(PipelineContext({"tile_size": 32, "stride": 16}))
    caplog.set_level("INFO")
    r.validate()
    # default set to 20 * tile_size
    assert r.max_window_size == 640
    assert "Defaulting max_window_size" in caplog.text

    # now test guard for non-multiple
    r2 = ReadWindowChunker(max_window_size=30)
    r2.attach_context(PipelineContext({"tile_size": 16, "stride": 8}))
    with pytest.raises(ValueError):
        r2.validate()

    # large warning
    r3 = ReadWindowChunker(max_window_size=10016)
    r3.attach_context(PipelineContext({"tile_size": 16, "stride": 8}))
    caplog.clear()
    r3.validate()
    assert "quite large" in caplog.text


def test_readwindowchunker_groups_tiles_into_windows():
    # Construct a plan with two spatial groups. tile_size=16, stride=16.
    plan = TilePlanStub(
        wsi_id="S",
        wsi_path="/s",
        dims=(128, 64),
        roi_index=0,
        roi_bounds=(0, 0, 64, 32),
        tiles=[
            (0, 0),
            (16, 0),
            (0, 16),
            (16, 16),  # group 1
            (48, 0),
        ],  # group 2 (falls in window starting at x=32)
        meta={},
    )
    r = ReadWindowChunker(max_window_size=32)
    r.attach_context(PipelineContext({"tile_size": 16, "stride": 16}))
    r.validate()

    tasks = list(r(iter([plan])))
    # Two windows: [0,0,32,32] with four tiles; [32,0,32,16] with one tile
    assert len(tasks) == 2
    a, b = tasks
    assert a.region == (0, 0, 32, 32)
    assert sorted(a.tiles) == [(0, 0), (0, 16), (16, 0), (16, 16)]
    assert b.region == (32, 0, 32, 16)
    assert b.tiles == [(48, 0)]


# ------------------- RegionReadAndBatch -------------------
@patch("wsi_patching.core.chunking_and_batching.get_xp_backend", new=lambda use_gpu: np)
@patch("wsi_patching.core.chunking_and_batching.read_region", new=fake_read_region)
def test_region_read_and_batch_happy_path_and_batch_split():
    # Build RegionTasks for a region 48x32 with four 16x16 tiles and one extra -> batches of 3 then 2
    tasks = [
        RegionTaskStub(
            wsi_id="S",
            wsi_path="/s",
            region=(0, 0, 48, 32),
            tiles=[(0, 0), (16, 0), (32, 0), (0, 16), (16, 16)],
            meta={"m": 1},
        )
    ]

    r = RegionReadAndBatch(batch_size=3, num_workers=2, dtype=np.uint8)
    r.attach_context(PipelineContext({"tile_size": 16, "level": 0, "use_gpu": False}))
    r.validate()

    batches = list(r(iter(tasks)))
    # Expect 2 batches: 3 then 2
    assert len(batches) == 2
    b1, b2 = batches

    # CollatedPatchBatch-like: has fields wsi_id, coords, patches, meta
    assert b1.wsi_id == "S" and b2.wsi_id == "S"
    assert len(b1.coords) == 3 and len(b2.coords) == 2
    # patches should be numpy arrays via XPStub.asarray
    assert isinstance(b1.patches, np.ndarray) and isinstance(b2.patches, np.ndarray)
    assert b1.patches.shape[1:] == (16, 16, 3)


@patch("wsi_patching.core.chunking_and_batching.get_xp_backend", new=lambda use_gpu: np)
def test_region_read_and_batch_skips_incomplete_patches():
    # region 20x20, tile_size 16 -> tile at (8,8) gives rx=8, ry=8; patch 12x12 -> should be skipped
    def _fake_read_region(path, x, y, w, h, level, use_gpu, num_workers_cucim):
        return np.full((h, w, 3), 1, dtype=np.uint8)

    with patch("wsi_patching.core.chunking_and_batching.read_region", new=_fake_read_region):
        tasks = [
            RegionTaskStub(
                wsi_id="S",
                wsi_path="/s",
                region=(0, 0, 20, 20),
                tiles=[(0, 0), (8, 8)],  # second will be partial (12x12)
                meta={},
            )
        ]
        r = RegionReadAndBatch(batch_size=10, num_workers=1, dtype=np.uint8, edge_policy="drop")
        r.attach_context(PipelineContext({"tile_size": 16, "level": 0, "use_gpu": False}))
        r.validate()

        out = list(r(iter(tasks)))
        assert len(out) == 1
        batch = out[0]
        # only the full (0,0) patch remains
        assert all(batch.coords[0] == 0)
        assert batch.patches.shape[0] == 1


@patch("wsi_patching.core.chunking_and_batching.get_xp_backend", new=lambda use_gpu: np)
def test_region_read_and_batch_pads_incomplete_patches_with_zeros():
    # region 20x20, tile_size 16 -> tile at (8,8) gives rx=8, ry=8; patch 12x12 -> should be padded to 16x16 with zeros
    def _fake_read_region(path, x, y, w, h, level, use_gpu, num_workers_cucim):
        return np.full((h, w, 3), 1, dtype=np.uint8)

    with patch("wsi_patching.core.chunking_and_batching.read_region", new=_fake_read_region):
        tasks = [
            RegionTaskStub(
                wsi_id="S",
                wsi_path="/s",
                region=(0, 0, 20, 20),
                tiles=[(0, 0), (8, 8)],  # second will be partial (12x12)
                meta={},
            )
        ]
        r = RegionReadAndBatch(batch_size=10, num_workers=1, dtype=np.uint8, edge_policy="pad_with_zeros")
        r.attach_context(PipelineContext({"tile_size": 16, "level": 0, "use_gpu": False}))
        r.validate()

        out = list(r(iter(tasks)))
        assert len(out) == 1  # Still one collated batch
        batch = out[0]  # Get the batch
        assert all(batch.coords[1] == 8)
        assert np.all(batch.patches[1, 12:, :, :] == 0)


@patch("wsi_patching.core.chunking_and_batching.get_xp_backend", new=lambda use_gpu: np)
def test_region_read_and_batch_pads_incomplete_patches_with_edge():
    # region 20x20, tile_size 16 -> tile at (8,8) gives rx=8, ry=8; patch 12x12 -> should be padded to 16x16 with zeros
    def _fake_read_region(path, x, y, w, h, level, use_gpu, num_workers_cucim):
        return np.full((h, w, 3), 1, dtype=np.uint8)

    with patch("wsi_patching.core.chunking_and_batching.read_region", new=_fake_read_region):
        tasks = [
            RegionTaskStub(
                wsi_id="S",
                wsi_path="/s",
                region=(0, 0, 20, 20),
                tiles=[(0, 0), (8, 8)],  # second will be partial (12x12)
                meta={},
            )
        ]
        r = RegionReadAndBatch(batch_size=10, num_workers=1, dtype=np.uint8, edge_policy="pad_with_edge")
        r.attach_context(PipelineContext({"tile_size": 16, "level": 0, "use_gpu": False}))
        r.validate()

        out = list(r(iter(tasks)))
        assert len(out) == 1  # Still one collated batch
        batch = out[0]  # Get the batch
        assert all(batch.coords[1] == 8)
        assert np.all(batch.patches[1, 12:, :, :] == 1)
