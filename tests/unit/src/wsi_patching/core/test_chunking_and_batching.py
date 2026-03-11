from unittest.mock import patch

import numpy as np
import pytest

from wsi_patching.core.chunking_and_batching import PatchExtractor, ReadWindowChunker, RegionReadAndBatch, TilePlanner
from wsi_patching.core.types.types import RegionTask, Slide, SlideWithROIs, TilePlan
from wsi_patching.regions_of_interest.rois import BoxROI
from wsi_patching.utils.meta_typing import PipelineContext


def fake_read_region(path, x, y, w, h, level, use_gpu, num_workers_cucim):
    # Return an HxWx3 array filled with unique value (for sanity checks if desired)
    return np.full((h, w, 3), fill_value=11, dtype=np.uint8)


# ------------------- TilePlanner -------------------
def test_tileplanner_whole_slide_no_rois_generates_tiles():
    slide = Slide("S", "/s", (64, 64), {})
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
    # full_inside_bounds would reject (0,0) tile, but center (16,16) lies inside -> accept in center_in_roi.
    slide = SlideWithROIs("S", "/s", (40, 40), {}, rois=[BoxROI(8, 8, 9, 9)])
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
    slide = Slide("S", "/s", (15, 15), {})
    tp = TilePlanner(tile_size=16, stride=16, tile_selection_mode="full_inside_bounds")
    tp.attach_context(PipelineContext({"tile_size": 16, "stride": 16, "level": 0}))
    tp.validate()

    caplog.set_level("WARNING")
    plans = list(tp(iter([slide])))
    # No emitted TilePlan (because there were no tiles)
    assert plans == []
    assert "No tiles found for slide S ROI 0" in caplog.text


def test_tileplanner_roi_smaller_than_tile_size_single_tile():
    """
    ROI is smaller than the tile size: we should only get a single tile
    anchored at the ROI start, even if stride < tile_size.
    """
    # ROI 1000x1000, tile_size 1024, stride 700
    slide = SlideWithROIs("S", "/s", (1000, 1000), {}, rois=[BoxROI(0, 0, 1000, 1000)])
    tp = TilePlanner(tile_size=1024, stride=700, tile_selection_mode="any_overlap")

    tp.attach_context(PipelineContext({"tile_size": 1024, "stride": 700, "level": 0}))
    tp.validate()

    plans = list(tp(iter([slide])))
    assert len(plans) == 1
    plan = plans[0]

    # Only a single tile is needed to cover the ROI
    assert len(plan.tiles) == 1
    assert plan.tiles[0] == (0, 0)


def test_tileplanner_large_roi_with_overlap_no_redundant_tiles():
    """
    ROI larger than tile, stride < tile_size:
    We want overlapping tiles that cover the ROI, but *not* an extra
    last row/column of redundant tiles (the 9 vs 4 bug).
    """
    # ROI 2000x2000, tile_size 1200, stride 900
    # Old behavior: 3x3 grid = 9 tiles.
    # New behavior (axis_positions): starts at [0, 900] -> 2x2 grid = 4 tiles.
    slide = SlideWithROIs("S", "/s", (2000, 2000), {}, rois=[BoxROI(0, 0, 2000, 2000)])
    tp = TilePlanner(tile_size=1200, stride=900, tile_selection_mode="any_overlap")

    tp.attach_context(PipelineContext({"tile_size": 1200, "stride": 900, "level": 0}))
    tp.validate()

    plans = list(tp(iter([slide])))
    assert len(plans) == 1
    plan = plans[0]

    # Expect exactly 4 tiles at the useful start positions
    expected_tiles = {(0, 0), (900, 0), (0, 900), (900, 900)}
    assert set(plan.tiles) == expected_tiles
    assert len(plan.tiles) == 4


def test_tileplanner_validate_warns_when_stride_larger_than_tile_size(caplog):
    """
    When stride > tile_size, validate() should emit a warning about gaps
    between tiles (behavior is allowed, but not guaranteed to cover ROI).
    """
    tp = TilePlanner(tile_size=16, stride=32, tile_selection_mode="any_overlap")
    tp.attach_context(PipelineContext({"tile_size": 16, "stride": 32, "level": 0}))

    caplog.set_level("WARNING")
    tp.validate()

    assert "Stride is larger than tile size" in caplog.text


# ------------------- ReadWindowChunker -------------------
def test_readwindowchunker_validate_defaults_and_guards(caplog):
    r = ReadWindowChunker(max_window_size=None)
    # seed context
    r.attach_context(PipelineContext({"tile_size": 32, "stride": 16}))
    caplog.set_level("INFO")
    r.validate()
    assert r.max_window_size == 4992
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
    plan = TilePlan(
        wsi_id="S",
        wsi_path="/s",
        dims=(128, 64),
        level=0,
        roi_index=0,
        roi_bounds=(0, 0, 64, 32),
        tiles=[
            # group 1
            (0, 0),
            (16, 0),
            (0, 16),
            (16, 16),
            # group 2 (falls in window starting at x=32)
            (48, 0),
        ],
        meta={},
    )
    r = ReadWindowChunker(max_window_size=32)
    r.attach_context(PipelineContext({"tile_size": 16, "stride": 16}))
    r.validate()

    tasks = list(r(iter([plan])))
    # Two windows: [0,0,32,32] with four tiles; [32,0,64,32] with one tile
    assert len(tasks) == 2
    a, b = tasks
    assert a.region == (0, 0, 32, 32)
    assert sorted(a.tiles) == [(0, 0), (0, 16), (16, 0), (16, 16)]
    assert b.region == (32, 0, 32, 32)
    assert b.tiles == [(48, 0)]


# ------------------- RegionReadAndBatch -------------------
@patch("wsi_patching.core.chunking_and_batching.get_xp_backend", new=lambda use_gpu: np)
@patch("wsi_patching.core.chunking_and_batching.read_region", new=fake_read_region)
def test_region_read_and_batch_happy_path_and_batch_split():
    # Build RegionTasks for a region 48x32 with four 16x16 tiles and one extra -> batches of 3 then 2
    tasks = [
        RegionTask(
            wsi_id="S",
            wsi_path="/s",
            wsi_dims=(64, 64),
            level=0,
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
            RegionTask(
                wsi_id="S",
                wsi_path="/s",
                wsi_dims=(20, 20),
                level=0,
                region=(0, 0, 20, 20),
                tiles=[(0, 0), (8, 8)],  # second will be partial (12x12)
                meta={},
            )
        ]
        r = RegionReadAndBatch(batch_size=10, num_workers=1, dtype=np.uint8, wsi_edge_policy="drop")
        r.attach_context(PipelineContext({"tile_size": 16, "level": 0, "use_gpu": False}))
        r.validate()

        out = list(r(iter(tasks)))
        assert len(out) == 1
        batch = out[0]
        # only the full (0,0) patch remains
        assert all(batch.coords[0] == 0)
        assert batch.patches.shape[0] == 1


@patch("wsi_patching.core.chunking_and_batching.get_xp_backend", new=lambda use_gpu: np)
def test_region_read_and_batch_pads_incomplete_patches_pad_with_zeros():
    # region 20x20, tile_size 16 -> tile at (8,8) gives rx=8, ry=8; patch 12x12 -> should be padded to 16x16 with zeros
    def _fake_read_region(path, x, y, w, h, level, use_gpu, num_workers_cucim):
        return np.full((h, w, 3), 1, dtype=np.uint8)

    with patch("wsi_patching.core.chunking_and_batching.read_region", new=_fake_read_region):
        tasks = [
            RegionTask(
                wsi_id="S",
                wsi_path="/s",
                wsi_dims=(20, 20),
                level=0,
                region=(0, 0, 20, 20),
                tiles=[(0, 0), (8, 8)],  # second will be partial (12x12)
                meta={},
            )
        ]
        r = RegionReadAndBatch(batch_size=10, num_workers=1, dtype=np.uint8, wsi_edge_policy="pad_with_zeros")
        r.attach_context(PipelineContext({"tile_size": 16, "level": 0, "use_gpu": False}))
        r.validate()

        out = list(r(iter(tasks)))
        assert len(out) == 1  # Still one collated batch
        batch = out[0]  # Get the batch
        assert all(batch.coords[1] == 8)
        assert np.all(batch.patches[1, 12:, :, :] == 0)


@patch("wsi_patching.core.chunking_and_batching.get_xp_backend", new=lambda use_gpu: np)
def test_region_read_and_batch_pads_incomplete_patches_pad_with_edge():
    # region 20x20, tile_size 16 -> tile at (8,8) gives rx=8, ry=8; patch 12x12 -> should be padded to 16x16 with edge
    def _fake_read_region(path, x, y, w, h, level, use_gpu, num_workers_cucim):
        return np.full((h, w, 3), 1, dtype=np.uint8)

    with patch("wsi_patching.core.chunking_and_batching.read_region", new=_fake_read_region):
        tasks = [
            RegionTask(
                wsi_id="S",
                wsi_path="/s",
                wsi_dims=(20, 20),
                level=0,
                region=(0, 0, 20, 20),
                tiles=[(0, 0), (8, 8)],  # second will be partial (12x12)
                meta={},
            )
        ]
        r = RegionReadAndBatch(batch_size=10, num_workers=1, dtype=np.uint8, wsi_edge_policy="pad_with_edge")
        r.attach_context(PipelineContext({"tile_size": 16, "level": 0, "use_gpu": False}))
        r.validate()

        out = list(r(iter(tasks)))
        assert len(out) == 1  # Still one collated batch
        batch = out[0]  # Get the batch
        assert all(batch.coords[1] == 8)
        assert np.all(batch.patches[1, 12:, :, :] == 1)


@patch("wsi_patching.core.chunking_and_batching.get_xp_backend", new=lambda use_gpu: np)
def test_region_read_and_batch_pads_incomplete_patches_within_roi_pad_with_zeros():
    # region 20x20, tile_size 16 -> tile at (8,8) gives rx=8, ry=8; patch 12x12 -> should be padded to 16x16 with zeros
    def _fake_read_region(path, x, y, w, h, level, use_gpu, num_workers_cucim):
        return np.full((h, w, 3), 1, dtype=np.uint8)

    with patch("wsi_patching.core.chunking_and_batching.read_region", new=_fake_read_region):
        tasks = [
            RegionTask(
                wsi_id="S",
                wsi_path="/s",
                wsi_dims=(40, 40),
                level=0,
                region=(0, 0, 20, 20),
                tiles=[(0, 0), (8, 8)],  # second will be partial (12x12)
                meta={},
            )
        ]
        r = RegionReadAndBatch(
            batch_size=10,
            num_workers=1,
            dtype=np.uint8,
            roi_edge_policy="use_wsi_edge_policy",
            wsi_edge_policy="pad_with_zeros",
        )
        r.attach_context(PipelineContext({"tile_size": 16, "level": 0, "use_gpu": False}))
        r.validate()

        out = list(r(iter(tasks)))
        assert len(out) == 1  # Still one collated batch
        batch = out[0]  # Get the batch
        assert all(batch.coords[1] == 8)
        assert np.all(batch.patches[1, 12:, :, :] == 0)


@patch("wsi_patching.core.chunking_and_batching.get_xp_backend", new=lambda use_gpu: np)
def test_region_read_and_batch_roi_edge_policy_read_from_image_expands_region():
    """
    When roi_edge_policy='read_from_image', RegionReadAndBatch should expand the
    region width/height to the next tile_size multiple, but not beyond wsi_dims.
    """
    recorded = {}

    def _fake_read_region(path, x, y, w, h, level, use_gpu, num_workers_cucim):
        recorded["size"] = (w, h)
        return np.zeros((h, w, 3), dtype=np.uint8)

    with patch("wsi_patching.core.chunking_and_batching.read_region", new=_fake_read_region):
        # region 20x20 within a 40x40 WSI, tile_size=16 -> should expand to 32x32
        tasks = [
            RegionTask(
                wsi_id="S", wsi_path="/s", wsi_dims=(40, 40), level=0, region=(0, 0, 20, 20), tiles=[(0, 0)], meta={}
            )
        ]
        r = RegionReadAndBatch(
            batch_size=10, num_workers=1, dtype=np.uint8, roi_edge_policy="read_from_image", wsi_edge_policy="drop"
        )
        r.attach_context(PipelineContext({"tile_size": 16, "level": 0, "use_gpu": False}))
        r.validate()

        list(r(iter(tasks)))
        assert recorded["size"] == (32, 32)


@patch("wsi_patching.core.chunking_and_batching.get_xp_backend", new=lambda use_gpu: np)
def test_region_read_and_batch_scales_coords_to_level0():
    """
    RegionReadAndBatch must convert level-N coordinates to level-0 before calling read_region.

    For a task with downsample=4.0 and region at level-2 coordinates (100, 200, 16, 16),
    read_region should be called with x=400, y=800 (level-0 coordinates).

    For a task with downsample=1.0 (level 0), coordinates should be passed unchanged.
    """
    recorded = {}

    def _fake_read_region(path, x, y, w, h, level, use_gpu, num_workers_cucim):
        recorded["xy"] = (x, y)
        return np.zeros((h, w, 3), dtype=np.uint8)

    with patch("wsi_patching.core.chunking_and_batching.read_region", new=_fake_read_region):
        # Non-zero level: downsample=4.0 means level-2 coords should be multiplied by 4
        tasks = [
            RegionTask(
                wsi_id="S",
                wsi_path="/s",
                wsi_dims=(2000, 2000),
                level=2,
                region=(100, 200, 16, 16),
                tiles=[(100, 200)],
                downsample=4.0,
                meta={},
            )
        ]
        r = RegionReadAndBatch(batch_size=10, num_workers=1, dtype=np.uint8)
        r.attach_context(PipelineContext({"tile_size": 16, "level": 2, "use_gpu": False}))
        r.validate()

        list(r(iter(tasks)))
        assert recorded["xy"] == (400, 800), (
            f"Expected level-0 coords (400, 800) but got {recorded['xy']}. "
            "read_region must receive level-0 coordinates."
        )

        # Level 0: downsample=1.0 means coordinates are passed unchanged
        tasks_l0 = [
            RegionTask(
                wsi_id="S",
                wsi_path="/s",
                wsi_dims=(2000, 2000),
                level=0,
                region=(50, 75, 16, 16),
                tiles=[(50, 75)],
                downsample=1.0,
                meta={},
            )
        ]
        r0 = RegionReadAndBatch(batch_size=10, num_workers=1, dtype=np.uint8)
        r0.attach_context(PipelineContext({"tile_size": 16, "level": 0, "use_gpu": False}))
        r0.validate()

        list(r0(iter(tasks_l0)))
        assert recorded["xy"] == (50, 75), (
            f"Expected unchanged coords (50, 75) for level-0 but got {recorded['xy']}."
        )


# ------------------- PatchExtractor (composite stage) -------------------
@patch("wsi_patching.core.chunking_and_batching.get_xp_backend", new=lambda use_gpu: np)
@patch("wsi_patching.core.chunking_and_batching.read_region", new=fake_read_region)
def test_patchextractor_end_to_end_whole_slide():
    """
    Basic end-to-end test of PatchExtractor using a whole slide with no ROIs.

    64x64 slide, tile_size=16, stride=16 -> 4x4 grid = 16 patches total.
    """
    slide = Slide("S", "/s", (64, 64), {})

    pe = PatchExtractor(
        tile_size=16,
        stride=16,
        max_batch_size=5,
        num_workers=0,
        wsi_edge_policy="pad_with_zeros",
        roi_edge_policy="use_wsi_edge_policy",
        dtype=np.uint8,
    )
    ctx = PipelineContext({"tile_size": 16, "stride": 16, "level": 0, "use_gpu": False})
    pe.attach_context(ctx)
    pe.validate()

    batches = list(pe(iter([slide])))
    assert len(batches) > 0

    total_patches = sum(b.patches.shape[0] for b in batches)
    # Expect full coverage: 4x4 = 16 patches
    assert total_patches == 16
