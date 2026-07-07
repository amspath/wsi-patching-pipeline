import numpy as np
import pytest

from wsi_patching.core.types.types import CollatedPatchBatch
from wsi_patching.filtering.remove_edge_tiles import RemoveEdgeTiles
from wsi_patching.utils.meta_typing import PipelineContext


def make_batch(coords, wsi_width, wsi_height, wsi_id="slideA", with_dims=True):
    coords = np.asarray(coords, dtype=np.int64)
    n = coords.shape[0]
    patches = np.zeros((n, 2, 2, 3), dtype=np.uint8)
    batch = CollatedPatchBatch(patches=patches, wsi_id=wsi_id, coords=coords, use_gpu=False)
    if with_dims:
        batch.add_meta_column("slide.width", np.full(n, wsi_width))
        batch.add_meta_column("slide.height", np.full(n, wsi_height))
    return batch


# -----------------------
# Parameter validation
# -----------------------
def test_init_rejects_negative_depth():
    with pytest.raises(ValueError, match="depth must be >= 0"):
        RemoveEdgeTiles(depth=-1)


def test_validate_requires_tile_size():
    f = RemoveEdgeTiles()
    f.attach_context(PipelineContext({}))
    with pytest.raises(KeyError):
        f.validate()
    f.attach_context(PipelineContext({"tile_size": 224}))
    f.validate()  # should not raise


# -----------------------
# Core filtering
# -----------------------
def test_removes_border_ring():
    tile = 224
    W = H = 1000
    positions = [0, 224, 448, 672, 896]  # 896 + 224 = 1120 > 1000 -> a border tile
    coords = [(x, y) for y in positions for x in positions]

    filt = RemoveEdgeTiles(depth=1)
    filt.attach_context(PipelineContext({"tile_size": tile}))

    out = list(filt([make_batch(coords, W, H)]))
    assert len(out) == 1
    kept = {tuple(c) for c in out[0].coords.tolist()}

    # margin = 224; keep iff 224 <= coord < 776  ->  {224, 448, 672}
    inner = [224, 448, 672]
    expected = {(x, y) for y in inner for x in inner}
    assert kept == expected
    assert out[0].patches.shape[0] == len(expected)


def test_depth_zero_is_noop():
    coords = [(0, 0), (224, 224), (448, 0)]
    filt = RemoveEdgeTiles(depth=0)
    filt.attach_context(PipelineContext({"tile_size": 224}))

    out = list(filt([make_batch(coords, 1000, 1000)]))
    assert len(out) == 1
    assert out[0].patches.shape[0] == len(coords)
    assert {tuple(c) for c in out[0].coords.tolist()} == set(coords)


def test_depth_two_removes_two_rings():
    tile = 100
    W = H = 1000  # tiles at 0..900
    positions = list(range(0, 1000, 100))
    coords = [(x, y) for y in positions for x in positions]

    filt = RemoveEdgeTiles(depth=2)
    filt.attach_context(PipelineContext({"tile_size": tile}))

    out = list(filt([make_batch(coords, W, H)]))
    kept = {tuple(c) for c in out[0].coords.tolist()}

    # margin = 200; keep iff 200 <= coord < 800 -> {200,300,400,500,600,700}
    inner = [200, 300, 400, 500, 600, 700]
    expected = {(x, y) for y in inner for x in inner}
    assert kept == expected


def test_all_tiles_on_border_yields_nothing():
    coords = [(0, 0), (896, 0), (0, 896), (896, 896)]
    filt = RemoveEdgeTiles(depth=1)
    filt.attach_context(PipelineContext({"tile_size": 224}))

    out = list(filt([make_batch(coords, 1000, 1000)]))
    assert out == []


def test_missing_dims_metadata_raises():
    coords = [(224, 224)]
    filt = RemoveEdgeTiles(depth=1)
    filt.attach_context(PipelineContext({"tile_size": 224}))

    with pytest.raises(KeyError, match="slide.width"):
        list(filt([make_batch(coords, 1000, 1000, with_dims=False)]))


def test_metadata_columns_are_filtered_in_sync():
    coords = [(0, 0), (448, 448), (896, 896)]  # only the middle survives depth=1
    batch = make_batch(coords, 1000, 1000)
    batch.add_meta_column("tag", np.array(["a", "b", "c"]))

    filt = RemoveEdgeTiles(depth=1)
    filt.attach_context(PipelineContext({"tile_size": 224}))

    out = list(filt([batch]))[0]
    assert out.patches.shape[0] == 1
    assert list(out.metadata.get("tag")) == ["b"]
    assert tuple(out.coords[0]) == (448, 448)
