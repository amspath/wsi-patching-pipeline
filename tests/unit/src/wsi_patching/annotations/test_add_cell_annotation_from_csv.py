# tests/test_add_cell_annotation_from_csv.py
import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from wsi_patching.annotations.add_cell_annotation_from_csv import AddCellAnnotationFromCSV
from wsi_patching.core.types.types import CollatedPatchBatch
from wsi_patching.utils.meta_typing import PipelineContext


# ---------- helpers ----------
def write_points_csv(path: Path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["x", "y", "label"])
        w.writeheader()
        for x, y, lab in rows:
            w.writerow({"x": x, "y": y, "label": lab})


def make_batch(wsi_id: str, coords: np.ndarray, n_channels: int = 3, patch_hw: int = 4):
    """
    Build a minimal CollatedPatchBatch: patches just zeros, metadata=None, use_gpu=False.
    """
    # (N, C, H, W) or (N, H, W, C) isn't important for this stage; it never touches pixels.
    # We'll use a simple (N, H, W, C) to be friendly with downstream tooling.
    N = coords.shape[0]
    patches = np.zeros((N, patch_hw, patch_hw, n_channels), dtype=np.uint8)
    return CollatedPatchBatch(wsi_id=wsi_id, coords=coords.astype(np.int32), patches=patches, use_gpu=False)


def run_stage_collect(stage: AddCellAnnotationFromCSV, batches: Iterable[CollatedPatchBatch]):
    """Run stage.__call__ over an iterable and collect yielded batches to a list."""
    stage.attach_context(PipelineContext({"tile_size": 16, "use_gpu": False}))
    return list(stage(iter(batches)))


# ---------- tests ----------
def test_annotations_added_basic(tmp_path):
    """Test basic functionality: annotations added correctly to tiles."""
    # CSV with two points falling into two different tiles
    csv_path = tmp_path / "a.csv"
    pts = [
        (5, 5, 1),  # inside tile [0,16) x [0,16)
        (20, 20, 2),  # inside tile [16,32) x [16,32)
    ]
    write_points_csv(csv_path, pts)

    stage = AddCellAnnotationFromCSV({"slideA": str(csv_path)}, filter_empty=False)

    coords = np.array(
        [
            [0, 0],  # should capture (5,5,1)
            [16, 16],  # should capture (20,20,2)
            [0, 16],  # empty
        ],
        dtype=np.int32,
    )
    batch = make_batch("slideA", coords)

    [out] = run_stage_collect(stage, [batch])

    # meta column must exist and have per-tile arrays (k,3) with int32
    anns = out.metadata["annotations"]
    assert isinstance(anns, list) and len(anns) == 3

    a0, a1, a2 = anns
    # first tile has one annotation
    assert a0.shape == (1, 3)
    assert a0.dtype == np.int32
    assert (a0[0] == np.array([5, 5, 1], dtype=np.int32)).all()

    # second tile has one annotation
    assert a1.shape == (1, 3)
    assert (a1[0] == np.array([4, 4, 2], dtype=np.int32)).all()

    # third tile empty with shape (0,3)
    assert a2.shape == (0, 3)

    # no filtering requested, lengths unchanged
    assert len(out.patches) == 3
    assert out.coords.shape[0] == 3


def test_filter_empty_removes_tiles_without_annotations(tmp_path):
    """Test that tiles without annotations are removed when filter_empty=True."""
    csv_path = tmp_path / "a.csv"
    pts = [(5, 5, 1)]
    write_points_csv(csv_path, pts)

    stage = AddCellAnnotationFromCSV({"slideA": str(csv_path)}, filter_empty=True)

    coords = np.array(
        [
            [0, 0],  # has (5,5,1)
            [16, 16],  # empty
            [0, 16],  # empty
        ],
        dtype=np.int32,
    )
    batch = make_batch("slideA", coords)

    [out] = run_stage_collect(stage, [batch])

    # Only the first tile should remain
    assert out.coords.shape == (1, 2)
    np.testing.assert_array_equal(out.coords[0], [0, 0])

    anns = out.metadata["annotations"]
    assert len(anns) == 1
    assert anns[0].shape == (1, 3)
    np.testing.assert_array_equal(anns[0][0], np.array([5, 5, 1], dtype=np.int32))


def test_strict_bounds_left_inclusive_right_exclusive(tmp_path):
    """Test that points on tile edges are handled correctly."""
    # Points placed exactly on edges to test (>= x0) & (< x1) and same for y
    csv_path = tmp_path / "a.csv"
    pts = [
        (0, 0, 10),  # included in tile [0,16) x [0,16)
        (15, 15, 11),  # included in tile [0,16) x [0,16)
        (16, 16, 12),  # NOT in [0,16) but included in [16,32)
    ]
    write_points_csv(csv_path, pts)

    stage = AddCellAnnotationFromCSV({"slideA": str(csv_path)}, filter_empty=False)

    coords = np.array(
        [
            [0, 0],  # should see (0,0,10) and (15,15,11) but NOT (16,16,12)
            [16, 16],  # should see (16,16,12)
        ],
        dtype=np.int32,
    )
    batch = make_batch("slideA", coords)

    [out] = run_stage_collect(stage, [batch])

    anns0, anns1 = out.metadata["annotations"]
    # first tile has two points
    assert anns0.shape == (2, 3)
    assert sorted(map(tuple, anns0.tolist())) == [(0, 0, 10), (15, 15, 11)]
    # second tile has one (the edge one)
    assert anns1.shape == (1, 3)
    assert tuple(anns1[0]) == (0, 0, 12)
