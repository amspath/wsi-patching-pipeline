import numpy as np
import pytest

from wsi_patching.core.types.types import CollatedPatchBatch
from wsi_patching.filtering.segmentation_mask_filter import SegmentationMaskFilter
from wsi_patching.utils.meta_typing import PipelineContext


def make_mask(h=100, w=100, tumor_box=None, value=255):
    """Grayscale uint8 mask; tumor_box = (y1, y2, x1, x2) filled with `value`."""
    mask = np.zeros((h, w), dtype=np.uint8)
    if tumor_box:
        y1, y2, x1, x2 = tumor_box
        mask[y1:y2, x1:x2] = value
    return mask


def make_batch(coords, wsi_id="slideA", roi_bounds=None, wsi_dims=(1000, 1000)):
    coords = np.asarray(coords, dtype=np.int64)
    n = coords.shape[0]
    patches = np.zeros((n, 2, 2, 3), dtype=np.uint8)
    batch = CollatedPatchBatch(
        patches=patches,
        wsi_id=wsi_id,
        coords=coords,
        use_gpu=False,
        wsi_dims=wsi_dims,
    )
    if roi_bounds is not None:
        batch.add_meta_column("roi_bounds", np.array([tuple(roi_bounds)] * n))
    return batch


def run(filt, batch):
    filt.attach_context(PipelineContext({"tile_size": 10}))
    return list(filt([batch]))


# -----------------------
# Parameter validation
# -----------------------
def test_init_rejects_bad_min_pixels():
    with pytest.raises(ValueError, match="min_foreground_pixels"):
        SegmentationMaskFilter({}, min_foreground_pixels=0)


def test_validate_requires_tile_size():
    f = SegmentationMaskFilter({"slideA": make_mask()})
    f.attach_context(PipelineContext({}))
    with pytest.raises(KeyError):
        f.validate()
    f.attach_context(PipelineContext({"tile_size": 224}))
    f.validate()  # should not raise


def test_missing_mask_raises():
    f = SegmentationMaskFilter({})
    batch = make_batch([(0, 0)], roi_bounds=(0, 0, 100, 100))
    with pytest.raises(KeyError, match="no mask provided"):
        run(f, batch)


# -----------------------
# Placement via roi_bounds (identity scale, origin ROI)
# -----------------------
def test_keeps_tumor_drops_background_identity_scale():
    mask = make_mask(100, 100, tumor_box=(30, 70, 30, 70))
    f = SegmentationMaskFilter({"slideA": mask})
    # roi_bounds (x, y, w, h) == mask dims -> scale 1:1, start (0, 0)
    batch = make_batch([(35, 35), (5, 5), (80, 80)], roi_bounds=(0, 0, 100, 100))
    out = run(f, batch)[0]
    kept = {tuple(c) for c in out.coords.tolist()}
    assert kept == {(35, 35)}  # only the in-tumor tile survives


def test_out_of_roi_coord_dropped():
    mask = make_mask(100, 100, tumor_box=(0, 100, 0, 100))  # all tumor
    f = SegmentationMaskFilter({"slideA": mask})
    batch = make_batch([(200, 200)], roi_bounds=(0, 0, 100, 100))
    assert run(f, batch) == []  # nothing left -> batch not yielded


def test_empty_mask_yields_nothing():
    mask = make_mask(100, 100, tumor_box=None)
    f = SegmentationMaskFilter({"slideA": mask})
    batch = make_batch([(10, 10), (50, 50)], roi_bounds=(0, 0, 100, 100))
    assert run(f, batch) == []


def test_metadata_columns_filtered_in_sync():
    mask = make_mask(100, 100, tumor_box=(40, 60, 40, 60))
    f = SegmentationMaskFilter({"slideA": mask})
    batch = make_batch([(45, 45), (5, 5)], roi_bounds=(0, 0, 100, 100))
    batch.add_meta_column("tag", np.array(["keep", "drop"]))
    out = run(f, batch)[0]
    assert out.patches.shape[0] == 1
    assert list(out.metadata.get("tag")) == ["keep"]


# -----------------------
# Placement via roi_bounds (2x scale, non-origin ROI)
# -----------------------
def test_2x_scale_maps_level0_to_mask():
    # ROI [0,0] extent (200, 200) over a 100x100 mask -> scale 2.0.
    # tile_size=10 -> footprint 5px. Tumor around mask (50, 50).
    mask = make_mask(100, 100, tumor_box=(48, 60, 48, 60))
    f = SegmentationMaskFilter({"slideA": mask})
    batch = make_batch([(100, 100)], roi_bounds=(0, 0, 200, 200))
    out = run(f, batch)[0]
    assert out.coords.shape[0] == 1


def test_roi_start_offset_applied():
    # ROI starts at (1000, 2000), 100x100 extent over 100x100 mask -> scale 1.
    mask = make_mask(100, 100, tumor_box=(48, 60, 48, 60))
    f = SegmentationMaskFilter({"slideA": mask})
    inside = make_batch([(1050, 2050)], roi_bounds=(1000, 2000, 100, 100))
    assert run(f, inside)[0].coords.shape[0] == 1
    before = make_batch([(10, 10)], roi_bounds=(1000, 2000, 100, 100))
    assert run(f, before) == []


# -----------------------
# Explicit placement override
# -----------------------
def test_placement_override_takes_priority_over_roi_bounds():
    mask = make_mask(100, 100, tumor_box=(40, 60, 40, 60))
    # Placement says identity/origin; roi_bounds claims a bogus far-away ROI.
    f = SegmentationMaskFilter(
        {"slideA": mask}, wsi_placements={"slideA": (0.0, 0.0, 1.0, 1.0)}
    )
    batch = make_batch([(50, 50)], roi_bounds=(9000, 9000, 100, 100))
    assert run(f, batch)[0].coords.shape[0] == 1


def test_placement_used_without_roi_bounds():
    mask = make_mask(100, 100, tumor_box=(40, 60, 40, 60))
    f = SegmentationMaskFilter(
        {"slideA": mask}, wsi_placements={"slideA": (0.0, 0.0, 1.0, 1.0)}
    )
    batch = make_batch([(50, 50)])  # no roi_bounds metadata at all
    assert run(f, batch)[0].coords.shape[0] == 1


def test_no_placement_and_no_roi_bounds_raises():
    mask = make_mask(100, 100, tumor_box=(40, 60, 40, 60))
    f = SegmentationMaskFilter({"slideA": mask})
    batch = make_batch([(50, 50)])  # no roi_bounds
    with pytest.raises(KeyError, match="roi_bounds"):
        run(f, batch)


# -----------------------
# min_foreground_pixels
# -----------------------
def test_min_foreground_pixels_threshold():
    # A single tumor pixel at mask (50, 50); tile_size=10, scale 1 -> footprint 10x10.
    mask = make_mask(100, 100, tumor_box=(50, 51, 50, 51))  # exactly 1 pixel
    keep_any = SegmentationMaskFilter({"s": mask}, min_foreground_pixels=1)
    drop_thresh = SegmentationMaskFilter({"s": mask}, min_foreground_pixels=2)
    b1 = make_batch([(45, 45)], wsi_id="s", roi_bounds=(0, 0, 100, 100))
    b2 = make_batch([(45, 45)], wsi_id="s", roi_bounds=(0, 0, 100, 100))
    assert run(keep_any, b1)[0].coords.shape[0] == 1
    assert run(drop_thresh, b2) == []


# -----------------------
# Mask loaded from a file path
# -----------------------
def test_mask_from_path(tmp_path):
    import cv2

    mask = make_mask(100, 100, tumor_box=(30, 70, 30, 70))
    p = tmp_path / "mask.png"
    cv2.imwrite(str(p), mask)
    f = SegmentationMaskFilter({"slideA": str(p)})
    batch = make_batch([(35, 35), (5, 5)], roi_bounds=(0, 0, 100, 100))
    out = run(f, batch)[0]
    assert {tuple(c) for c in out.coords.tolist()} == {(35, 35)}
