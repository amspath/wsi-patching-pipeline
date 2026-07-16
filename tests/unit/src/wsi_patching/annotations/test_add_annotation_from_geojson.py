from pathlib import Path
from typing import Iterable

import numpy as np
import orjson

from wsi_patching.annotations.add_annotation_from_geojson import AddAnnotationFromGeoJSON
from wsi_patching.core.types.types import CollatedPatchBatch
from wsi_patching.utils.meta_typing import PipelineContext


def _square(cx, cy, half=1):
    """A small square polygon centred on (cx, cy)."""
    return [[[cx - half, cy - half], [cx + half, cy - half], [cx + half, cy + half], [cx - half, cy + half]]]


def write_geojson(path: Path, cells):
    """cells: list of (cx, cy, label, celltype)."""
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": _square(cx, cy)},
            "properties": {"cell_label": label, "celltype": celltype},
        }
        for cx, cy, label, celltype in cells
    ]
    path.write_bytes(orjson.dumps({"type": "FeatureCollection", "features": features}))


def make_batch(wsi_id: str, coords: np.ndarray, patch_hw: int = 4):
    n = coords.shape[0]
    patches = np.zeros((n, patch_hw, patch_hw, 3), dtype=np.uint8)
    return CollatedPatchBatch(wsi_id=wsi_id, coords=coords.astype(np.int32), patches=patches, use_gpu=False)


def run_stage(stage: AddAnnotationFromGeoJSON, batches: Iterable[CollatedPatchBatch], tile_size: int = 16):
    stage.attach_context(PipelineContext({"tile_size": tile_size, "use_gpu": False}))
    return list(stage(iter(batches)))


def test_annotations_assigned_by_centroid_and_shifted(tmp_path):
    gj = tmp_path / "a.geojson"
    write_geojson(gj, [(5, 5, 1, "A"), (20, 20, 2, "B")])

    stage = AddAnnotationFromGeoJSON({"slideA": str(gj)}, filter_empty=False)
    coords = np.array([[0, 0], [16, 16], [0, 16]], dtype=np.int32)
    [out] = run_stage(stage, [make_batch("slideA", coords)])

    anns = out.metadata["annotations"]
    assert len(anns) == 3
    # tile (0,0) captures the (5,5) cell, geometry shifted to patch-local coords
    assert len(anns[0]) == 1
    cell = anns[0][0]
    assert cell["cell_label"] == 1 and cell["celltype"] == "A"
    assert cell["geometry"]["coordinates"] == _square(5, 5)  # x0,y0 = 0,0 -> unchanged
    # tile (16,16) captures the (20,20) cell, shifted by (16,16)
    assert len(anns[1]) == 1
    assert anns[1][0]["cell_label"] == 2
    assert anns[1][0]["geometry"]["coordinates"] == _square(4, 4)
    # tile (0,16) is empty
    assert anns[2] == []


def test_annotations_shifted_by_wsi_offset(tmp_path):
    """GeoJSON coords are relative to the slide's tissue bounds; wsi_offsets
    converts them into the same level-0 frame patches are read from."""
    gj = tmp_path / "a.geojson"
    write_geojson(gj, [(5, 5, 1, "A")])

    stage = AddAnnotationFromGeoJSON(
        {"slideA": str(gj)}, filter_empty=False, wsi_offsets={"slideA": (100, 200)}
    )
    # patch at (100,200) in level-0 frame == (0,0) in GeoJSON-relative frame
    coords = np.array([[100, 200]], dtype=np.int32)
    [out] = run_stage(stage, [make_batch("slideA", coords)])

    anns = out.metadata["annotations"]
    assert len(anns[0]) == 1
    assert anns[0][0]["geometry"]["coordinates"] == _square(5, 5)  # unchanged relative to patch origin


def test_filter_empty_and_property_subset(tmp_path):
    gj = tmp_path / "a.geojson"
    write_geojson(gj, [(5, 5, 1, "A")])

    stage = AddAnnotationFromGeoJSON(
        {"slideA": str(gj)}, property_keys=["cell_label"], include_geometry=False, filter_empty=True
    )
    coords = np.array([[0, 0], [16, 16]], dtype=np.int32)
    [out] = run_stage(stage, [make_batch("slideA", coords)])

    # only the non-empty tile survives
    assert out.coords.shape == (1, 2)
    np.testing.assert_array_equal(out.coords[0], [0, 0])
    cell = out.metadata["annotations"][0][0]
    assert cell == {"cell_label": 1}  # celltype dropped, geometry excluded


def test_downsample_scale_from_meta(tmp_path):
    """GeoJSON is level-0 px; patch coords are requested-resolution px. The stage
    scales by 1/slide.downsample read from the patch meta column."""
    gj = tmp_path / "a.geojson"
    write_geojson(gj, [(40, 40, 1, "A")])  # level-0 centroid

    # downsample=2 -> level-0 (40,40) maps to patch-space (20,20) -> tile (16,16).
    coords = np.array([[0, 0], [16, 16]], dtype=np.int32)
    batch = make_batch("slideA", coords)
    batch.add_meta_column("slide.downsample", np.array([2.0, 2.0]))

    stage = AddAnnotationFromGeoJSON({"slideA": str(gj)}, filter_empty=False)
    [out] = run_stage(stage, [batch])
    anns = out.metadata["annotations"]

    assert anns[0] == []
    assert len(anns[1]) == 1
    # square(40,40,half=1) * 0.5 - (16,16) -> centred on (4,4), half rounds to 0/1
    verts = anns[1][0]["geometry"]["coordinates"][0]
    assert all(0 <= x < 16 and 0 <= y < 16 for x, y in verts), verts
    assert [int(round(sum(v[i] for v in verts[:4]) / 4)) for i in (0, 1)] == [4, 4]


def test_coord_scale_override(tmp_path):
    """coord_scale overrides the auto 1/downsample (e.g. micron GeoJSON)."""
    gj = tmp_path / "a.geojson"
    write_geojson(gj, [(40, 40, 1, "A")])

    coords = np.array([[16, 16]], dtype=np.int32)
    batch = make_batch("slideA", coords)
    batch.add_meta_column("slide.downsample", np.array([2.0]))  # must be ignored

    stage = AddAnnotationFromGeoJSON({"slideA": str(gj)}, filter_empty=False, coord_scale=0.5)
    [out] = run_stage(stage, [batch])
    assert len(out.metadata["annotations"][0]) == 1


def test_matches_bruteforce_at_scale(tmp_path):
    """Grid-bucket matching must equal naive centroid-in-patch over many features,
    including geometry-less features and non-grid-aligned patches."""
    rng = np.random.default_rng(0)
    ts = 16
    n = 20_000
    cells = [(int(x), int(y), i, "A") for i, (x, y) in enumerate(rng.integers(0, 2000, (n, 2)))]

    gj = tmp_path / "big.geojson"
    features = [
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": _square(cx, cy)},
         "properties": {"cell_label": lbl}}
        for cx, cy, lbl, _ in cells
    ]
    # a geometry-less feature must be silently ignored (never matches)
    features.append({"type": "Feature", "geometry": {}, "properties": {"cell_label": -1}})
    gj.write_bytes(orjson.dumps({"type": "FeatureCollection", "features": features}))

    coords = rng.integers(0, 2000, (200, 2)).astype(np.int32)  # includes unaligned origins
    stage = AddAnnotationFromGeoJSON({"slideA": str(gj)}, property_keys=["cell_label"],
                                     include_geometry=False, filter_empty=False)
    [out] = run_stage(stage, [make_batch("slideA", coords)], tile_size=ts)
    anns = out.metadata["annotations"]

    for (x0, y0), got in zip(coords, anns):
        expected = {c[2] for c in cells if x0 <= c[0] < x0 + ts and y0 <= c[1] < y0 + ts}
        assert {c["cell_label"] for c in got} == expected
