from collections import defaultdict
from itertools import chain
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import orjson

from wsi_patching.core.pipeline import Stage
from wsi_patching.core.types.types import CollatedPatchBatch


class AddAnnotationFromGeoJSON(Stage):
    """Attach GeoJSON annotations to each patch.

    Generic reader for QuPath-style GeoJSON FeatureCollections. Each feature is
    assigned to the patch its geometry centroid falls into (strict left-inclusive,
    right-exclusive bounds, like AddCellAnnotationFromCSV). Per patch a list of
    cell dicts is added as a metadata column; each dict carries the configured
    feature properties and (optionally) the geometry shifted to patch-local int
    coordinates. The WebDatasetWriter JSON-serializes this column per sample.

    GeoJSON coords are assumed to be in level-0 pixels; patch coords live in the
    requested-resolution space (level-N / resampled). They are reconciled by scaling
    the GeoJSON coords by ``1 / slide.downsample`` (read from the patch metadata, so
    it works at any mpp/level, not just resolution=0, unit="level"). Override with
    ``coord_scale`` when the GeoJSON is not in level-0 pixels (e.g. microns: pass
    ``1 / target_mpp``; already-target-space: pass ``1.0``).
    """

    def __init__(
        self,
        wsi_to_geojson_mapping: Dict[str, str],
        property_keys: Optional[List[str]] = None,  # None = keep ALL feature properties
        include_geometry: bool = True,
        geometry_key: str = "geometry",
        column_name: str = "annotations",
        filter_empty: bool = False,
        wsi_offsets: Optional[Dict[str, Tuple[int, int]]] = None,
        coord_scale: Optional[float] = None,  # None = auto (1 / slide.downsample)
    ):
        self.wsi_to_geojson_mapping = wsi_to_geojson_mapping
        self.property_keys = property_keys
        self.include_geometry = include_geometry
        self.geometry_key = geometry_key
        self.column_name = column_name
        self.filter_empty = filter_empty
        self.wsi_offsets = wsi_offsets
        self.coord_scale = coord_scale

        # caches for the currently loaded WSI
        self._cx: Optional[np.ndarray] = None  # (N,) centroid x
        self._cy: Optional[np.ndarray] = None  # (N,) centroid y
        self._features: Optional[List[dict]] = None  # parsed GeoJSON features (kept as-is)
        self._buckets: Optional[Dict[Tuple[int, int], List[int]]] = None  # (gx,gy) grid cell -> fids
        self._ox: int = 0  # level-0 x offset of the currently loaded WSI's GeoJSON coords
        self._oy: int = 0  # level-0 y offset of the currently loaded WSI's GeoJSON coords

    def validate(self) -> None:
        self.ctx.require_key("tile_size")
        self.ctx.require_key("use_gpu")

    def __call__(self, it: Iterable[CollatedPatchBatch]) -> Iterable[CollatedPatchBatch]:
        it = iter(it)
        try:
            first: CollatedPatchBatch = next(it)
        except StopIteration:
            return

        tile_size: int = int(self.ctx["tile_size"])

        # GeoJSON (level-0 px) -> patch (requested-resolution) space scale.
        if self.coord_scale is not None:
            scale = float(self.coord_scale)
        else:
            ds_col = first.metadata.get("slide.downsample")
            scale = 1.0 / float(ds_col[0]) if ds_col is not None else 1.0

        self._ensure_loaded(first.wsi_id, tile_size, scale)
        it = chain([first], it)

        cx, cy, features, buckets = self._cx, self._cy, self._features, self._buckets
        assert cx is not None and cy is not None and features is not None and buckets is not None
        ox, oy = self._ox, self._oy

        for batch in it:
            all_annotations: List[List[dict]] = []

            for x0, y0 in batch.coords:
                x0 = int(x0)
                y0 = int(y0)
                x1 = x0 + tile_size
                y1 = y0 + tile_size

                cells: List[dict] = []
                for fid in _candidate_fids(buckets, x0, y0, x1, y1, tile_size):
                    if not (x0 <= cx[fid] < x1 and y0 <= cy[fid] < y1):
                        continue
                    feature = features[fid]
                    props = feature.get("properties", {})
                    if self.property_keys is None:
                        cell: dict = dict(props)
                    else:
                        cell = {k: props.get(k) for k in self.property_keys}
                    if self.include_geometry:
                        geom = feature.get("geometry") or {}
                        cell[self.geometry_key] = {
                            "type": geom.get("type"),
                            # (coord + off) * scale - patch_xy, folded into a single shift.
                            "coordinates": _shift_coords(
                                geom.get("coordinates", []), scale, x0 - ox * scale, y0 - oy * scale
                            ),
                        }
                    cells.append(cell)

                all_annotations.append(cells)

            batch.add_meta_column(self.column_name, all_annotations)

            if self.filter_empty:
                mask = np.array([len(a) != 0 for a in all_annotations], dtype=bool)
                batch.filter_on_mask(mask)

            if len(batch.patches) > 0:
                yield batch

    def _ensure_loaded(self, wsi_id: str, tile_size: int, scale: float = 1.0) -> None:
        if wsi_id not in self.wsi_to_geojson_mapping:
            raise KeyError(f"No GeoJSON file provided for WSI '{wsi_id}'")
        path = self.wsi_to_geojson_mapping[wsi_id]
        with open(path, "rb") as f:
            data = orjson.loads(f.read())

        ox, oy = self.wsi_offsets.get(wsi_id, (0, 0)) if self.wsi_offsets is not None else (0, 0)

        features = data["features"] if isinstance(data, dict) else list(data)
        n = len(features)

        # Flatten every feature's coords once, then compute all centroids with a single
        # segmented mean (np.add.reduceat) instead of one np.asarray+mean per feature.
        flat: List[Tuple[float, float]] = []
        counts = np.empty(n, dtype=np.int64)
        for i, feat in enumerate(features):
            pts = list(_iter_xy(feat.get("geometry", {}).get("coordinates", [])))
            counts[i] = len(pts)
            flat.extend(pts)

        cx = np.full(n, np.nan, dtype=np.float64)  # NaN = geometry-less, never matches
        cy = np.full(n, np.nan, dtype=np.float64)
        nz = counts > 0
        if flat:
            arr = np.asarray(flat, dtype=np.float64)  # (M, 2)
            starts = (np.cumsum(counts) - counts)[nz]  # segment starts of non-empty features
            sums = np.add.reduceat(arr, starts, axis=0)  # (num_nonempty, 2)
            cnz = counts[nz].astype(np.float64)
            # Scale level-0 centroids (with level-0 offset) into patch space.
            cx[nz] = (sums[:, 0] / cnz + ox) * scale
            cy[nz] = (sums[:, 1] / cnz + oy) * scale

        # Grid-bucket the centroids by tile_size cell -> O(1) patch lookup, no R-tree.
        buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        gx = np.floor_divide(cx[nz], tile_size).astype(np.int64)
        gy = np.floor_divide(cy[nz], tile_size).astype(np.int64)
        for fid, kx, ky in zip(np.nonzero(nz)[0], gx, gy):
            buckets[(int(kx), int(ky))].append(int(fid))

        self._features = features
        self._cx, self._cy = cx, cy
        self._ox, self._oy = ox, oy
        self._buckets = buckets


def _candidate_fids(buckets, x0, y0, x1, y1, tile_size):
    """Yield feature ids in the grid cells covering [x0,x1) x [y0,y1).

    Grid-aligned patches (x0,y0 multiples of tile_size) hit a single cell; the ranges
    keep it correct for unaligned / overlapping-stride patches too.
    """
    for gx in range(x0 // tile_size, (x1 - 1) // tile_size + 1):
        for gy in range(y0 // tile_size, (y1 - 1) // tile_size + 1):
            yield from buckets.get((gx, gy), ())


def _iter_xy(coords: Any):
    """Yield every (x, y) leaf coordinate pair from a GeoJSON coordinates structure."""
    if (
        isinstance(coords, (list, tuple))
        and len(coords) == 2
        and isinstance(coords[0], (int, float))
        and isinstance(coords[1], (int, float))
    ):
        yield coords[0], coords[1]
    else:
        for c in coords:
            yield from _iter_xy(c)


def _shift_coords(coords: Any, scale: float, x0: float, y0: float) -> Any:
    """Recursively rebuild coords as round(coord * scale - (x0, y0)), int-valued."""
    if (
        isinstance(coords, (list, tuple))
        and len(coords) == 2
        and isinstance(coords[0], (int, float))
        and isinstance(coords[1], (int, float))
    ):
        return [int(round(coords[0] * scale - x0)), int(round(coords[1] * scale - y0))]
    return [_shift_coords(c, scale, x0, y0) for c in coords]
