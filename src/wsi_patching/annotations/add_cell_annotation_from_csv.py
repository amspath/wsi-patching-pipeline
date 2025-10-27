import csv
from itertools import chain
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from rtree import index

from wsi_patching.core.pipeline import Stage
from wsi_patching.core.types.types import CollatedPatchBatch


class AddCellAnnotationFromCSV(Stage):
    """
    Same behavior as your class, but without pandas.
    CSV is read with csv.DictReader -> NumPy arrays (x, y, label).
    R-tree is bulk-loaded once per WSI. Queries use pure NumPy masks.
    """

    def __init__(
        self,
        wsi_to_csv_mapping: Dict[str, str],
        x_col: str = "x",
        y_col: str = "y",
        label_col: str = "label",
        filter_empty: bool = False,
    ):
        self.wsi_to_csv_mapping = wsi_to_csv_mapping
        self.x_col = x_col
        self.y_col = y_col
        self.label_col = label_col
        self.filter_empty = filter_empty

        # caches for the currently loaded WSI
        self._x: Optional[np.ndarray] = None  # shape (N,), float or int
        self._y: Optional[np.ndarray] = None  # shape (N,), float or int
        self._label: Optional[np.ndarray] = None  # shape (N,), dtype=object or numeric
        self._rtree: Optional[index.Index] = None

    def validate(self) -> None:
        self.ctx.require_key("tile_size")
        self.ctx.require_key("use_gpu")

    def __call__(self, it: Iterable[CollatedPatchBatch]) -> Iterable[CollatedPatchBatch]:
        try:
            first: CollatedPatchBatch = next(it)
        except StopIteration:
            return

        # Ensure the right slide is loaded
        self._ensure_loaded(first.wsi_id)
        it = chain([first], it)

        x_arr, y_arr, lbl_arr, rtx = self._x, self._y, self._label, self._rtree
        assert x_arr is not None and y_arr is not None and lbl_arr is not None and rtx is not None

        tile_size: int = int(self.ctx["tile_size"])
        EMPTY = np.empty((0, 3))

        for batch in it:
            all_annotations: List[np.ndarray] = []

            # batch.coords is (N, 2) -> (x0,y0) top-left
            for x0, y0 in batch.coords:
                x1 = x0 + tile_size
                y1 = y0 + tile_size

                # R-tree returns candidate integer ids
                cand_ids = list(rtx.intersection((x0, y0, x1, y1)))

                if cand_ids:
                    # Use NumPy indexing & boolean mask (strict left-inclusive bounds like your code)
                    idx = np.fromiter(cand_ids, dtype=np.int64, count=len(cand_ids))
                    xs = x_arr[idx]
                    ys = y_arr[idx]

                    inside = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
                    if inside.any():
                        sel = idx[inside]
                        # Pack (k,3) object as before: (x, y, label)
                        k = sel.size
                        packed = np.empty((k, 3), dtype=np.int32)
                        packed[:, 0] = x_arr[sel] - x0
                        packed[:, 1] = y_arr[sel] - y0
                        packed[:, 2] = lbl_arr[sel]
                        all_annotations.append(packed)
                        continue

                all_annotations.append(EMPTY)

            batch.add_meta_column("annotations", all_annotations)

            if self.filter_empty:
                mask = np.array([a.shape[0] != 0 for a in all_annotations], dtype=bool)
                batch.filter_on_mask(mask)

            if len(batch.patches) > 0:
                yield batch

    def _ensure_loaded(self, wsi_id: str) -> None:
        if wsi_id not in self.wsi_to_csv_mapping:
            raise KeyError(f"No CSV file provided for WSI '{wsi_id}'")
        csv_path = self.wsi_to_csv_mapping[wsi_id]
        self._x, self._y, self._label = self._read_csv_to_arrays(csv_path, self.x_col, self.y_col, self.label_col)
        self._rtree = self._build_rtree(self._x, self._y)

    @staticmethod
    def _read_csv_to_arrays(
        csv_path: str, x_col: str, y_col: str, label_col: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Lightweight CSV to NumPy arrays. Works with numeric x/y and arbitrary label types (string/int).
        For ~50k rows this is fast and avoids pandas allocations entirely.
        """
        xs: List[float] = []
        ys: List[float] = []
        labels: List[object] = []

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            # Basic validation
            missing = {c for c in (x_col, y_col, label_col) if c not in reader.fieldnames}
            if missing:
                raise ValueError(f"CSV {csv_path} missing required column(s): {', '.join(sorted(missing))}")

            for row in reader:
                # Cast x/y as float (or int if you prefer). Keep label as object/string
                xs.append(float(row[x_col]))
                ys.append(float(row[y_col]))
                labels.append(row[label_col])

        x_arr = np.asarray(xs, dtype=np.uint32)
        y_arr = np.asarray(ys, dtype=np.uint32)
        label_arr = np.asarray(labels, dtype=np.int64)

        return x_arr, y_arr, label_arr

    @staticmethod
    def _build_rtree(x_arr: np.ndarray, y_arr: np.ndarray) -> index.Index:
        """
        Bulk build R-tree: much faster than calling insert() in a loop.
        """
        props = index.Property()
        props.dimension = 2

        # items format: (id, (xmin, ymin, xmax, ymax), obj)
        # We store points as zero-area rectangles (x, y, x, y). obj=None to keep it minimal.
        n = x_arr.shape[0]
        items = ((i, (x_arr[i], y_arr[i], x_arr[i], y_arr[i]), None) for i in range(n))
        return index.Index(items, properties=props)
