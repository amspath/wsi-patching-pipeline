from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np

from wsi_patching.backends.cupy_numpy import is_cupy


class MetaColStorage:
    __slots__ = ("_data", "_n")

    def __init__(self, length: int) -> None:
        self._n = length
        self._data: Dict[str, Any] = {}

    def __len__(self) -> int:
        return self._n

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def get_sample(self, i: int) -> Dict[str, Any]:
        """Return a single sample as a dict of column name to value."""
        n = self._n
        if not (0 <= i < n):
            raise IndexError("Index out of range")

        return {k: self._data[k][i] for k in self.columns()}

    def get_all_row_wise(self) -> List[Dict[str, Any]]:
        """Return all metadata as a list of dictionaries, one per row."""
        return [self.get_sample(i) for i in range(self._n)]

    def columns(self) -> List[str]:
        return list(self._data.keys())

    def add_column(self, name: str, values: Union[list, np.ndarray]) -> None:
        if type(values) not in (list, np.ndarray):
            raise TypeError("Column values must be a list or numpy array")

        self._ensure_serializable(values, self._n)

        self._data[name] = values

    def drop_column(self, name: str) -> None:
        if name in self._data:
            del self._data[name]
        else:
            raise KeyError(f"Column '{name}' does not exist")

    def take(self, idx: np.ndarray) -> "MetaColStorage":
        """Return a new MetaColStorage with selected rows."""
        new_mcs = MetaColStorage(length=idx.sum())
        for k, v in self._data.items():
            new_mcs.add_column(k, self._take(v, idx))
        return new_mcs

    def _ensure_serializable(self, values: Any, n: int):
        """Ensure length n and leave types JSON-convertible or convertible."""
        # Allow numpy/cupy arrays or Python sequences
        if hasattr(values, "shape"):
            if values.shape[0] != n:
                raise ValueError("Column length mismatch")
            return values
        # Python sequence
        if len(values) != n:
            raise ValueError("Column length mismatch")
        return values

    def _take(self, x: Any, idx):
        """Take rows from array/list/column using idx or boolean mask."""
        if hasattr(x, "__array_interface__") or (is_cupy(x)):
            # numpy/cupy array
            return x[idx]

        # integer array over list
        return [x[idx] for idx, bool in enumerate(np.asarray(idx).tolist()) if bool]


@dataclass(frozen=True)
class EndOfStream:
    pass


@dataclass(frozen=True)
class EndOfQueue:
    pass
