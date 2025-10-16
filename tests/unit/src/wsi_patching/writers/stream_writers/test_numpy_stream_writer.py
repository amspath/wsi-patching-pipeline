import numpy as np
import pytest

from wsi_patching.writers.stream_writers.numpy_stream_writer import NumpyStreamWriter


def _mk_bhwc(n, h, w, c, dtype=np.float32):
    """Create a BHWC array with sequential values; easy to check later."""
    return np.arange(n * h * w * c, dtype=dtype).reshape(n, h, w, c)


class _DummyMetadata:
    def __init__(self, rows):
        self._rows = rows

    def get_all_row_wise(self):
        return self._rows


class _DummyBatch:
    """Lightweight stand-in for CollatedPatchBatch."""

    def __init__(self, wsi_id, patches, coords, metadata_rows):
        self.wsi_id = wsi_id
        self.patches = patches
        self.coords = coords
        self.metadata = _DummyMetadata(metadata_rows)


@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
@pytest.mark.parametrize("out_dtype", [np.float32, np.float16])
def test_stream_single_batch_layout_and_dtype(layout, out_dtype):
    N, H, W, C = 3, 4, 5, 3
    # Non-default dtype to test conversion
    patches = _mk_bhwc(N, H, W, C, dtype=np.float64)
    coords = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.int32)
    meta_rows = {"level": [0, 0, 1], "tile_ix": [1, 2, 3]}

    writer = NumpyStreamWriter(layout=layout, dtype=out_dtype)

    (wsi_id, imgs, coords_np, meta_out) = next(
        writer.stream(_DummyBatch(wsi_id="S1", patches=patches, coords=coords, metadata_rows=meta_rows))
    )

    # Basic returns
    assert wsi_id == "S1"
    assert imgs.dtype == out_dtype
    assert coords_np.dtype == np.int64
    assert meta_out == meta_rows  # pass-through

    # Shapes and a value check for layout
    if layout == "NCHW":
        assert imgs.shape == (N, C, H, W)
        # BHWC -> BCHW: pixel [0, y=0, x=0, ch=1] becomes imgs[0, 1, 0, 0]
        assert imgs[0, 1, 0, 0] == out_dtype(patches[0, 0, 0, 1])
    else:
        assert imgs.shape == (N, H, W, C)
        assert imgs[0, 0, 0, 1] == out_dtype(patches[0, 0, 0, 1])

    # Coords preserved numerically (but upcast to int64)
    np.testing.assert_array_equal(coords_np, coords.astype(np.int64))


def test_stream_handles_single_channel_and_preserves_values():
    N, H, W, C = 2, 2, 2, 1
    patches = _mk_bhwc(N, H, W, C, dtype=np.float32)
    coords = np.array([[1, 2], [3, 4]], dtype=np.int64)
    meta_rows = [{"a": 1}, {"a": 2}]  # also works if this is a list of row dicts

    writer = NumpyStreamWriter(layout="NCHW", dtype=np.float32)

    wsi_id, imgs, coords_np, meta_out = next(
        writer.stream(_DummyBatch(wsi_id="A", patches=patches, coords=coords, metadata_rows=meta_rows))
    )

    assert wsi_id == "A"
    assert imgs.shape == (N, C, H, W)  # NCHW with C=1
    # Expected first sample: HWC -> CHW
    expected0 = np.transpose(patches[0], (2, 0, 1))
    np.testing.assert_array_equal(imgs[0], expected0)
    np.testing.assert_array_equal(coords_np, coords.astype(np.int64))
    assert meta_out == meta_rows


@pytest.mark.skipif(__import__("importlib").util.find_spec("cupy") is None, reason="cupy not installed")
def test_stream_converts_cupy_to_numpy_when_present():
    import cupy as cp

    N, H, W, C = 2, 3, 4, 2
    # Create CuPy BHWC
    patches_cu = cp.arange(N * H * W * C, dtype=cp.float32).reshape(N, H, W, C)
    coords = np.array([[7, 8], [9, 10]], dtype=np.int32)
    meta_rows = {"ok": [True, True]}

    writer = NumpyStreamWriter(layout="NHWC", dtype=np.float32)
    wsi_id, imgs, coords_np, meta_out = next(
        writer.stream(_DummyBatch(wsi_id="CUPY", patches=patches_cu, coords=coords, metadata_rows=meta_rows))
    )

    assert wsi_id == "CUPY"
    assert isinstance(imgs, np.ndarray)
    assert imgs.shape == (N, H, W, C)
    assert imgs.dtype == np.float32
    np.testing.assert_array_equal(coords_np, coords.astype(np.int64))
    assert meta_out == meta_rows
