# test_torch_stream_writer.py
import importlib.util

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wsi_patching.writers.stream_writers.torch_stream_writer import TorchStreamWriter


def _mk_bhwc(n, h, w, c, dtype=np.float32):
    """Create a BHWC array with sequential values; easy to check later."""
    return np.arange(n * h * w * c, dtype=dtype).reshape(n, h, w, c)


class _DummyMetadata:
    def __init__(self, rows):
        self._rows = rows

    def get_all_row_wise(self):
        return self._rows


class _DummyBatch:
    """Lightweight stand-in for CollatedPatchBatch with only the fields the writer uses."""

    def __init__(self, wsi_id, patches, coords, metadata_rows):
        self.wsi_id = wsi_id
        self.patches = patches
        self.coords = coords
        self.metadata = _DummyMetadata(metadata_rows)
        # attribute present in your real CollatedPatchBatch; not required by TorchStreamWriter
        self.use_gpu = False


@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.float16])
def test_stream_single_batch_layout_and_dtype(layout, out_dtype):
    N, H, W, C = 3, 4, 5, 3
    patches = _mk_bhwc(N, H, W, C, dtype=np.float64)  # non-default input dtype
    coords = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.int32)
    meta_rows = {"level": [0, 0, 1], "tile_ix": [1, 2, 3]}

    # Provide device explicitly to avoid needing a PipelineContext in validate()
    writer = TorchStreamWriter(layout=layout, dtype=out_dtype, device=torch.device("cpu"))

    (wsi_id, imgs_t, coords_t, meta_out) = next(
        writer.stream(_DummyBatch(wsi_id="S1", patches=patches, coords=coords, metadata_rows=meta_rows))  # type: ignore
    )

    # Basic returns
    assert wsi_id == "S1"
    assert imgs_t.dtype == out_dtype
    assert imgs_t.device.type == "cpu"
    assert coords_t.dtype == torch.long
    assert coords_t.device.type == "cpu"
    assert meta_out == meta_rows

    # Shapes and value checks
    if layout == "NCHW":
        assert imgs_t.shape == (N, C, H, W)
        # BHWC -> NCHW: [n=0, y=0, x=0, ch=1] -> [0, 1, 0, 0]
        assert torch.isclose(imgs_t[0, 1, 0, 0], torch.as_tensor(patches[0, 0, 0, 1], dtype=out_dtype))
    else:
        assert imgs_t.shape == (N, H, W, C)
        assert torch.isclose(imgs_t[0, 0, 0, 1], torch.as_tensor(patches[0, 0, 0, 1], dtype=out_dtype))

    # Coords preserved numerically (but upcast to long)
    torch.testing.assert_close(coords_t, torch.as_tensor(coords, dtype=torch.long))


def test_stream_single_channel_preserves_values_and_shapes():
    N, H, W, C = 2, 2, 2, 1
    patches = _mk_bhwc(N, H, W, C, dtype=np.float32)
    coords = np.array([[1, 2], [3, 4]], dtype=np.int64)
    meta_rows = [{"a": 1}, {"a": 2}]

    writer = TorchStreamWriter(layout="NCHW", dtype=torch.float32, device=torch.device("cpu"))
    wsi_id, imgs_t, coords_t, meta_out = next(
        writer.stream(_DummyBatch(wsi_id="A", patches=patches, coords=coords, metadata_rows=meta_rows))  # type: ignore
    )

    assert wsi_id == "A"
    assert imgs_t.shape == (N, C, H, W)
    # Expected first sample: HWC -> CHW
    expected0 = torch.as_tensor(np.transpose(patches[0], (2, 0, 1)), dtype=torch.float32)
    assert torch.equal(imgs_t[0], expected0)
    torch.testing.assert_close(coords_t, torch.as_tensor(coords, dtype=torch.long))
    assert meta_out == meta_rows


def test_stream_length_mismatch_raises():
    N, H, W, C = 2, 2, 2, 1
    patches = _mk_bhwc(N, H, W, C, dtype=np.float32)  # N = 2
    coords = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)  # 3 -> mismatch

    writer = TorchStreamWriter(layout="NCHW", device=torch.device("cpu"))

    with pytest.raises(ValueError) as ei:
        _ = next(writer.stream(_DummyBatch("X", patches, coords, metadata_rows={})))  # type: ignore
    msg = str(ei.value)
    assert "Batch length mismatch" in msg
    assert "wsi_id=X" in msg


@pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="cupy not installed")
def test_stream_accepts_cupy_array_on_cpu_device():
    import cupy as cp

    N, H, W, C = 2, 3, 4, 2
    patches_cu = cp.arange(N * H * W * C, dtype=cp.float32).reshape(N, H, W, C)
    coords = np.array([[7, 8], [9, 10]], dtype=np.int32)
    meta_rows = {"ok": [True, True]}

    writer = TorchStreamWriter(layout="NHWC", dtype=torch.float32, device=torch.device("cpu"))
    wsi_id, imgs_t, coords_t, meta_out = next(
        writer.stream(_DummyBatch(wsi_id="CUPY", patches=patches_cu, coords=coords, metadata_rows=meta_rows))  # type: ignore
    )

    assert wsi_id == "CUPY"
    assert imgs_t.device.type == "cpu"
    assert imgs_t.dtype == torch.float32
    assert imgs_t.shape == (N, H, W, C)
    torch.testing.assert_close(coords_t, torch.as_tensor(coords, dtype=torch.long))
    assert meta_out == meta_rows
