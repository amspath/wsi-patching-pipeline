import numpy as np
import pytest
import torch

from wsi_patching.core.types.types import CollatedPatchBatch
from wsi_patching.utils.meta_typing import PipelineContext
from wsi_patching.writers.torch_mem_writer import InMemoryPatchDataset, TorchMemoryWriter


def _mk_bhwc(n, h, w, c, dtype=np.float32):
    x = np.arange(n * h * w * c, dtype=dtype).reshape(n, h, w, c)
    return x


# ---------------- InMemoryPatchDataset ----------------
def test_inmemory_dataset_len_and_getitem():
    imgs = torch.zeros((3, 1, 2, 2))  # NCHW
    coords = torch.tensor([[0, 0], [1, 2], [3, 4]], dtype=torch.long)
    ids = ["A", "A", "B"]

    ds = InMemoryPatchDataset(imgs, coords, ids, layout="NCHW", metadata=[{}, {}, {}])
    assert len(ds) == 3
    item = ds[1]
    assert set(item.keys()) == {"image", "coord", "wsi_id", "meta"}
    assert torch.equal(item["coord"], coords[1])
    assert item["wsi_id"] == "A"


# ---------------- TorchMemoryWriter ----------------
def _attach_cpu_ctx_and_open(writer, monkeypatch, use_gpu=False):
    # Force device selection to CPU regardless of ctx setting
    monkeypatch.setattr(
        "wsi_patching.writers.torch_mem_writer.get_torch_device", lambda use_gpu_flag: torch.device("cpu"), raising=True
    )
    ctx = PipelineContext({"use_gpu": use_gpu})
    writer.attach_context(ctx)
    writer.validate()
    writer.open()


def test_empty_close_produces_empty_dataset(monkeypatch):
    w = TorchMemoryWriter(layout="NCHW", dtype=torch.float32)
    _attach_cpu_ctx_and_open(w, monkeypatch, use_gpu=False)

    w.close()
    ds = w.get_output()
    assert isinstance(ds, InMemoryPatchDataset)
    assert len(ds) == 0
    assert ds.images.shape == (0, 1, 1, 1)
    assert ds.coords.shape == (0, 2)
    assert ds.images.device.type == "cpu"
    assert ds.images.dtype == torch.float32


@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
def test_single_batch_roundtrip(monkeypatch, layout):
    N, H, W, C = 2, 3, 4, 3
    patches = _mk_bhwc(N, H, W, C, dtype=np.float64)  # non-default dtype on purpose
    coords = np.array([[10, 20], [30, 40]], dtype=np.int32)

    w = TorchMemoryWriter(layout=layout, dtype=torch.float32)
    _attach_cpu_ctx_and_open(w, monkeypatch)

    w.write(CollatedPatchBatch("S1", coords, patches, use_gpu=False))
    w.close()
    ds = w.get_output()

    assert len(ds) == N
    assert ds.images.device.type == "cpu"
    assert ds.images.dtype == torch.float32
    assert ds.coords.dtype == torch.long
    assert ds.wsi_ids == ["S1", "S1"]

    if layout == "NCHW":
        assert ds.images.shape == (N, C, H, W)
        # Check a couple of values moved to channel-first correctly
        # BHWC -> NCHW: pixel [0, y=0, x=0, ch=1] -> images[0, 1, 0, 0]
        assert torch.isclose(ds.images[0, 1, 0, 0], torch.tensor(patches[0, 0, 0, 1], dtype=torch.float32))
    else:
        assert ds.images.shape == (N, H, W, C)
        assert torch.isclose(ds.images[0, 0, 0, 1], torch.tensor(patches[0, 0, 0, 1], dtype=torch.float32))


def test_length_mismatch_raises(monkeypatch):
    N, H, W, C = 2, 2, 2, 1
    patches = _mk_bhwc(N, H, W, C)
    coords = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)  # 3 vs 2 -> mismatch

    w = TorchMemoryWriter(layout="NCHW")
    _attach_cpu_ctx_and_open(w, monkeypatch)

    with pytest.raises(ValueError) as ei:
        w.write(CollatedPatchBatch("X", coords, patches, use_gpu=False))
    assert "must have same first dimension" in str(ei.value)


def test_multiple_batches_concat_and_ids(monkeypatch):
    b1 = CollatedPatchBatch("A", np.array([[1, 2], [3, 4]], dtype=np.int64), _mk_bhwc(2, 2, 2, 1), use_gpu=False)
    b2 = CollatedPatchBatch(
        "B", np.array([[5, 6], [7, 8], [9, 10]], dtype=np.int64), _mk_bhwc(3, 2, 2, 1), use_gpu=False
    )

    w = TorchMemoryWriter(layout="NCHW")
    _attach_cpu_ctx_and_open(w, monkeypatch)

    w.write(b1)
    w.write(b2)

    # Don’t call close(): get_output() should finalize on demand
    ds = w.get_output()

    assert len(ds) == 5
    assert ds.images.shape == (5, 1, 2, 2)
    assert ds.coords.shape == (5, 2)
    assert ds.wsi_ids == ["A", "A", "B", "B", "B"]

    # sanity check first item equals permuted first element of b1
    expected0 = torch.as_tensor(np.transpose(b1.patches[0], (2, 0, 1)), dtype=torch.float32)  # HWC -> CHW
    assert torch.equal(ds.images[0], expected0)
