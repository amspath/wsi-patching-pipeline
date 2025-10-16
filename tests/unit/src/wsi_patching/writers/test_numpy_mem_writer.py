import numpy as np
import pytest

from wsi_patching.core.types.types import CollatedPatchBatch
from wsi_patching.writers.numpy_stream_writer import NumpyMemoryWriter


def _mk_bhwc(n, h, w, c, dtype=np.float32):
    """Create a BHWC array with sequential values; easy to check later."""
    x = np.arange(n * h * w * c, dtype=dtype).reshape(n, h, w, c)
    return x


def test_close_empty_yields_empty_outputs():
    w = NumpyMemoryWriter()
    # No writes, close immediately
    w.close()
    ids, imgs, coords, meta = w.get_output()
    assert imgs.shape == (0, 1, 1, 1)
    assert imgs.dtype == np.float32
    assert coords.shape == (0, 2)
    assert coords.dtype == np.int64
    assert ids == []


@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
def test_write_single_batch_and_close(layout):
    N, H, W, C = 3, 4, 5, 3
    patches = _mk_bhwc(N, H, W, C, dtype=np.float64)  # non-default dtype to test conversion
    coords = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.int32)

    w = NumpyMemoryWriter(layout=layout, dtype=np.float32)
    # write one batch from slide "S1"
    w.write(CollatedPatchBatch(wsi_id="S1", patches=patches, coords=coords, use_gpu=False))
    w.close()

    ids, imgs, coords, meta = w.get_output()

    # dtype conversion and ids replicated per patch
    assert imgs.dtype == np.float32
    assert coords.dtype == np.int64
    assert ids == ["S1"] * N

    # shape/layout checks
    if layout == "NCHW":
        assert imgs.shape == (N, C, H, W)
        # verify a few values moved to channel-first correctly
        # e.g., pixel [0, y=0, x=0, ch=1] in BHWC becomes imgs[0, 1, 0, 0]
        assert imgs[0, 1, 0, 0] == np.float32(patches[0, 0, 0, 1])
    else:
        assert imgs.shape == (N, H, W, C)
        assert imgs[0, 0, 0, 1] == np.float32(patches[0, 0, 0, 1])

    # coords passed through as 64-bit ints
    np.testing.assert_array_equal(coords, coords.astype(np.int64))


def test_multiple_batches_are_concatenated_and_ids_extended():
    # First batch: 2 items, second: 3 items
    b1 = CollatedPatchBatch(
        wsi_id="A", patches=_mk_bhwc(2, 2, 2, 1), coords=np.array([[1, 2], [3, 4]], dtype=np.int64), use_gpu=False
    )
    b2 = CollatedPatchBatch(
        wsi_id="B",
        patches=_mk_bhwc(3, 2, 2, 1),
        coords=np.array([[5, 6], [7, 8], [9, 10]], dtype=np.int64),
        use_gpu=False,
    )

    w = NumpyMemoryWriter(layout="NCHW")
    w.write(b1)
    w.write(b2)

    # get_output should auto-close if not yet closed
    ids, imgs, coords, meta = w.get_output()

    # 5 total samples
    assert imgs.shape[0] == 5
    assert coords.shape[0] == 5
    assert ids == ["A", "A", "B", "B", "B"]

    # NCHW with C=1, H=W=2
    assert imgs.shape[1:] == (1, 2, 2)

    # sanity: first sample should match transformed first element of b1
    # b1.patches is BHWC, C=1 so the NCHW slice should be [:,0,:,:]
    expected0 = np.transpose(b1.patches[0], (2, 0, 1))  # HWC -> CHW
    np.testing.assert_array_equal(imgs[0], expected0.astype(np.float32))
