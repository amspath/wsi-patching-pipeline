from io import BytesIO
from pathlib import Path

import numpy as np
import orjson
import pytest

torch = pytest.importorskip("torch")

from wsi_patching.writers.materialize_writers.webdataset.webdataset_loader import WebDatasetLoader


# ------------------------ helpers ------------------------
def _png_bytes_from_array(arr_hw3: np.ndarray) -> bytes:
    """Encode HWC uint8 RGB array to PNG bytes using Pillow (skip test if not available)."""
    PIL = pytest.importorskip("PIL.Image")
    im = PIL.fromarray(arr_hw3)
    bio = BytesIO()
    im.save(bio, format="PNG")
    return bio.getvalue()


def _write_shard(tmpdir: Path, items):
    """Write a webdataset shard with given items: list of (key, img_bytes, meta_dict)."""
    import webdataset as wds

    shard = tmpdir / "shard-000000.tar"
    with wds.ShardWriter(str(tmpdir / "shard-%06d.tar"), maxcount=1000, verbose=0) as sink:  # type: ignore
        for key, img_bytes, meta in items:
            sink.write(
                {"__key__": key, "png": img_bytes, "meta": orjson.dumps(meta, option=orjson.OPT_SERIALIZE_NUMPY)}
            )
    # Some versions write an empty first shard on open; ensure we produced at least one shard file
    assert shard.exists()


# ------------------------ unit tests ------------------------
def test_key_in_train_prefix_filtering():
    loader = WebDatasetLoader(sampled_wsi_names=["WSI_A", "WSI_B"])
    assert loader.key_in_train({"__key__": "WSI_A/xyz"}) is True
    assert loader.key_in_train({"__key__": "WSI_B_001/abc"}) is True
    assert loader.key_in_train({"__key__": "OTHER/thing"}) is False
    # No filter configured -> always True
    loader2 = WebDatasetLoader(sampled_wsi_names=None)
    assert loader2.key_in_train({"__key__": "ANY"}) is True


def test_get_dataset_raises_if_no_shards(tmp_path: Path):
    loader = WebDatasetLoader(tar_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        _ = loader.get_dataset()


@pytest.mark.filterwarnings("ignore::UserWarning:webdataset.*")
def test_dataset_reads_and_filters(tmp_path: Path):
    # Build two tiny RGB images and write a shard
    img_a = np.zeros((4, 5, 3), dtype=np.uint8)
    img_b = np.full((4, 5, 3), 255, dtype=np.uint8)
    items = [("A/0001", _png_bytes_from_array(img_a), {"id": 1}), ("B/0001", _png_bytes_from_array(img_b), {"id": 2})]
    _write_shard(tmp_path, items)

    # Only keep keys starting with "A"
    loader = WebDatasetLoader(tar_dir=tmp_path, sampled_wsi_names=["A"], batch_size=2)
    ds = loader.get_dataset()

    got_keys = []
    for sample in ds:
        print(sample)
        # Each sample is a dict {"key", "image", "meta"}
        assert set(sample.keys()) == {"key", "patch", "meta"}
        got_keys.append(sample["key"])

    assert all(k.startswith("A") for k in got_keys)
    assert set(got_keys) == {"A/0001"}


@pytest.mark.filterwarnings("ignore::UserWarning:webdataset.*")
def test_dataloader_with_safe_collate_numpy_path(tmp_path: Path):
    # Two RGB images, ensure batch size 2
    img1 = np.zeros((3, 4, 3), dtype=np.uint8)  # H=3, W=4
    img2 = np.dstack(
        [
            np.zeros((3, 4), np.uint8),  # R=0
            np.full((3, 4), 128, np.uint8),  # G=128
            np.full((3, 4), 255, np.uint8),
        ]
    )  # B=255
    items = [("S/0001", _png_bytes_from_array(img1), {"i": 1}), ("S/0002", _png_bytes_from_array(img2), {"i": 2})]
    _write_shard(tmp_path, items)

    loader = WebDatasetLoader(tar_dir=tmp_path, batch_size=2)
    dl = loader.get_dataloader(
        num_workers=0,  # keep single-process to be stable in tests
        persistent_workers=False,  # required when num_workers=0
        prefetch_factor=None,  # not used when num_workers=0
        safe_collate=True,
        drop_last=False,
        pin_memory=False,
    )

    batches = list(dl)
    assert len(batches) == 1
    images, keys, metas = batches[0]

    # Shapes/types: [B, C, H, W], float32 in [0,1]
    assert images.shape == (2, 3, 3, 4)
    assert images.dtype == torch.float32
    assert images.min().item() >= 0.0 and images.max().item() <= 1.0

    assert keys == ["S/0001", "S/0002"] or keys == ["S/0002", "S/0001"]
    assert isinstance(metas[0], dict) and isinstance(metas[1], dict)


# ------------------------ safe_collate unit coverage ------------------------
def test_safe_collate_handles_empty_batch():
    loader = WebDatasetLoader()
    patches, keys, metas = loader.safe_collate([])
    assert patches.shape == (0,)
    assert keys == []
    assert metas == []


def test_safe_collate_numpy_and_torch_inputs_and_rgba_drop_alpha():
    loader = WebDatasetLoader()

    # Numpy HWC uint8 (RGB) — H=4, W=5
    np_img = (np.random.rand(4, 5, 3) * 255).astype(np.uint8)

    # Torch CHW uint8 (RGBA) — match H=4, W=5; alpha will be dropped to 3 channels
    torch_img_rgba = torch.stack(
        [
            torch.zeros((4, 5), dtype=torch.uint8),  # R
            torch.full((4, 5), 128, dtype=torch.uint8),  # G
            torch.full((4, 5), 64, dtype=torch.uint8),  # B
            torch.full((4, 5), 255, dtype=torch.uint8),  # A (dropped)
        ],
        dim=0,
    )  # shape [4, 4, 5] -> CHW with 4 channels

    batch = [{"key": "k1", "patch": np_img, "meta": {"a": 1}}, {"key": "k2", "patch": torch_img_rgba, "meta": {"b": 2}}]

    images, keys, metas = loader.safe_collate(batch)

    assert images.shape == (2, 3, 4, 5)  # two items, 3 channels after dropping alpha
    assert torch.is_floating_point(images)
    assert 0.0 <= images.min() <= images.max() <= 1.0
    assert keys == ["k1", "k2"]
    assert metas == [{"a": 1}, {"b": 2}]


def test_safe_collate_rejects_bad_shapes_and_types():
    loader = WebDatasetLoader()

    # Bad numpy shape (HW)
    bad_np = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        loader.safe_collate([{"key": "x", "patch": bad_np, "meta": {}}])

    # Bad torch shape (not CHW)
    bad_torch = torch.zeros((5, 5), dtype=torch.float32)
    with pytest.raises(ValueError):
        loader.safe_collate([{"key": "y", "patch": bad_torch, "meta": {}}])

    # Unsupported type
    with pytest.raises(TypeError):
        loader.safe_collate([{"key": "z", "patch": "not-an-image", "meta": {}}])


def test_safe_collate_stack_failure_is_wrapped():
    loader = WebDatasetLoader()
    # Different spatial sizes to force stack failure
    np1 = (np.random.rand(3, 4, 3) * 255).astype(np.uint8)
    np2 = (np.random.rand(5, 6, 3) * 255).astype(np.uint8)

    with pytest.raises(RuntimeError) as ei:
        _ = loader.safe_collate([{"key": "a", "patch": np1, "meta": {}}, {"key": "b", "patch": np2, "meta": {}}])
    assert "Failed to stack patches" in str(ei.value)
