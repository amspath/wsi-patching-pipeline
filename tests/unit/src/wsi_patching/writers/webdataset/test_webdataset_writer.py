import tarfile
from pathlib import Path

import numpy as np

from wsi_patching.writers.webdataset.webdataset_writer import WebDatasetWriter


class DummyPatch:
    """Minimal stand-in for wsi_patching.utils.types.Patch (attributes only)."""

    def __init__(self, key: str, h: int = 2, w: int = 3, c: int = 1):
        # tiny PNG-like array bytes; WebDataset doesn't validate content here
        self.key = key
        # Use raw bytes; ShardWriter will just store them as a .png member
        self.patch = np.zeros((h, w, c), dtype=np.uint8).tobytes()
        # Minimal JSON-serializable metadata
        self.meta = {"k": key}


def _list_shards(outdir: Path):
    return sorted(outdir.glob("shard-*.tar"))


def _count_members(tar_path: Path):
    if tar_path.stat().st_size == 0:
        return 0, 0

    with tarfile.open(tar_path, "r") as tf:
        names = [m.name for m in tf.getmembers() if m.isfile()]
    return (sum(n.endswith(".png") for n in names), sum(n.endswith(".meta") for n in names))


def test_open_creates_outdir_and_sink(tmp_path: Path):
    w = WebDatasetWriter(outdir=tmp_path, shard_size=5, shuffle_buffer_size=10)
    w.open()
    assert tmp_path.exists() and tmp_path.is_dir()

    # Some ShardWriter versions create shard-000000.tar on open.
    shards = _list_shards(tmp_path)
    assert len(shards) == 1
    if shards:
        # If present, it may be zero bytes; consider that "empty".
        pngs, jsons = _count_members(shards[0])
        assert pngs == 0 and jsons == 0

    w.close()


def test_write_below_threshold_does_not_flush_until_close(tmp_path: Path):
    w = WebDatasetWriter(outdir=tmp_path, shard_size=10, shuffle_buffer_size=5)
    w.open()

    # 3 < shuffle_threshold -> no flush yet
    for i in range(3):
        w.write(DummyPatch(f"k{i}"))

    assert len(_list_shards(tmp_path)) == 1  # Opening creates zero-byte shard
    w.close()

    shards = _list_shards(tmp_path)
    assert len(shards) == 1  # flushed on close
    # 3 samples -> 3 png + 3 json in the single shard
    pngs, jsons = _count_members(shards[0])
    assert pngs == 3 and jsons == 3
    assert w.write_count == 3


def test_flush_on_threshold_and_sharding(tmp_path: Path):
    # Small shard size to force multiple shards; low buffer to trigger flushes during write
    w = WebDatasetWriter(outdir=tmp_path, shard_size=3, shuffle_buffer_size=2)
    w.open()

    # 5 samples total => ceil(5/3)=2 shards expected after close
    for i in range(5):
        w.write(DummyPatch(f"s{i}"))

    # with threshold=2, at least one flush likely happened already, but total shards only guaranteed after close
    w.close()

    shards = _list_shards(tmp_path)
    assert len(shards) == 2
    # Total items across both shards: 5 samples -> 5 png + 5 json
    total_png = total_json = 0
    for sh in shards:
        p, j = _count_members(sh)
        total_png += p
        total_json += j
    assert total_png == 5 and total_json == 5
    assert w.write_count == 5


def test_multiple_flushes_with_immediate_threshold(tmp_path: Path):
    # shuffle_buffer_size=1 -> flush after each write; shard_size=2 -> every two samples new shard
    w = WebDatasetWriter(outdir=tmp_path, shard_size=2, shuffle_buffer_size=1)
    w.open()

    for i in range(3):
        w.write(DummyPatch(f"t{i}"))

    w.close()
    shards = _list_shards(tmp_path)
    # 3 samples with shard_size=2 ⇒ 2 shards
    assert len(shards) == 2
    # First shard should have 2 samples, second 1 sample
    png_counts = [_count_members(sh)[0] for sh in shards]
    assert sorted(png_counts) == [1, 2]
    assert w.write_count == 3


def test_close_is_idempotent(tmp_path: Path):
    w = WebDatasetWriter(outdir=tmp_path, shard_size=2, shuffle_buffer_size=1)
    w.open()
    w.write(DummyPatch("x"))
    w.close()

    shards_before = _list_shards(tmp_path)
    # Call close again — should not error or create more shards
    w.close()
    shards_after = _list_shards(tmp_path)
    assert shards_after == shards_before
