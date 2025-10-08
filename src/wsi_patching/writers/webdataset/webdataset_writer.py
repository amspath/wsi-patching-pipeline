import random
from pathlib import Path
from typing import List, Optional

import orjson
import webdataset as wds

from wsi_patching.core.types.types import EncodedCollatedPatchBatch
from wsi_patching.writers.writer_base import WriterBase


class WebDatasetWriter(WriterBase):
    """Writer for WebDataset shards."""

    def __init__(self, outdir: Path = Path("./output/"), shard_size: int = 200, shuffle_buffer_size: int = 500):
        super().__init__()
        self.outdir = outdir
        self.shard_size = int(shard_size)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.shard_pattern = str(self.outdir / "shard-%06d.tar")
        self.write_count = 0

        # runtime (allocated in open)
        self._buffer: List[EncodedCollatedPatchBatch] = []
        self._sink: Optional[wds.ShardWriter] = None

    def open(self) -> None:
        self.log.info("Opening...")
        self.outdir.mkdir(parents=True, exist_ok=True)
        # allocate sink in the writer process
        self._sink = wds.ShardWriter(self.shard_pattern, maxcount=self.shard_size, verbose=0)

    def write(self, batch: EncodedCollatedPatchBatch) -> None:
        for sample_idx in range(batch.coords.shape[0]):
            wsi_id, coord, _, meta = batch.get(sample_idx)
            key = f"{wsi_id}_{coord[0]}_{coord[1]}"
            encoded_patch = batch.encoded_patches[sample_idx]
            self._buffer.append((key, encoded_patch, meta))

            if len(self._buffer) >= self.shuffle_buffer_size:
                self._flush_buffer()

    def close(self) -> None:
        if self._buffer:
            self.log.info(f"Closing... Flushing remaining {len(self._buffer)} samples in buffer.")
            self.log.info(f"Meta example: {self._buffer[0][2]}")
        while self._buffer:
            self._flush_buffer()
        if self._sink is not None:
            self.log.info(f"Processed {self.write_count} samples. Closing sink...")
            self._sink.close()
            self._sink = None

    def _flush_buffer(self) -> None:
        if not self._buffer or self._sink is None:
            return
        self.log.info(f"Flushing buffer of size: {len(self._buffer)}")
        random.shuffle(self._buffer)
        # Write up to shard_size at a time for better mixing; leftover stays buffered.
        for _ in range(min(self.shard_size, len(self._buffer))):
            s = self._buffer.pop()
            self.write_count += 1
            self._sink.write(
                {"__key__": s[0], "png": s[1], "meta": orjson.dumps(s[2], option=orjson.OPT_SERIALIZE_NUMPY)}
            )
        self.log.info(f"Buffer size after flush: {len(self._buffer)}")
