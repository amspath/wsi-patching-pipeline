import logging
import random
from pathlib import Path
from typing import List, Optional

import webdataset as wds

from wsi_patching.utils.types import Patch
from wsi_patching.writers.writer import WriterBase


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
        self._buffer: List[Patch] = []
        self._sink: Optional[wds.ShardWriter] = None

    def open(self) -> None:
        logging.info("WebDatasetWriter opening...")
        self.outdir.mkdir(parents=True, exist_ok=True)
        # allocate sink in the writer process
        self._sink = wds.ShardWriter(self.shard_pattern, maxcount=self.shard_size, verbose=0)

    def write(self, sample: Patch) -> None:
        self._buffer.append(sample)
        if len(self._buffer) >= self.shuffle_buffer_size:
            self._flush_buffer()

    def close(self) -> None:
        if self._buffer:
            self._flush_buffer()
        if self._sink is not None:
            logging.info(f"WebDatasetWriter processed {self.write_count} samples. Closing sink...")
            self._sink.close()
            self._sink = None

    def _flush_buffer(self) -> None:
        if not self._buffer or self._sink is None:
            return
        logging.info(f"[writer] Flushing buffer of size: {len(self._buffer)}")
        random.shuffle(self._buffer)
        # Write up to shard_size at a time for better mixing; leftover stays buffered.
        for _ in range(min(self.shard_size, len(self._buffer))):
            s = self._buffer.pop()
            self.write_count += 1
            self._sink.write({"__key__": s.key, "png": s.patch, "json": s.meta})
        logging.info(f"[writer] Buffer size after flush: {len(self._buffer)}")
