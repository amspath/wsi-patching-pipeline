import logging
import random
from pathlib import Path
from typing import List, Union

import webdataset as wds

from wsi_patching.utils.logging_config import init_logging
from wsi_patching.utils.types import EncodedPatch, EndOfQueue, EndOfStream


class WebDatasetWriter:
    """
    Writer for WebDataset shards.

    Usage:
    - Single process: call with an iterable of samples.
    - Multi process: call `start_writer(queue, outdir, shard_size, shuffle_buffer_size)`.

    Each sample should have:
      - "__key__" (str)
      - "sample_bytes" (bytes)
      - "json_bytes" (bytes)
    """

    def __init__(self, outdir: Path = "./output/", shard_size: int = 200, shuffle_buffer_size: int = 500):
        self.outdir = Path(outdir)
        self.shard_size = int(shard_size)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.shard_pattern = str(self.outdir / "shard-%06d.tar")
        self.write_count = 0

    def start_writer(self, queue) -> None:
        """Multi-process mode: consume from queue and write shards."""
        init_logging()
        logging.info("Writer process started.")
        self.outdir.mkdir(parents=True, exist_ok=True)
        sink = wds.ShardWriter(self.shard_pattern, maxcount=self.shard_size, verbose=0)

        buffer: List[EncodedPatch] = []
        while True:
            sample: Union[EncodedPatch, EndOfStream, EndOfQueue] = queue.get()
            if isinstance(sample, EndOfQueue):
                logging.info("Received shutdown signal.")
                break
            if isinstance(sample, EndOfStream):
                continue
            buffer.append(sample)
            if len(buffer) >= self.shuffle_buffer_size:
                self._flush_buffer(buffer, sink)

        if buffer:
            self._flush_buffer(buffer, sink)

        logging.info(f"Writer processed {self.write_count} samples.")
        sink.close()

    def _flush_buffer(self, buffer: List[EncodedPatch], sink: wds.ShardWriter) -> None:
        logging.info(f"Flushing buffer of size: {len(buffer)}")
        random.shuffle(buffer)
        for _ in range(min(self.shard_size, len(buffer))):
            s = buffer.pop()
            self.write_count += 1
            sink.write({"__key__": s.key, "png": s.patch_bytes, "json": s.json_dict})
        logging.info(f"Buffer size after flush: {len(buffer)}")
