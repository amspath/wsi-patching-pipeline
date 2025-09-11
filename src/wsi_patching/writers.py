# ------------------------------
# Stages (Writer lane)
# ------------------------------
import io
import json
import os
import random
import tarfile
from typing import Iterable, List, Optional

import webdataset as wds

from wsi_patching.core import Stage
from wsi_patching.typing import Sample


class RandomizedShardWriter(Stage):
    """
    Writer stage that buffers 4×shard_size (configurable), shuffles the buffer,
    and writes exactly `shard_size` samples per shard. On EOS, flushes remaining
    samples into final shard(s). Assumes each input Sample has "png" (bytes).
    """

    placement = "writer"

    def __init__(self, pattern: str, shard_size: int = 500, buffer_multiplier: int = 4, seed: Optional[int] = None):
        self.pattern = pattern
        self.shard_size = int(shard_size)
        self.buffer_limit = int(buffer_multiplier) * self.shard_size
        self.rng = random.Random(seed) if seed is not None else random

        # If webdataset is unavailable, fall back to an internal tar shard writer
        self._use_wds = wds is not None

    # ---------------- internal helpers ----------------
    class _TarShardWriter:
        def __init__(self, pattern: str, shard_size: int):
            self.pattern = pattern
            self.shard_size = shard_size
            self._tar = None
            self._count = 0
            self._idx = 0
            self._open_new()

        def _open_new(self):
            if self._tar is not None:
                self._tar.close()
            path = self.pattern.replace("%06d", f"{self._idx:06d}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._tar = tarfile.open(path, "w")
            self._count = 0
            self._idx += 1

        def write(self, key: str, png_bytes: bytes, meta: dict):
            # rotate?
            if self._count >= self.shard_size:
                self._open_new()
            # png
            info = tarfile.TarInfo(name=f"{key}.png")
            info.size = len(png_bytes)
            self._tar.addfile(info, io.BytesIO(png_bytes))
            # json
            jbytes = json.dumps(meta).encode("utf-8")
            jinfo = tarfile.TarInfo(name=f"{key}.json")
            jinfo.size = len(jbytes)
            self._tar.addfile(jinfo, io.BytesIO(jbytes))
            self._count += 1

        def close(self):
            if self._tar is not None:
                self._tar.close()
                self._tar = None

    def _open_writer(self):
        if self._use_wds:
            return wds.ShardWriter(self.pattern, maxcount=self.shard_size)
        else:
            return self._TarShardWriter(self.pattern, self.shard_size)

    def _close_writer(self, writer):
        # both classes expose close()
        writer.close()

    def _write_one(self, writer, s: Sample):
        key = s.get("__key__")
        if not key:
            x, y = s["coord"]
            key = f"{s['wsi_id']}-{x}-{y}-L{s['level']}"
        meta = {k: v for k, v in s.items() if k not in ("png", "__key__")}
        if self._use_wds:
            writer.write({"__key__": key, "png": s["png"], "json": json.dumps(meta).encode("utf-8")})
        else:
            writer.write(key, s["png"], meta)

    # --------------- stage entrypoint -----------------
    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        """
        Consume an iterator of encoded samples (each with 'png' bytes) and
        write randomized shards. Yields nothing (sink).
        """
        buf: List[Sample] = []
        writer = self._open_writer()

        def write_shard_from_buffer():
            # assumes len(buf) >= self.shard_size
            shard = buf[: self.shard_size]
            del buf[: self.shard_size]
            for s in shard:
                self._write_one(writer, s)

        try:
            for s in it:
                buf.append(s)
                if len(buf) >= self.buffer_limit:
                    # Shuffle the entire buffer, then write exactly one shard
                    self.rng.shuffle(buf)
                    write_shard_from_buffer()

            # End of stream: flush everything left (may be multiple shards + final partial)
            while len(buf) >= self.shard_size:
                self.rng.shuffle(buf)
                write_shard_from_buffer()

            if buf:
                # Final partial shard (write as is after one last shuffle)
                self.rng.shuffle(buf)
                for s in buf:
                    self._write_one(writer, s)
                buf.clear()

        finally:
            self._close_writer(writer)

        # sinks yield nothing
        return iter(())


class ToWebDataset(Stage):
    """
    Single-process writer that drains encoded samples and writes to shards.
    Uses webdataset.ShardWriter if available; otherwise a minimal tar fallback.
    Each sample must have:
      - a unique key (we derive from wsi_id + coord + level),
      - "png" bytes payload,
      - optional JSON sidecar with metadata (auto-generated).
    """

    placement = "writer"

    def __init__(self, pattern: str, maxcount: int = 25_000):
        self.pattern = pattern
        self.maxcount = int(maxcount)

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        # Writer stages are executed inside the writer process (see writer_process_main).
        # We consume everything and yield nothing (sink).
        if wds is not None:
            shard = wds.ShardWriter(self.pattern, maxcount=self.maxcount)
            try:
                for s in it:
                    key = s.get("__key__") or f"{s['wsi_id']}-{s['coord'][0]}-{s['coord'][1]}-L{s['level']}"
                    meta = {k: v for k, v in s.items() if k not in ("png", "__key__")}
                    shard.write({"__key__": key, "png": s["png"], "json": json.dumps(meta).encode("utf-8")})
            finally:
                shard.close()
        else:
            # Minimal tar fallback (rolls by count)
            shard_idx = 0
            shard_count = 0
            tar = None

            def open_new_tar(idx: int):
                nonlocal tar
                tar_path = self.pattern.replace("%06d", f"{idx:06d}")
                os.makedirs(os.path.dirname(tar_path), exist_ok=True)
                tar = tarfile.open(tar_path, "w")

            def close_tar():
                nonlocal tar
                if tar is not None:
                    tar.close()
                    tar = None

            try:
                open_new_tar(shard_idx)
                for s in it:
                    if shard_count >= self.maxcount:
                        close_tar()
                        shard_idx += 1
                        shard_count = 0
                        open_new_tar(shard_idx)

                    key = s.get("__key__") or f"{s['wsi_id']}-{s['coord'][0]}-{s['coord'][1]}-L{s['level']}"
                    png_bytes = s["png"]
                    meta = {k: v for k, v in s.items() if k not in ("png", "__key__")}
                    # Write PNG
                    info = tarfile.TarInfo(name=f"{key}.png")
                    info.size = len(png_bytes)
                    tar.addfile(info, io.BytesIO(png_bytes))
                    # Write JSON
                    jbytes = json.dumps(meta).encode("utf-8")
                    jinfo = tarfile.TarInfo(name=f"{key}.json")
                    jinfo.size = len(jbytes)
                    tar.addfile(jinfo, io.BytesIO(jbytes))
                    shard_count += 1
            finally:
                close_tar()

        # Sink: yields nothing
        return iter(())
