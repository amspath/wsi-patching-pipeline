from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import orjson
import torch
import webdataset as wds
from torch.utils.data import DataLoader


class WebDatasetLoader:
    def __init__(
        self,
        outdir: Path = Path("./output/"),
        sampled_wsi_names: Optional[List[str]] = None,
        shuffle_size: int = 20_000,
        batch_size: int = 32,
    ):
        """
        Configurable dataset loader for PNG WebDataset shards that returns
        image, key, and meta (json).

        Args:
            num_workers (int): DataLoader workers. Higher means more randomness in batches.
                It is highly advised to use > 0 workers.
            outdir (Path): Directory containing *.tar shards.
            sampled_wsi_names (Sequence[str] | None): Keep a sample only if its __key__
                starts with any of these prefixes (WSI names). If None, keep all.
            shuffle_size (int): Per-worker shuffle buffer size (larger = more random).
            batch_size (int): DataLoader batch size.
        """
        self.outdir = outdir
        self.sampled_wsi_names = sampled_wsi_names
        self.shuffle_size = shuffle_size
        self.batch_size = batch_size

        self.shards: List[str] = []

    # ---- filtering helper ----------------------------------------------------
    def key_in_train(self, sample: Dict[str, Any]) -> bool:
        if not self.sampled_wsi_names:
            return True
        key = sample.get("__key__", "")
        # Fast prefix check against provided WSI names
        return any(key.startswith(prefix) for prefix in self.sampled_wsi_names)

    # ---- dataset & dataloader -----------------------------------------------
    def get_dataset(self) -> Iterable[Dict[str, Any]]:
        self.shards = sorted(glob(str(self.outdir / "*.tar")))
        if not self.shards:
            raise FileNotFoundError(f"No shards found in {self.outdir}")

        # Build core pipeline
        ds = wds.WebDataset(self.shards, shardshuffle=50)

        # Optional prefix-filter on __key__
        if self.sampled_wsi_names:
            ds = ds.select(self.key_in_train)

        # Shuffle within shards
        if self.shuffle_size and self.shuffle_size > 0:
            ds = ds.shuffle(self.shuffle_size)

        # Decode PNG to RGB8 (numpy HWC uint8) and meta to dict
        ds = ds.decode("torch").to_tuple("__key__", "image", "meta")

        # Map to a simple dict the rest of the code can rely on
        def _map(sample):
            key, img, meta = sample  # img: HWC uint8 (numpy), meta: dict
            return {"key": key, "image": img, "meta": orjson.loads(meta.decode("utf-8"))}

        ds = ds.map(_map)
        return ds

    def get_dataloader(
        self,
        num_workers: int = 4,
        drop_last: bool = True,
        pin_memory: bool = True,
        safe_collate: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 4,
    ) -> DataLoader:
        dataset = self.get_dataset()
        collate_fn = self.safe_collate if safe_collate else None

        extra = {}
        if persistent_workers is not None:
            extra["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            extra["prefetch_factor"] = prefetch_factor

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            **extra,
        )

    # ---- collate that “just works” ------------------------------------------
    def safe_collate(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        - Converts HWC uint8 numpy images to float32 tensors in [0,1] and stacks to [B, C, H, W].
        - Keeps keys as a list[str].
        - Keeps metas as a list[dict].
        """
        if not batch:
            return {"images": torch.empty(0), "keys": [], "metas": []}

        # Images: convert each HWC uint8 numpy -> CHW float32 tensor in [0,1]
        imgs: List[torch.Tensor] = []
        for sample in batch:
            img = sample["image"]
            if isinstance(img, np.ndarray):
                # numpy HWC -> torch CHW
                t = torch.from_numpy(img)  # [H, W, C], uint8
                if t.ndim != 3 or t.shape[-1] not in (1, 3, 4):
                    raise ValueError(f"Unexpected image shape: {t.shape}")
                t = t.permute(2, 0, 1).contiguous()  # CHW
                # If RGBA, drop alpha
                if t.shape[0] == 4:
                    t = t[:3, ...]
                t = t.float().div_(255.0)
            elif isinstance(img, torch.Tensor):
                # Assume already CHW float or uint8; normalize to float [0,1]
                t = img
                if t.ndim == 3 and t.shape[0] in (1, 3, 4):
                    if t.dtype == torch.uint8:
                        t = t.float().div_(255.0)
                    if t.shape[0] == 4:
                        t = t[:3, ...]
                else:
                    raise ValueError(f"Unexpected tensor image shape: {t.shape}")
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")
            imgs.append(t)

        try:
            images = torch.stack(imgs, dim=0)  # [B, C, H, W]
        except Exception as e:
            raise RuntimeError(f"Failed to stack images: {e}")

        keys = [s["key"] for s in batch]
        metas = [s["meta"] for s in batch]

        return images, keys, metas
