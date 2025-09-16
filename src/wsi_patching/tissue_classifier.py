import logging
from typing import Iterable, List

import numpy as np
import torch

from wsi_patching.core.pipeline import Stage
from wsi_patching.utils.types import PatchBatch, PatchSample


class DummyTissueClassifier(Stage):
    """
    Simulates a batched GPU op. For each batch:
      - Convert to tensor (if torch available)
      - Compute a trivial "tissue score" (mean intensity)
      - Attach score & binary label; return the same batch structure

    device:
      - "cuda" to prefer GPU if available (default)
      - "cpu" to force CPU path
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    def __call__(self, it: Iterable[PatchBatch]) -> Iterable[PatchBatch]:
        for item in it:
            batch: List[PatchSample] = item.samples
            patches = [s.patch for s in batch if s.patch is not None]

            # Convert to tensor (B,H,W,C) -> normalize to [0,1]
            arr = np.stack(patches, axis=0)  # uint8
            ten = torch.from_numpy(arr).float() / 255.0  # B,H,W,C
            ten = ten.permute(0, 3, 1, 2)  # B,C,H,W
            if self.device == "cuda":
                ten = ten.cuda(non_blocking=True)
            # Simple "score": mean over (C,H,W)
            scores = ten.mean(dim=(1, 2, 3)).detach().cpu().numpy()

            # Filter patches that have a score under 0.5
            patch_batch = PatchBatch(samples=[sample for sample, score in zip(batch, scores) if score > 0.5])

            logging.info(f"Yielding batch from wsi: {patch_batch.samples[0].wsi_id} size: {len(patch_batch.samples)}")

            yield patch_batch
