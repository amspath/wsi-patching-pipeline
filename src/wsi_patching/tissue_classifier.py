import logging
from typing import Iterable, List

import numpy as np
import torch

from wsi_patching.core.pipeline import Sample, Stage


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

    def __call__(self, it: Iterable[Sample]) -> Iterable[Sample]:
        for item in it:
            batch: List[Sample] = item["batch"]
            patches = [s["patch"] for s in batch if s.get("patch") is not None]

            # Convert to tensor (B,H,W,C) -> normalize to [0,1]
            arr = np.stack(patches, axis=0)  # uint8
            ten = torch.from_numpy(arr).float() / 255.0  # B,H,W,C
            ten = ten.permute(0, 3, 1, 2)  # B,C,H,W
            if self.device == "cuda":
                ten = ten.cuda(non_blocking=True)
            # Simple "score": mean over (C,H,W)
            scores = ten.mean(dim=(1, 2, 3)).detach().cpu().numpy()
            for s, sc in zip(batch, scores):
                s["tissue_score"] = float(sc)
                s["is_tissue"] = bool(sc > 0.5)

            logging.info(f"Yielding batch from wsi: {batch[0]['wsi_id']} size: {len(batch)}")

            yield {"batch": batch}
