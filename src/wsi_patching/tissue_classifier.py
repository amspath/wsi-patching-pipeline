import logging
from typing import Iterable, List

import numpy as np
import torch

from wsi_patching.core.pipeline import Stage
from wsi_patching.utils.types import CollatedPatchBatch, PatchSample


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

    def __init__(self):
        pass

    def validate(self):
        self.ctx.require_key("use_gpu")

    def __call__(self, it: Iterable[CollatedPatchBatch]) -> Iterable[CollatedPatchBatch]:
        for collated_patch_batch in it:
            patches = collated_patch_batch.patches

            # Convert to tensor (B,H,W,C) -> normalize to [0,1]
            if self.ctx["use_gpu"]:
                ten = torch.as_tensor(patches, device="cuda").float() / 255.0  # B,H,W,C
                ten = ten.cuda(non_blocking=True)
            else:
                ten = torch.from_numpy(patches).float() / 255.0  # B,H,W,C

            ten = ten.permute(0, 3, 1, 2)  # B,C,H,W

            # Simple "score": mean over (C,H,W)
            scores = ten.mean(dim=(1, 2, 3)).detach().cpu().numpy()

            # Filter patches that have a score under 0.5
            filtered_coords_and_patches = [
                (coord, patch)
                for coord, patch, score in zip(collated_patch_batch.coords, patches, scores)
                if score > 0.5
            ]

            # Create filtered output patch batch.
            patch_batch: CollatedPatchBatch = CollatedPatchBatch(
                wsi_id=collated_patch_batch.wsi_id,
                patches=[patch for _, patch in filtered_coords_and_patches],
                coords=[coord for coord, _ in filtered_coords_and_patches],
                meta=collated_patch_batch.meta,
            )

            logging.info(f"Yielding batch from wsi: {patch_batch.wsi_id} size: {len(patch_batch.patches)}")

            yield patch_batch
