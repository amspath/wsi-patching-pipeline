import logging
from dataclasses import replace
from typing import Iterable

import torch

from wsi_patching.core.pipeline import Stage
from wsi_patching.utils.types import CollatedPatchBatch


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

        assert torch.cuda.is_available() or not self.ctx["use_gpu"], "No CUDA available, cannot use GPU mode"

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

            # Create filter mask
            mask = scores > 0.5

            # Filter patch batch with mask
            patch_batch = replace(
                collated_patch_batch,
                coords=[c for c, m in zip(collated_patch_batch.coords, mask) if m],
                patches=patches[mask],
            )

            logging.info(f"Yielding batch from wsi: {patch_batch.wsi_id} size: {len(patch_batch.patches)}")

            yield patch_batch
