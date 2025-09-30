from typing import Iterable

import torch

from wsi_patching.backends.torch_device import get_torch_device
from wsi_patching.core.pipeline import Stage
from wsi_patching.utils.types import CollatedPatchBatch


class DummyTissueClassifierFilter(Stage):
    """
    Simulates a batched GPU op. For each batch:
      - Convert to tensor (if torch available)
      - Compute a trivial "tissue score" (mean intensity)
      - Attach score & binary label; return the same batch structure
    """

    def __init__(self):
        pass

    def validate(self):
        self.ctx.require_key("use_gpu")

        self._device = get_torch_device(self.ctx["use_gpu"])

    def __call__(self, it: Iterable[CollatedPatchBatch]) -> Iterable[CollatedPatchBatch]:
        for collated_patch_batch in it:
            patches = collated_patch_batch.patches

            # Convert to tensor (B,H,W,C) -> normalize to [0,1]
            ten = torch.as_tensor(patches, device=self._device).float() / 255.0  # B,H,W,C
            ten = ten.permute(0, 3, 1, 2)  # B,C,H,W

            if self.ctx["use_gpu"]:
                ten = ten.cuda(non_blocking=True)

            # Simple "score": mean over (C,H,W)
            scores = ten.mean(dim=(1, 2, 3)).detach().cpu().numpy()

            # Create filter mask
            mask = scores > 0.5

            collated_patch_batch.add_col("dummy_tissue_classifier_score", scores)

            # Filter patch batch with mask
            collated_patch_batch.filter(mask, use_gpu=self.ctx["use_gpu"])

            self.log.info(
                f"Yielding batch from wsi: {collated_patch_batch.wsi_id} size: {len(collated_patch_batch.patches)}"
            )

            yield collated_patch_batch
