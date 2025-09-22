import logging
from dataclasses import replace
from importlib.resources import as_file, files
from typing import Iterable, List, Union

import cupy as cp  # optional
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.models import mobilenet_v3_small

from wsi_patching.core.pipeline import Stage
from wsi_patching.utils.types import CollatedPatchBatch


class CellVitTissueClassifierFilter(Stage):
    """
    Tissue classifier Stage using a MobileNetV3-small checkpoint:
      - Expects checkpoint with key "model_state_dict"
      - Class 0 is considered 'tissue' (kept); others are filtered out
      - Adds per-patch predictions & probabilities to batch.meta

    Context (optional but supported):
      - ctx['use_gpu']: bool -> prefer CUDA if available
    """

    def __init__(self):
        self.model_train_size = 224

        self.model: Union[nn.Module, None] = None

        # ImageNet normalization (shape [1,3,1,1] to broadcast)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self._ckpt_resource = files("wsi_patching").joinpath("assets/models/cellvit_tissue_detector.pt")

    def validate(self):
        # Use ctx['use_gpu'] if provided by the pipeline, else prefer_gpu
        self.ctx.require_key("use_gpu")
        self.ctx.require_key("tile_size")

        assert torch.cuda.is_available() or not self.ctx["use_gpu"], "No CUDA available, cannot use GPU mode"

        if self.ctx["use_gpu"]:
            self.device: torch.device = torch.device("cuda")
        else:
            self.device: torch.device = torch.device("cpu")

        if self.ctx["tile_size"] < 32:
            logging.warning("The tissue classifier was not trained on very small patches (<32px). Results may be poor.")

    def __call__(self, it: Iterable[CollatedPatchBatch]) -> Iterable[CollatedPatchBatch]:
        self._lazy_load()

        for collated_patch_batch in it:
            patches = collated_patch_batch.patches  # (N,H,W,C) uint8, np or cp

            # Torch only accepts NumPy for from_numpy; move to host if needed
            if isinstance(patches, cp.ndarray):
                patches_np = cp.asnumpy(patches)
            else:
                patches_np = patches

            # Forward pass in chunks
            preds_list: List[torch.Tensor] = []
            probs_list: List[torch.Tensor] = []

            with torch.inference_mode():
                t = self._preprocess_batch(patches_np)  # (B,3,S,S) on device
                logits = self.model(t)  # (B,4)
                probs = torch.softmax(logits, dim=1)  # (B,4)
                preds = torch.argmax(probs, dim=1)  # (B,)
                preds_list.append(preds)
                probs_list.append(probs)

            preds_all = torch.cat(preds_list, dim=0)  # (N,)

            # Class 0 == tissue -> keep
            keep_mask_np = preds_all.detach().cpu().numpy() == 0  # boolean (N,)

            # Filter coords & patches while preserving backend type
            new_coords = [c for c, m in zip(collated_patch_batch.coords, keep_mask_np) if m]

            if self.ctx["use_gpu"]:
                keep_mask_backend = cp.asarray(keep_mask_np)
                new_patches = patches[keep_mask_backend]
            else:
                new_patches = patches[keep_mask_np]

            patch_batch = replace(collated_patch_batch, coords=new_coords, patches=new_patches)

            logging.info(
                f"Yield CellVitTissueClassifier: wsi={patch_batch.wsi_id} in={len(patches_np)} kept={len(new_patches)}"
            )
            yield patch_batch

    def _preprocess_batch(self, batch_imgs: np.ndarray) -> torch.Tensor:
        """
        batch_imgs: uint8 (B,H,W,C) NumPy
        returns: float32 (B,3,S,S) Torch tensor on self.device
        """
        t = torch.as_tensor(batch_imgs).permute(0, 3, 1, 2).float() / 255.0  # (B,3,H,W) on CPU
        t = t.to(self.device, non_blocking=True)
        t = (t - self.mean) / self.std
        return t

    def _lazy_load(self):
        if self.model is not None:
            return

        self.device = torch.device("cuda") if (self.ctx["use_gpu"]) else torch.device("cpu")

        model = mobilenet_v3_small(weights=None)

        # Replace classifier head to 4 classes
        if isinstance(model.classifier[-1], nn.Linear):
            in_features = model.classifier[-1].in_features
        else:
            # MobileNetV3 small's last layer is Linear(1024, 1000) by default
            in_features = 1024
        model.classifier[-1] = nn.Linear(in_features, 4)

        # Get a real filesystem path (works even if package is in a zip)
        with as_file(self._ckpt_resource) as ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location=self.device)

        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state, strict=True)
        self.model = model.to(self.device).eval()

        # Move normalization buffers
        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)

        logging.info(f"Loaded checkpoint to {self.device}")
