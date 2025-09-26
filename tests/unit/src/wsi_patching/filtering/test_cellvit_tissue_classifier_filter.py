from dataclasses import dataclass
from typing import List
from unittest.mock import patch

import numpy as np
import pytest
import torch

from wsi_patching.filtering.cellvit_tissue_classifier_filter import CellVitTissueClassifierFilter
from wsi_patching.utils.meta_typing import PipelineContext
from wsi_patching.utils.types import CollatedPatchBatch


def _mk_batch(brights: List[int], h=4, w=5, c=3):
    """
    Build a BHWC uint8 batch where each item is a constant intensity in [0..255].
    """
    arrs = [np.full((1, h, w, c), v, dtype=np.uint8) for v in brights]
    patches = np.concatenate(arrs, axis=0)  # [B,H,W,C]
    coords = np.array([[i, i] for i in range(len(brights))])
    return patches, coords


class DummyThreshModel(torch.nn.Module):
    """
    A tiny model that returns logits with class 0 (tissue) if mean(x)>thresh,
    else class 1. This lets us drive keep/drop behavior deterministically.
    """

    def __init__(self, thresh=0.5):
        super().__init__()
        self.thresh = thresh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, S, S], values assumed roughly in [0,1] after preprocessing
        m = x.mean(dim=(1, 2, 3))  # [B]
        keep = (m > self.thresh).to(x.dtype)
        # logits: 4 classes, put high score on cls-0 if keep else cls-1
        B = x.shape[0]
        logits = torch.zeros((B, 4), dtype=x.dtype, device=x.device)
        logits[torch.arange(B), keep.long() * 1] = -1.0  # lower for non-selected
        logits[torch.arange(B), 0] = keep * 10.0  # strong class-0 when keep
        logits[torch.arange(B), 1] += (1 - keep) * 10.0  # strong class-1 otherwise
        return logits


def _patch_lazy_load_to_dummy(filter_obj: CellVitTissueClassifierFilter, device: torch.device, thresh=0.5):
    """
    Replace _lazy_load on the instance so it simply sets a dummy model and moves mean/std.
    """

    def _lazy():
        filter_obj.model = DummyThreshModel(thresh=thresh).to(device).eval()
        filter_obj.mean = filter_obj.mean.to(device)
        filter_obj.std = filter_obj.std.to(device)

    filter_obj._lazy_load = _lazy  # type: ignore[attr-defined]


# -------- tests --------
@patch(
    "wsi_patching.filtering.cellvit_tissue_classifier_filter.get_torch_device", new=lambda use_gpu: torch.device("cpu")
)
def test_validate_requires_keys():
    f = CellVitTissueClassifierFilter()
    f.attach_context(PipelineContext({}))
    with pytest.raises(KeyError):
        f.validate()


@patch(
    "wsi_patching.filtering.cellvit_tissue_classifier_filter.get_torch_device", new=lambda use_gpu: torch.device("cpu")
)
def test_filter_keeps_only_class0_on_cpu():
    # Build batch: bright -> keep, dim -> drop (according to DummyThreshModel threshold 0.5)
    patches, coords = _mk_batch([255, 10, 200, 0])  # [B=4]
    batch = CollatedPatchBatch(wsi_id="WSI1", patches=patches, coords=coords, meta_cols={})

    f = CellVitTissueClassifierFilter()
    f.attach_context(PipelineContext({"use_gpu": False, "tile_size": 128}))
    f.validate()

    # Make lazy load inject dummy model on CPU
    _patch_lazy_load_to_dummy(f, device=torch.device("cpu"), thresh=0.5)

    out_batches = list(f(iter([batch])))
    assert len(out_batches) == 1
    out = out_batches[0]

    # Expected: keep indices where intensity -> mean/255 > 0.5 -> 255, 200 (drop 10, 0)
    assert out.patches.shape[0] == 2
    assert np.array_equal(out.coords, np.array([[0, 0], [2, 2]]))
    assert out.wsi_id == "WSI1"
    # spot-check values
    assert np.all(out.patches[0] == 255) and np.all(out.patches[1] == 200)
