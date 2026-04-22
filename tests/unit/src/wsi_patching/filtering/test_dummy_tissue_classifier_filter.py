from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wsi_patching.core.types.types import CollatedPatchBatch
from wsi_patching.filtering.dummy_tissue_classifier_filter import DummyTissueClassifierFilter
from wsi_patching.utils.meta_typing import PipelineContext


def _mk_uniform_sample(val_uint8: int, h=2, w=3, c=3):
    patch = np.full((1, h, w, c), val_uint8, dtype=np.uint8)  # B=1
    coord = [(0, 0)]
    return patch, coord


def _mk_mix_batch():
    p255, c255 = _mk_uniform_sample(255)
    p128, c128 = _mk_uniform_sample(128)
    p127, c127 = _mk_uniform_sample(127)
    p000, c000 = _mk_uniform_sample(0)
    patches = np.concatenate([p255, p128, p127, p000], axis=0)  # BHWC
    coords = np.asarray(c255 + c128 + c127 + c000)
    return patches, coords


@patch("wsi_patching.filtering.dummy_tissue_classifier_filter.get_torch_device")
def test_validate_requires_use_gpu_key(mock_get_dev):
    mock_get_dev.return_value = torch.device("cpu")

    f = DummyTissueClassifierFilter()

    # Missing key -> KeyError; device resolver must NOT be called
    f.attach_context(PipelineContext({}))
    with pytest.raises(KeyError):
        f.validate()
    mock_get_dev.assert_not_called()

    # Present key -> no error; device resolver IS called with False
    f.attach_context(PipelineContext({"use_gpu": False}))
    f.validate()
    mock_get_dev.assert_called_once_with(False)


@patch(
    "wsi_patching.filtering.dummy_tissue_classifier_filter.get_torch_device", new=lambda use_gpu: torch.device("cpu")
)
def test_filter_basic_cpu_keeps_and_drops_correct_items():
    patches, coords = _mk_mix_batch()
    batch = CollatedPatchBatch(wsi_id="S1", patches=patches, coords=coords, use_gpu=False)

    f = DummyTissueClassifierFilter()
    f.attach_context(PipelineContext({"use_gpu": False}))
    f.validate()

    out_batches = list(f(iter([batch])))
    assert len(out_batches) == 1
    out = out_batches[0]

    assert out.patches.shape[0] == 2
    assert len(out.coords) == 2
    assert out.wsi_id == "S1"
    assert np.all(out.patches[0] == 255)
    assert np.all(out.patches[1] == 128)


@patch(
    "wsi_patching.filtering.dummy_tissue_classifier_filter.get_torch_device", new=lambda use_gpu: torch.device("cpu")
)
def test_filter_empty_iterable_produces_no_output():
    f = DummyTissueClassifierFilter()
    f.attach_context(PipelineContext({"use_gpu": False}))
    f.validate()
    assert list(f(iter([]))) == []
