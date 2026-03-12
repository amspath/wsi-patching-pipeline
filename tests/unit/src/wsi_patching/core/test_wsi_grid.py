from unittest.mock import call, patch

import pytest

from wsi_patching.core.pipeline import PipelineContext
from wsi_patching.core.wsi_grid import WSIGrid


def test_export_context_sets_expected_keys():
    grid = WSIGrid(slides=["/data/A.svs"], use_gpu=True, resolution=0.5, unit="mpp")
    ctx = PipelineContext({})
    grid.export_context(ctx)

    assert ctx["resolution"] == 0.5
    assert ctx["unit"] == "mpp"
    assert ctx["use_gpu"] is True
    # fallback_mode is not exported by WSIGrid; it is owned by RegionReadAndBatch.
    assert "fallback_mode" not in ctx


@patch("wsi_patching.core.wsi_grid.get_torch_device", new=lambda use_gpu: None)
@patch("wsi_patching.core.wsi_grid.validate_xp_backend")
@patch("wsi_patching.core.wsi_grid.validate_slide_backend")
def test_validate_invokes_backends(mock_validate_slide, mock_validate_xp):
    grid = WSIGrid(slides=[], use_gpu=False, resolution=0.5, unit="mpp")
    grid.attach_context(PipelineContext({}))
    grid.validate()
    mock_validate_slide.assert_called_once_with(False)
    mock_validate_xp.assert_called_once_with(False)

    # Also check use_gpu=True path
    grid2 = WSIGrid(slides=[], use_gpu=True, resolution=0.5, unit="mpp")
    grid2.attach_context(PipelineContext({}))
    grid2.validate()
    assert mock_validate_slide.call_args_list[-1] == call(True)
    assert mock_validate_xp.call_args_list[-1] == call(True)


def test_validate_does_not_require_fallback_mode_in_context():
    """WSIGrid.validate() must not require fallback_mode; it is owned by RegionReadAndBatch."""
    with (
        patch("wsi_patching.core.wsi_grid.get_torch_device", new=lambda use_gpu: None),
        patch("wsi_patching.core.wsi_grid.validate_xp_backend"),
        patch("wsi_patching.core.wsi_grid.validate_slide_backend"),
    ):
        grid = WSIGrid(slides=[], use_gpu=False, resolution=0.5, unit="mpp")
        grid.attach_context(PipelineContext({}))
        # Should not raise even though fallback_mode is absent from context.
        grid.validate()


def test_for_slide_returns_single_slide_clone():
    grid = WSIGrid(
        slides=["/data/A.svs", "/data/B.svs"], use_gpu=True, resolution=0.5, unit="mpp"
    )

    g2 = grid.for_slide("/data/C.svs")
    assert isinstance(g2, WSIGrid)
    assert g2.slides == ["/data/C.svs"]
    assert g2.use_gpu is True
    assert g2.resolution == 0.5
    assert g2.unit == "mpp"


@patch("wsi_patching.core.wsi_grid.Path.exists", return_value=True)
@patch("wsi_patching.core.wsi_grid.get_virtual_slide_dims")
def test_call_yields_slide_objects_with_dims(mock_virtual_dims, mock_exists):
    """WSIGrid.__call__ should yield Slide objects with virtual dims from get_virtual_slide_dims."""

    def _virtual_dims(path, resolution, unit):
        if path.endswith("A.svs"):
            return (1000, 800, 2.0)
        return (640, 480, 4.0)

    mock_virtual_dims.side_effect = _virtual_dims

    grid = WSIGrid(
        slides=["/slides/A.svs", "/slides/B.svs"], use_gpu=False, resolution=0.5, unit="mpp"
    )
    grid.attach_context(PipelineContext({}))

    out = list(grid(iter(())))
    assert len(out) == 2

    s0, s1 = out
    # WSIGrid always sets level=0 (virtual level), resample_factor=1.0
    assert s0.wsi_id == "A"
    assert s0.wsi_path.endswith("/slides/A.svs")
    assert s0.dims == (1000, 800)
    assert s0.level == 0
    assert s0.downsample == 2.0
    assert s0.resample_factor == pytest.approx(1.0)
    assert isinstance(s0.meta, dict)
    assert s0.meta["slide.wsi_id"] == "A"
    assert s0.meta["slide.requested_resolution"] == 0.5
    assert s0.meta["slide.requested_unit"] == "mpp"
    assert s0.meta["slide.path"].endswith("/slides/A.svs")
    # fallback_mode and selected_level are no longer in WSIGrid's meta
    assert "slide.requested_fallback_mode" not in s0.meta
    assert "slide.selected_level" not in s0.meta

    assert s1.wsi_id == "B"
    assert s1.wsi_path.endswith("/slides/B.svs")
    assert s1.dims == (640, 480)
    assert s1.level == 0
    assert s1.downsample == 4.0
    assert s1.resample_factor == pytest.approx(1.0)

    # get_virtual_slide_dims called once per slide with correct args
    mock_virtual_dims.assert_has_calls(
        [call("/slides/A.svs", 0.5, "mpp"), call("/slides/B.svs", 0.5, "mpp")]
    )


@patch("wsi_patching.core.wsi_grid.Path.exists", return_value=True)
@patch("wsi_patching.core.wsi_grid.get_virtual_slide_dims")
def test_call_always_sets_resample_factor_1_and_level_0(mock_virtual_dims, mock_exists):
    """WSIGrid always sets resample_factor=1.0 and level=0 regardless of resolution/unit."""
    mock_virtual_dims.return_value = (2000, 1500, 2.0)

    grid = WSIGrid(slides=["/slides/A.svs"], use_gpu=False, resolution=0.5, unit="mpp")
    grid.attach_context(PipelineContext({}))
    out = list(grid(iter(())))
    assert len(out) == 1
    slide = out[0]

    assert slide.level == 0
    assert slide.resample_factor == pytest.approx(1.0)
    assert slide.dims == (2000, 1500)
    assert slide.downsample == pytest.approx(2.0)

