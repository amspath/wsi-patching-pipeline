from unittest.mock import call, patch

import pytest

from wsi_patching.core.pipeline import PipelineContext
from wsi_patching.core.wsi_grid import WSIGrid


def test_export_context_sets_expected_keys():
    grid = WSIGrid(slides=["/data/A.svs"], use_gpu=True, resolution=0.5, unit="mpp", fallback_mode="nearest")
    ctx = PipelineContext({})
    grid.export_context(ctx)

    assert ctx["resolution"] == 0.5
    assert ctx["unit"] == "mpp"
    assert ctx["fallback_mode"] == "nearest"
    assert ctx["use_gpu"] is True


@patch("wsi_patching.core.wsi_grid.get_torch_device", new=lambda use_gpu: None)
@patch("wsi_patching.core.wsi_grid.validate_xp_backend")
@patch("wsi_patching.core.wsi_grid.validate_slide_backend")
def test_validate_invokes_backends(mock_validate_slide, mock_validate_xp):
    grid = WSIGrid(slides=[], use_gpu=False, resolution=0.5, unit="mpp")
    grid.validate()
    mock_validate_slide.assert_called_once_with(False)
    mock_validate_xp.assert_called_once_with(False)

    # Also check use_gpu=True path
    grid2 = WSIGrid(slides=[], use_gpu=True, resolution=0.5, unit="mpp")
    grid2.validate()
    assert mock_validate_slide.call_args_list[-1] == call(True)
    assert mock_validate_xp.call_args_list[-1] == call(True)


def test_for_slide_returns_single_slide_clone():
    grid = WSIGrid(
        slides=["/data/A.svs", "/data/B.svs"], use_gpu=True, resolution=0.5, unit="mpp", fallback_mode="nearest"
    )

    g2 = grid.for_slide("/data/C.svs")
    assert isinstance(g2, WSIGrid)
    assert g2.slides == ["/data/C.svs"]
    assert g2.use_gpu is True
    assert g2.resolution == 0.5
    assert g2.unit == "mpp"
    assert g2.fallback_mode == "nearest"


@patch("wsi_patching.core.wsi_grid.Path.exists", return_value=True)
@patch("wsi_patching.core.wsi_grid.get_level_downsamples")
@patch("wsi_patching.core.wsi_grid.get_dimensions_for_level")
@patch("wsi_patching.core.wsi_grid.get_level_for_resolution")
def test_call_yields_slide_objects_with_dims(mock_get_level, mock_dims, mock_downsamples, mock_exists):
    # Mock per-path selected levels
    def _get_level(path, resolution, unit, fallback_mode):
        if path.endswith("A.svs"):
            return 1
        return 2

    mock_get_level.side_effect = _get_level

    # Mock per-path dimensions (note: new signature is (path, level))
    def _dims(path, level):
        if path.endswith("A.svs"):
            assert level == 1
            return (1000, 800)
        assert level == 2
        return (640, 480)

    mock_dims.side_effect = _dims

    # Mock downsample factors: level 0=1.0, level 1=2.0, level 2=4.0
    mock_downsamples.return_value = [1.0, 2.0, 4.0]

    grid = WSIGrid(
        slides=["/slides/A.svs", "/slides/B.svs"], use_gpu=False, resolution=0.5, unit="mpp", fallback_mode="nearest"
    )

    out = list(grid(iter(())))
    assert len(out) == 2

    s0, s1 = out
    # basic attribute checks
    assert s0.wsi_id == "A"
    assert s0.wsi_path.endswith("/slides/A.svs")
    assert s0.dims == (1000, 800)
    assert s0.level == 1
    assert s0.downsample == 2.0
    assert isinstance(s0.meta, dict)
    assert s0.meta["slide.wsi_id"] == "A"
    assert s0.meta["slide.requested_resolution"] == 0.5
    assert s0.meta["slide.requested_unit"] == "mpp"
    assert s0.meta["slide.requested_fallback_mode"] == "nearest"
    assert s0.meta["slide.selected_level"] == 1
    assert s0.meta["slide.path"].endswith("/slides/A.svs")

    assert s1.wsi_id == "B"
    assert s1.wsi_path.endswith("/slides/B.svs")
    assert s1.dims == (640, 480)
    assert s1.level == 2
    assert s1.downsample == 4.0
    assert isinstance(s1.meta, dict)
    assert s1.meta["slide.selected_level"] == 2

    # get_level_for_resolution called with our resolution/unit/fallback_mode
    mock_get_level.assert_has_calls(
        [call("/slides/A.svs", 0.5, "mpp", "nearest"), call("/slides/B.svs", 0.5, "mpp", "nearest")]
    )

    # get_dimensions_for_level called with (path, level)
    mock_dims.assert_has_calls([call("/slides/A.svs", 1), call("/slides/B.svs", 2)])


@patch("wsi_patching.core.wsi_grid.Path.exists", return_value=True)
@patch("wsi_patching.core.wsi_grid.get_resample_factor")
@patch("wsi_patching.core.wsi_grid.get_level_downsamples")
@patch("wsi_patching.core.wsi_grid.get_dimensions_for_level")
@patch("wsi_patching.core.wsi_grid.get_level_for_resolution")
def test_call_resample_fallback_computes_virtual_dims(
    mock_get_level, mock_dims, mock_downsamples, mock_resample_factor, mock_exists
):
    """When fallback_mode='resample', WSIGrid should compute virtual dims and set resample_factor."""
    mock_get_level.return_value = 0  # read from level 0 (0.25 mpp)
    mock_dims.return_value = (4000, 3000)  # level-0 actual dims
    mock_downsamples.return_value = [1.0, 4.0]  # level 0 downsample=1.0
    mock_resample_factor.return_value = 2.0  # requested 0.5 / actual 0.25 = 2x

    grid = WSIGrid(slides=["/slides/A.svs"], use_gpu=False, resolution=0.5, unit="mpp", fallback_mode="resample")
    out = list(grid(iter(())))
    assert len(out) == 1
    slide = out[0]

    # Virtual dims should be actual_dims / resample_factor
    assert slide.dims == (2000, 1500)
    # Virtual downsample: level-0 downsample (1.0) * resample_factor (2.0)
    assert slide.downsample == pytest.approx(2.0)
    assert slide.resample_factor == pytest.approx(2.0)
    assert slide.level == 0
    assert slide.meta["slide.resample_factor"] == pytest.approx(2.0)

    mock_resample_factor.assert_called_once_with("/slides/A.svs", 0, 0.5, "mpp")


@patch("wsi_patching.core.wsi_grid.Path.exists", return_value=True)
@patch("wsi_patching.core.wsi_grid.get_level_downsamples")
@patch("wsi_patching.core.wsi_grid.get_dimensions_for_level")
@patch("wsi_patching.core.wsi_grid.get_level_for_resolution")
def test_call_no_resample_factor_1_by_default(mock_get_level, mock_dims, mock_downsamples, mock_exists):
    """Non-resample fallback modes should produce resample_factor=1.0 (no resampling)."""
    mock_get_level.return_value = 1
    mock_dims.return_value = (1000, 800)
    mock_downsamples.return_value = [1.0, 2.0]

    grid = WSIGrid(slides=["/slides/A.svs"], use_gpu=False, resolution=0.5, unit="mpp", fallback_mode="nearest")
    out = list(grid(iter(())))
    assert len(out) == 1
    slide = out[0]

    assert slide.resample_factor == pytest.approx(1.0)
    assert slide.dims == (1000, 800)  # unchanged
