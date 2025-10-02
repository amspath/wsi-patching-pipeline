from unittest.mock import call, patch

from wsi_patching.core.pipeline import PipelineContext
from wsi_patching.core.wsi_grid import WSIGrid


def test_export_context_sets_expected_keys():
    grid = WSIGrid(slides=["/data/A.svs"], tile_size=256, stride=128, use_gpu=True, level=2)
    ctx = PipelineContext({})
    grid.export_context(ctx)
    assert ctx["tile_size"] == 256
    assert ctx["stride"] == 128
    assert ctx["level"] == 2
    assert ctx["use_gpu"] is True


@patch("wsi_patching.core.wsi_grid.get_torch_device", new=lambda use_gpu: None)
@patch("wsi_patching.core.wsi_grid.validate_xp_backend")
@patch("wsi_patching.core.wsi_grid.validate_slide_backend")
def test_validate_invokes_backends(mock_validate_slide, mock_validate_xp):
    grid = WSIGrid(slides=[], tile_size=256, stride=128, use_gpu=False, level=0)
    grid.validate()
    mock_validate_slide.assert_called_once_with(False)
    mock_validate_xp.assert_called_once_with(False)

    # Also check use_gpu=True path
    grid2 = WSIGrid(slides=[], tile_size=256, stride=128, use_gpu=True, level=0)
    grid2.validate()
    assert mock_validate_slide.call_args_list[-1] == call(True)
    assert mock_validate_xp.call_args_list[-1] == call(True)


def test_for_slide_returns_single_slide_clone():
    grid = WSIGrid(slides=["/data/A.svs", "/data/B.svs"], tile_size=512, stride=256, use_gpu=True, level=1)
    g2 = grid.for_slide("/data/C.svs")
    assert isinstance(g2, WSIGrid)
    assert g2.slides == ["/data/C.svs"]
    assert g2.tile_size == 512
    assert g2.stride == 256
    assert g2.use_gpu is True
    assert g2.level == 1


@patch("wsi_patching.core.wsi_grid.Path.exists", return_value=True)
@patch("wsi_patching.core.wsi_grid.get_dimensions_for_level")
def test_call_yields_slide_objects_with_dims(mock_dims, mock_exists):
    # Mock per-path dimensions
    def _dims(path, level, use_gpu):
        if path.endswith("A.svs"):
            return (1000, 800)
        return (640, 480)

    mock_dims.side_effect = _dims

    grid = WSIGrid(slides=["/slides/A.svs", "/slides/B.svs"], tile_size=256, stride=128, use_gpu=False, level=0)

    out = list(grid(iter(())))
    assert len(out) == 2

    s0, s1 = out
    # basic attribute checks
    assert s0.wsi_id == "A"
    assert s0.wsi_path.endswith("/slides/A.svs")
    assert s0.dims == (1000, 800)
    assert isinstance(s0.meta, dict)

    assert s1.wsi_id == "B"
    assert s1.wsi_path.endswith("/slides/B.svs")
    assert s1.dims == (640, 480)
    assert isinstance(s1.meta, dict)

    # get_dimensions_for_level called with our level/use_gpu
    mock_dims.assert_has_calls([call("/slides/A.svs", 0, False), call("/slides/B.svs", 0, False)])
