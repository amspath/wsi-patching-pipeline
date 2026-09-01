# API Reference

All public components are importable from `wsi_patching` unless noted otherwise.

---

## Entry point

### `WSIGrid`

The source stage; every pipeline starts here.

```python
from wsi_patching import WSIGrid
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `slides` | `List[str]` | — | Paths to WSI files |
| `resolution` | `float` | `0.25` | Desired resolution value |
| `unit` | `"level" \| "mpp" \| "downsample"` | `"mpp"` | Unit of `resolution` |
| `use_gpu` | `bool` | `False` | Enable GPU-accelerated backends (CuPy) |
| `fallback_mode` | `"nearest" \| "floor" \| "ceil" \| "error" \| "resample"` | `"nearest"` | What to do when the exact resolution is unavailable |
| `resample_interpolation` | `"nearest" \| "linear" \| "cubic" \| "area" \| "lanczos"` | `"lanczos"` | Interpolation used when `fallback_mode="resample"` |
| `max_relative_deviation` | `float \| None` | `0.01` | Max relative deviation between the request and the selected level for `fallback_mode="nearest"`; `None` disables the check |

---

## Regions of interest

### `AttachROIs`

```python
from wsi_patching import AttachROIs
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `providers` | `List[ROIProvider]` | — | One or more ROI providers; ROIs from all providers are merged |
| `on_empty` | `"error" \| "whole_slide"` | `"error"` | What to do when a slide has no ROIs |

### `RectROIProvider`

```python
from wsi_patching import RectROIProvider
```

| Parameter | Type | Description |
|---|---|---|
| `rois` | `Dict[str, List[Tuple[int, int, int, int]]]` | Maps slide stem → list of `(x, y, width, height)` tuples in pixels at the selected resolution level |

### `RectROIfromXMLProvider`

Reads rectangular ROIs from ASAP-style XML annotation files.

```python
from wsi_patching import RectROIfromXMLProvider
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `rois` | `Dict[str, str]` | — | Maps slide stem → path to XML annotation file |
| `annotation_group` | `str` | `"roi"` | `PartOfGroup` attribute value to filter annotations by |
| `annotation_level` | `Optional[int]` | `None` | Pyramid level the annotation coordinates were drawn at; `None` means same level as the pipeline |

### `WholeSlideProvider`

Produces a single ROI covering the entire slide. Takes no parameters.

```python
from wsi_patching import WholeSlideProvider
```

---

## Core stages

### `PatchExtractor`

Composite stage that combines `TilePlanner` + `ReadWindowChunker` + `RegionReadAndBatch`. Use this in most pipelines.

```python
from wsi_patching import PatchExtractor
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tile_size` | `int` | — | Square patch size in pixels |
| `stride` | `int` | — | Spacing between patch top-left corners |
| `tile_selection_mode` | `"any_overlap" \| "full_inside_bounds" \| "center_in_roi"` | `"any_overlap"` | How to decide if a tile is inside the ROI |
| `max_batch_size` | `int` | `200` | Maximum patches per output batch |
| `wsi_edge_policy` | `"drop" \| "pad_with_zeros" \| "pad_with_edge"` | `"pad_with_zeros"` | How to handle tiles that extend beyond the WSI edge |
| `roi_edge_policy` | `"read_from_image" \| "use_wsi_edge_policy"` | `"use_wsi_edge_policy"` | How to handle tiles at the ROI boundary |
| `dtype` | `np.dtype` | `np.uint8` | Patch array dtype |
| `max_window_size` | `Optional[int]` | `None` | Max read-window size; defaults to `20 × tile_size` |

### Sub-stages (advanced)

If you need to customise individual steps inside `PatchExtractor`, you can chain the three sub-stages yourself:

| Stage | Description |
|---|---|
| `TilePlanner(tile_size, stride, tile_selection_mode)` | Computes tile coordinates for each ROI |
| `ReadWindowChunker(max_window_size=None)` | Groups tiles into efficient read windows |
| `RegionReadAndBatch(batch_size, wsi_edge_policy, roi_edge_policy, dtype)` | Reads pixel data and collates into batches |

---

## Filters

All filters accept a `CollatedPatchBatch` and yield a filtered `CollatedPatchBatch`. They can be inserted with `.then()` after `PatchExtractor`.

### `LowContrastBackgroundFilter`

```python
from wsi_patching import LowContrastBackgroundFilter
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `range_threshold` | `float` | `0.2` | Minimum grayscale dynamic range (max − min, in `[0,1]`) for a patch to be kept |
| `float_precision` | `"float32" \| "float64"` | `"float32"` | Working precision for grayscale conversion |

### `OtsuFilter`

```python
from wsi_patching import OtsuFilter
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tissue_is_darker` | `bool` | `True` | If `True`, pixels darker than the Otsu threshold are classified as tissue |
| `num_bins` | `int` | `256` | Histogram bins for Otsu computation |
| `float_precision` | `"float32" \| "float64"` | `"float32"` | Working precision |
| `min_tissue_fraction` | `float` | `0.0` | Minimum fraction of tissue pixels for a patch to be kept |

### `PenArtifactFilter`

Detects blue, green, and red pen markings using batched GPU-accelerated colour thresholds.

```python
from wsi_patching import PenArtifactFilter
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `diff_thresh` | `float` | `5.0` | Minimum colour channel difference to classify a pixel as pen |
| `max_pen_fraction` | `float` | `0.01` | Maximum fraction of pen pixels before a patch is discarded |

### `CellVitTissueClassifierFilter`

Uses a MobileNetV3 model (from CellViT) to classify patches as tissue or background. **Requires the `[gpu]` extra.**

```python
from wsi_patching import CellVitTissueClassifierFilter
```

| Parameter | Type | Description |
|---|---|---|
| `model_file_path` | `str \| Path` | Path to the `.pth` model weights file |

---

## Transforms

### `MacenkoNormalizer`

Applies Macenko stain normalisation. Fits on the first batch of each slide. **Requires the `[gpu]` extra.**

```python
from wsi_patching import MacenkoNormalizer
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `alpha` | `float` | `1.0` | Percentile for robust SVD |
| `beta` | `float` | `0.15` | Transparency threshold |
| `light_intensity` | `int` | `255` | Maximum transmitted light intensity |
| `pixel_limit` | `Optional[int]` | `500_000` | Max pixels sampled for fitting; `None` uses all pixels |

!!! warning
    If the first batch of a slide contains only background, fitting will produce poor results. Consider placing a background filter before `MacenkoNormalizer`.

---

## Encoders

### `PNGEncoder`

Encodes patches to PNG bytes. Required before `WebDatasetWriter`.

```python
from wsi_patching import PNGEncoder
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `compress_level` | `int` | `1` | PNG compression level (0–9); 1 is fast with decent compression |
| `threads` | `Optional[int]` | `None` | Worker threads for parallel encoding; `None` uses one per CPU |

---

## Stream writers

Stream writers yield results to the caller directly; use with `.stream()`.

### `NumpyStreamWriter`

```python
from wsi_patching import NumpyStreamWriter
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `layout` | `"NCHW" \| "NHWC"` | `"NCHW"` | Output array axis order |
| `dtype` | `np.dtype` | `np.float32` | Output array dtype |

Yields: `(wsi_id: str, images: np.ndarray, coords: np.ndarray, meta: list[dict])`

### `TorchStreamWriter`

```python
from wsi_patching import TorchStreamWriter  # requires [gpu] extra
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `layout` | `"NCHW" \| "NHWC"` | `"NCHW"` | Output tensor axis order |
| `dtype` | `torch.dtype` | `torch.float32` | Output tensor dtype |
| `device` | `Optional[torch.device]` | `None` | Target device; `None` uses the GPU when `use_gpu=True` |

Yields: `(wsi_id: str, images: torch.Tensor, coords: torch.Tensor, meta: list[dict])`

---

## Materialize writers

Materialize writers write to disk; use with `.materialize()`.

### `WebDatasetWriter`

```python
from wsi_patching import WebDatasetWriter
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `outdir` | `Path` | `Path("./output/")` | Directory where shard tar files are written |
| `shard_size` | `int` | `200` | Maximum number of patches per shard file |
| `shuffle_buffer_size` | `Optional[int]` | `None` | Buffer size before a random flush; defaults to `shard_size × 3` |

Shard files are named `shard-000000.tar`, `shard-000001.tar`, etc. Each entry contains keys `png` (PNG bytes), `meta` (JSON), and `__key__` (e.g. `slide_a_1024_2048`).

---

## Pipeline execution

### `.stream()` and `.materialize()`

Both methods share the same parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_workers` | `int` | `4` | Number of concurrent producer threads (one per slide) |
| `writer_prefetch_factor` | `int` | `2` | Queue depth = `writer_prefetch_factor × num_workers` |
| `profile` | `bool` | `False` | Enable per-stage timing profiling |
| `verbosity_level` | `LogLevel` | `"WARNING"` | Logging level (`"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`) |
| `gracefully_handle_producer_errors` | `bool` | `False` | Skip failed slides instead of raising |

### After execution

| Attribute / Method | Description |
|---|---|
| `pipeline.failed_slides` | `List[str]` of slide names that failed (non-empty only when `gracefully_handle_producer_errors=True`) |
| `pipeline.print_profile()` | Print a formatted profile table to stdout |
| `pipeline.get_profile()` | Return the profile as a dict (`{"by_stage": {...}, "by_slide": {...}}`) |
