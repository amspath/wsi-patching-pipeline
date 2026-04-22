# Quickstart

Two complete, runnable examples covering the two main pipeline modes.

!!! tip "Skip ROI setup"
    Use `WholeSlideProvider()` instead of `RectROIProvider` to patch the entire slide with no setup required.

---

## Example 1: Stream patches to NumPy

Best for inference pipelines where you process patches in memory without writing to disk.

```python
from pathlib import Path
from wsi_patching import AttachROIs, NumpyStreamWriter, PatchExtractor, RectROIProvider, WSIGrid

slides = [
    "./data/slide_a.tiff",
    "./data/slide_b.tiff",
]

# Map slide stem → list of (x, y, width, height) ROI tuples (pixel coords at selected resolution)
rois_dict = {Path(s).stem: [(0, 0, 18000, 10000)] for s in slides}

p = (
    WSIGrid(slides=slides, resolution=0, unit="level")
    .then(AttachROIs(providers=[RectROIProvider(rois_dict)]))
    .then(PatchExtractor(tile_size=256, stride=256))
    .to(NumpyStreamWriter(layout="NCHW"))
)

for wsi_id, images, coords, meta in p.stream(num_workers=4):
    # images: np.ndarray of shape (N, 3, 256, 256), dtype float32
    # coords: np.ndarray of shape (N, 2), pixel (x, y) top-left of each patch
    # meta:   list of dicts, one per patch, with slide metadata
    print(f"{wsi_id}: {images.shape}")
```

**Notes:**

- `resolution=0, unit="level"` selects native level 0 (full resolution). See [Concepts — Resolution](concepts.md#resolution-and-fallback-modes) for other options.
- Batches are **not ordered per slide** when `num_workers > 1`. Use `wsi_id` to track which slide each batch belongs to, or set `num_workers=1` for ordered output.
- `layout="NCHW"` transposes from the native `NHWC` read order. Use `layout="NHWC"` to skip the transpose.
- `NumpyStreamWriter` also accepts a `dtype` parameter (default `np.float32`).

---

## Example 2: Materialize to WebDataset

Best for creating large training datasets on disk as shuffled WebDataset shards.

```python
from pathlib import Path
from wsi_patching import AttachROIs, PatchExtractor, PNGEncoder, RectROIProvider, WebDatasetWriter, WSIGrid

slides = [
    "./data/slide_a.tiff",
    "./data/slide_b.tiff",
]

rois_dict = {Path(s).stem: [(0, 0, 18000, 10000)] for s in slides}

p = (
    WSIGrid(slides=slides, resolution=0, unit="level")
    .then(AttachROIs(providers=[RectROIProvider(rois_dict)]))
    .then(PatchExtractor(tile_size=224, stride=224, max_batch_size=200))
    .then(PNGEncoder())
    .to(WebDatasetWriter(outdir=Path("./output/"), shard_size=300, shuffle_buffer_size=500))
)

p.materialize(num_workers=4, profile=True)
p.print_profile()
```

**Notes:**

- `PNGEncoder` is **required** before `WebDatasetWriter`. The pipeline checks this at construction time and raises a `TypeError` immediately if the types do not match.
- Output shards are written to `./output/shard-000000.tar`, `shard-000001.tar`, etc.
- Each shard entry has keys `__key__` (e.g. `slide_a_1024_2048`), `png` (PNG bytes), and `meta` (JSON-encoded metadata dict).
- `shuffle_buffer_size` controls how many patches accumulate before a random flush to disk. Larger values improve shuffle quality at the cost of memory.
- After `materialize()`, `p.failed_slides` contains the names of any slides that were skipped due to errors (empty list if all succeeded).

**Sample profile output:**

```
=== Pipeline Profile (isolated timings only) ===
Stage                                Yields         Wall (s)   Avg (ms/yield)
PNGEncoder.isolated                    640            1.440s          2.412ms

--- Per slide breakdown ---
[slide_a]
  PNGEncoder.isolated          yields=  320    wall=  0.762s    avg=  2.382ms
[slide_b]
  PNGEncoder.isolated          yields=  320    wall=  0.778s    avg=  2.432ms
```
