# wsi-patching

High-performance streaming of whole-slide image (WSI) patches for digital pathology. Ideal for building training dataset pipelines or for streaming patches during inference without overwhelming memory.

## What you get

- Streaming, regionized tiling of WSIs
- Per-slide producer threads feeding a bounded queue
- Single writer for continuous output (WebDataset shards, NumPy arrays, PyTorch tensors)
- Optional batched GPU steps (filtering, stain normalization)
- Built-in per-stage profiling with per-slide breakdowns
- Extensible: swap in your own ROI logic, classifiers, encoders, or sinks

## Install

```bash
pip install "wsi-patching @ git+https://github.com/amspath/wsi-patching-pipeline.git"
```

## Minimal example

```python
from wsi_patching import WSIGrid, PatchExtractor, NumpyStreamWriter
from wsi_patching.regions_of_interest import AttachROIs, WholeSlideProvider

slides = ["slide_a.tiff", "slide_b.tiff"]

p = (
    WSIGrid(slides=slides, resolution=0, unit="level")
    .then(PatchExtractor(tile_size=256, stride=256))
    .to(NumpyStreamWriter(layout="NCHW"))
)

for wsi_id, images, coords, meta in p.stream(num_workers=4):
    print(wsi_id, images.shape)  # e.g. (N, 3, 256, 256)
```

## Next steps

- [Installation](installation.md) — CPU, GPU, and development setups
- [Quickstart](quickstart.md) — two complete runnable examples
- [Concepts](concepts.md) — threading model, context, resolution options
