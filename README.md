[![Unit tests](https://github.com/amspath/wsi-patching-pipeline/actions/workflows/unit_tests.yaml/badge.svg)](https://github.com/amspath/wsi-patching-pipeline/actions/workflows/unit_tests.yaml)

# wsi-patching-pipeline
A pragmatic pipeline for streaming whole-slide image (WSI) patches with region prefetch, per-WSI multiprocessing producers, and a single async WebDataset writer. It’s designed as a runnable skeleton you can extend: swap in your own ROI logic, classifiers, encoders, or sinks.

✨ What you get
- Streaming, regionized tiling of WSIs (cuCIM preferred; Pillow fallback for small images).
- Per-slide producers (multiprocessing) feeding a bounded MP queue.
- Single writer process continuously sharding to WebDataset .tar files.
- Batched GPU steps.
- Built-in isolated stage profiling per slide + aggregated stats.


## 1) Install

Python ≥3.8 is recommended.
```
# in a fresh venv or conda env
pip install -e .        # For cpu only (not all functionality is supported for cpu only)
pip install -e .[gpu]
```

> If you run without gpu, the backend will rely on OpenSlide to open images. Openslide requires system libraries. It is your own responsibility to install these. The easiest way to install those is to create your environment through conda and add the required system libraries in there. 

## 2) Checkout the examples
`demo.py` shows you how to build a basic pipeline for creating a basic WebDataset
```python
p = (
        WSIGrid(slides=slides, tile_size=224, stride=224, level=0, use_gpu=True)
        .then(AttachROIs(providers=[RectROIProvider(rois_dict)]))
        .then(TilePlanner())
        .then(ReadWindowChunker(max_window_size=224 * 40))
        .then(RegionReadAndBatch(batch_size=200, num_workers=8))
        .then(CellVitTissueClassifierFilter())
        .then(PNGEncoder())
        .to(WebDatasetWriter())
)
p.run(cpu_processes=4)
```

`numpy_mem_writer_demo.py` shows you how to build a basic pipeline for patching up a wsi in memory, without the need of writing to disk (RAM heavy for larger datasets, obviously).
```python
p = (
        WSIGrid(slides=slides, tile_size=256, stride=256, level=0, use_gpu=True)
        .then(TilePlanner())
        .then(ReadWindowChunker(max_window_size=8192))
        .then(RegionReadAndBatch(batch_size=800, num_workers=4))
        .to(NumpyMemoryWriter(layout="NCHW"))
)
np_patch_array, np_coords_array, list_of_wsi_ids = p.run(cpu_processes=2)
```

# More will come
- [ ] More readme stuff
- [ ] Tests
- [ ] More basic stuff (basic xml ROI reader, cellvit tissue classifier, etc)
