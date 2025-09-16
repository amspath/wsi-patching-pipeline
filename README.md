# wsi-patching-pipeline
A pragmatic pipeline for streaming whole-slide image (WSI) patches with region prefetch, per-WSI multiprocessing producers, and a single async WebDataset writer. It’s designed as a runnable skeleton you can extend: swap in your own ROI logic, classifiers, encoders, or sinks.

✨ What you get
- Streaming, regionized tiling of WSIs (cuCIM preferred; Pillow fallback for small images).
- Per-slide producers (multiprocessing) feeding a bounded MP queue.
- Single writer process continuously sharding to WebDataset .tar files.
- (Optional) batched GPU step via a dummy “tissue score” classifier.
- Built-in isolated stage profiling per slide + aggregated stats.


## 1) Install

Python ≥3.9 is recommended. Linux is strongly recommended for cuCIM.
```
# in a fresh venv or conda env
pip install -e .
```

## 2) Checkout the example.py
It shows you how to build a basic pipeline
```python
pipeline = (
        WSIGrid(slides=slides, tile_size=256, stride=256, level=0)
        .then(AttachROIs(providers=[RectROIProvider(rois_dict)]))
        .then(TilePlanner())
        .then(ReadWindowChunker(max_window_size=4096))
        .then(RegionReadAndBatch(batch_size=args.batch, num_workers=args.num_workers))
        .then(DummyTissueClassifier("cuda"))
        .then(PNGEncoder())
        .then(WebDatasetWriter())
)
```

> Check out the `examples/demo.py` for a basic use case of this library. After you have set up input and parameters, you can run it with `python examples/demo.py`.

# More will come
- [ ] More readme stuff
- [ ] Tests
- [ ] Abstract writer to easily extend and write different writers
- [ ] More basic stuff (i.e. numpy encoder, basic xml ROI reader, cellvit tissue classifier, etc)
- [ ] Functionality for just running the pipeline on a single tiff and getting a list of patches (instead of saving to a dataset)
- [ ] More examples
