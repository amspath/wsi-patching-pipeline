[![Unit tests](https://github.com/amspath/wsi-patching-pipeline/actions/workflows/unit_tests.yaml/badge.svg)](https://github.com/amspath/wsi-patching-pipeline/actions/workflows/unit_tests.yaml)

# wsi-patching-pipeline
A pragmatic pipeline for streaming whole-slide image (WSI) patches with region prefetch, per-WSI multiprocessing producers, and a single async WebDataset writer. It’s designed as a runnable skeleton you can extend: swap in your own ROI logic, classifiers, encoders, or sinks by building components on the `custom_stage` module facilities.

✨ What you get
- Streaming, regionized tiling of WSIs (cuCIM preferred; Pillow fallback for small images).
- Per-slide producers (multiprocessing) feeding a bounded MP queue.
- Single writer process for continuous writing in the sink (i.e. to a webdataset).
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
        .then(ReadWindowChunker())
        .then(RegionReadAndBatch(batch_size=args.batch, num_workers=args.num_workers))
        .then(CellVitTissueClassifierFilter())
        .then(PNGEncoder())
        .to(WebDatasetWriter(shard_size=300, shuffle_buffer_size=500))
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

## 3) Build your own components
This library is setup such that you can easily build your own components to suit your own needs and pop it into the pipeline. Components can either extend the `Stage` (a processing stage) or `WriterBase` (a sink) components. 

#### Creating a stage component:
```python
from wsi_patching.custom_component import Stage

class CustomerStage(Stage):
        def __init__(self, ...):
                ...
        
        def export_context(self, ctx: "PipelineContext") -> None:
                # Seed/override global grid parameters for other stages to read, i.e.
                ctx["tile_size"] = self.tile_size

        def validate(self) -> None:
                # Validate your class before starting processing, i.e.
                self.ctx.require_key("use_gpu")
                if self.ctx['some_key'] < self.some_init_param:
                        ...
        
        def __call__(self, it: Iterable[<PreviousStageOutputType>]) -> Iterable[NextStageInputType]:
                # The logic of your stage. You should specifiy the type of your call function. 
                # These should align with the preceeding and succeeding stages (checked at initialization).
                ...
```

#### Creating a custom sink component:
```python
from wsi_patching.custom_component import WriterBase

class CustomerWriter(WriterBase):
        def __init__(self, ...):
                ...

        def open(self) -> None:
                # Opening your writer
        
        def write(self, sample: <PreviousStageOutputType>) -> None:
                # What to do with a single sample

        def close(self) -> None:
                # Closing up the buffer
        
        def get_output(self) -> Any:
                # Optional: if you want to output something in memory.
```

# More will come
- [ ] Macenko tile normalization
- [ ] More filters
- [ ] Retrained MobileNet classifier for tissue detection with proper documentation
