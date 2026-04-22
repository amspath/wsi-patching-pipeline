[![Unit tests](https://github.com/amspath/wsi-patching-pipeline/actions/workflows/unit_tests.yaml/badge.svg)](https://github.com/amspath/wsi-patching-pipeline/actions/workflows/unit_tests.yaml)
[![Integration tests](https://github.com/amspath/wsi-patching-pipeline/actions/workflows/integration_tests.yaml/badge.svg)](https://github.com/amspath/wsi-patching-pipeline/actions/workflows/integration_tests.yaml)

# wsi-patching-pipeline
A pragmatic pipeline for streaming whole-slide image (WSI) patches with region prefetch, per-WSI multiprocessing producers, and an async writer. It's ideal for building pipelines that create datasets, or for patching in a streaming fashion during inference without overloading your memory. It’s designed as a runnable skeleton you can extend: swap in your own ROI logic, classifiers, encoders, or sinks by building components on the `custom_component` module facilities.

✨ What you get
- Streaming, regionized tiling of WSIs.
- Per-slide producers (theading) feeding a bounded queue.
- Single writer process for continuous writing in the sink (i.e. to a webdataset, or numpy arrays).
- Batched GPU steps.
- Built-in isolated stage profiling per slide + aggregated stats.


## 1) Library install

Python ≥3.10 <3.14 is recommended. 3.14 is yet unsupported.
```
# CPU install
pip install "wsi-patching @ git+https://github.com/amspath/wsi-patching-pipeline.git"

# GPU install
pip install "wsi-patching[gpu] @ git+https://github.com/amspath/wsi-patching-pipeline.git"
```
> For some stages, GPUs are required, and thus the GPU install is required. 


## 2) Checkout the examples
`numpy_mem_writer_demo.py` shows you how to build a basic pipeline for patching up a wsi in memory, without the need of writing to disk (RAM heavy for larger datasets, obviously).
```python
p = (
  WSIGrid(slides=slides, resolution=0, unit="level")
  .then(AttachROIs(providers=[RectROIProvider(rois_dict)]))
  .then(PatchExtractor(tile_size=256, stride=256))
  .to(NumpyStreamWriter(layout="NCHW"))
)
stream = p.stream(num_workers=4)
for wsi_id, final_images, final_coords, meta in stream:
    ...
```

> Useful to know: Stream writers do not order the patches per WSI. This improves speed. You can check the wsi_id returned from the stream to check which WSI each batch belongs to. If you per se want ordered batches, you can set the num_workers to 1. 

`webdataset_materialize_writer_demo.py` shows you how to build a basic pipeline for creating a WebDataset (A materializing pipeline)
```python
p = (
  WSIGrid(slides=slides, resolution=0, unit="level")
  .then(AttachROIs(providers=[RectROIProvider(rois_dict)]))
  .then(PatchExtractor(tile_size=224, stride=224))
  .then(PNGEncoder())
  .to(WebDatasetWriter(shard_size=300, shuffle_buffer_size=500))
)
p.materialize(num_workers=4)
```

## 3) Check out the currently available components:
#### Core components
- `WSIGrid`: Your starter block
- `AttachROIs`: Attach an ROI provider class to ensure that only your regions of interest are patched up. There are two basic ROI providers implemented, being a `RectROIProvider`, and a `RectROIfromXMLProvider`. More to come when needed.
- `PatchExtractor`: A necessary component in every pipeline. This will nicely read and batch up all your patches. The PatchExtractor is a composite stage, which contains three stages. If you want to adapt the PatchExtractor logic, you can individually add the substages to your pipeline and swap out stages for custom ones.
- A sink component to define what the output should be. There are two types of writers:
  - `MaterializeWriterBase` writers: For writers that materialize data (i.e. write to disk). These types of writers are particularly useful for making large training datasets. Examples include:
    - `WebdatasetWriter`: For writing to/creating a webdataset.
  - `StreamWriterBase` writers: For writers that stream batches of data to memory. These types of writers are particularly useful for patching during inference or in any situation where you don't necessarily want to write to your disk. Examples include:
    - `NumpyStreamWriter`: For obtaining numpy patches. 
    - `TorchStreamWriter`: For obtaining torch tensors.

#### Other components
- Filters: For filtering out your patches that you do not need
  - `LowContrastBackgroundFilter`: A simple filter for filtering out background with very little difference between pixels.
  - `OtsuFilter`: Applying otsu's method and filtering on a threshold.
  - `PenArtifactFilter`: Applying histolabs blue, green and red pen filters, but using our own batched, gpu accelerated implementation.
  - `CellVitTissueClassifierFilter`: Using CellVits original tissue classifier, it classifies patches as background using a mobilenetv3. 
- Transforms: For transforming your patches 
  - `Macenko Normalizer`: Applies Macenko normalizer, fitting on the first batch it encounters (watch out for the first batch being a background batch).
- Encoders: For encoding your patches into the right format
  - `PNGEncoder` Transforming your patches into PNGs. Particularly useful for the WebDatasetWriter.


More to come! Request if you would like your stage to be in the library.

## 4) Build your own components
This library is setup such that you can easily build your own components to suit your own needs and pop it into the pipeline. Components can either extend the `Stage` (a processing stage) or `WriterBase` (a sink) components. 

#### Creating a stage component:
```python
from wsi_patching.custom_component import Stage, PipelineContext, # <PreviousStageOutputType> and <NextStageInputType> can also be found here

class CustomStage(Stage):
    def __init__(self, ...):
            ...
    
    def export_context(self, ctx: "PipelineContext") -> None:
            # Optional: Seed/override global grid parameters for other stages to read, i.e.
            ctx["tile_size"] = self.tile_size
    def validate(self) -> None:
            # Optional: Validate your class before starting processing, i.e.
            self.ctx.require_key("use_gpu")
            if self.ctx['some_key'] < self.some_init_param:
                    ...
    
    def __call__(self, it: Iterable[<PreviousStageOutputType>]) -> Iterable[<NextStageInputType>]:
            # The logic of your stage. You should specifiy the type of your call function. 
            # These should align with the preceeding and succeeding stages (checked at initialization).
            ...
```

#### Creating a custom sink component:
```python
from wsi_patching.custom_component import StreamWriterBase, MaterializeWriterBase, # <PreviousStageOutputType> can also be found here

class CustomWriter(MaterializeWriterBase):
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

# StreamWriterBase is slightly simpler
```

#### Profiling runtime of a custom component:
```python
import time

class CustomStage(Stage/WriterBase):
    ...

    def __call__(self, it: ...):
        prof = self.get_current_profiler()
        for something in it:
            # Start measuring the iteration
            t0 = time.perf_counter()

            # Do some heavy operation
            output = ...

            # Stop the clock
            dt = time.perf_counter() - t0
            if output:
                prof.add_time("CustomStage", dt, yielded=True)
                yield output
            else:
                prof.add_time("CustomStage", dt, yielded=False)
```
This will count per slide per iteration breakdown of how fast this stage is, in the form of:
```
=== Pipeline Profile (isolated timings only) ===
Stage                                Yields         Wall (s)   Avg (ms/yield)
PNGEncoder.isolated                    640            1.440s          2.412ms

--- Per slide breakdown ---
[RBIO-GC072-HE-02]
  PNGEncoder.isolated          yields=  320    wall=  0.762s    avg=  2.382ms
[RBIO-GC072-HE-01]
  PNGEncoder.isolated          yields=  320    wall=  0.778s    avg=  2.432ms
```

## 5) Development install

Python ≥3.10 is recommended. UV is also recommended as the package manager
```
git clone https://github.com/amspath/wsi-patching-pipeline.git
cd wsi-patching-pipeline
uv sync --extra dev,gpu
```


## 6) More will come
- [ ] Retrained MobileNet classifier for tissue detection with proper documentation. 
- [ ] Controlled batch sizes in each of the shipped writers

## 7) Contributing
Feel free to contribute by opening a pull request or adding an issue to the github. We develop this library based on our own usage of it. So if something is not implemented, we just haven´t had the need for it yet. However, we are very open to add new functionality.