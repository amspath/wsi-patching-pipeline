[![Unit tests](https://github.com/amspath/wsi-patching-pipeline/actions/workflows/unit_tests.yaml/badge.svg)](https://github.com/amspath/wsi-patching-pipeline/actions/workflows/unit_tests.yaml)

# wsi-patching-pipeline
A pragmatic pipeline for streaming whole-slide image (WSI) patches with region prefetch, per-WSI multiprocessing producers, and a single async WebDataset writer. It’s designed as a runnable skeleton you can extend: swap in your own ROI logic, classifiers, encoders, or sinks by building components on the `custom_component` module facilities.

✨ What you get
- Streaming, regionized tiling of WSIs (cuCIM preferred; Pillow fallback for small images).
- Per-slide producers (multiprocessing) feeding a bounded MP queue.
- Single writer process for continuous writing in the sink (i.e. to a webdataset).
- Batched GPU steps.
- Built-in isolated stage profiling per slide + aggregated stats.


## 1) Library install

Python ≥3.8 <3.14 is recommended. 3.14 is yet unsupported.
```
# CPU install (not all functionality is supported for cpu only)
pip install "wsi-patching @ git+https://github.com/amspath/wsi-patching-pipeline.git"

# GPU install
pip install "wsi-patching[gpu] @ git+https://github.com/amspath/wsi-patching-pipeline.git"
```

> If you run without gpu, the backend will rely on OpenSlide to open images. Openslide requires system libraries. It is your own responsibility to install these. The easiest way to install those is to create your environment through conda and add the required system libraries in there. 


## 2) Dev install

Python ≥3.8 is recommended.
```
git clone https://github.com/amspath/wsi-patching-pipeline.git
cd wsi-patching-pipeline
pip install -e .        # For cpu only
pip install -e .[gpu]
```

## 3) Checkout the examples
`demo.py` shows you how to build a basic pipeline for creating a WebDataset
```python
p = (
    WSIGrid(slides=slides, level=0, use_gpu=True)
    .then(AttachROIs(providers=[RectROIProvider(rois_dict)]))
    .then(PatchExtractor(tile_size=224, stride=224, max_batch_size=800))
    .then(CellVitTissueClassifierFilter())
    .then(PNGEncoder())
    .to(WebDatasetWriter(shard_size=300, shuffle_buffer_size=500))
)
p.run(cpu_processes=4)
```

`numpy_mem_writer_demo.py` shows you how to build a basic pipeline for patching up a wsi in memory, without the need of writing to disk (RAM heavy for larger datasets, obviously).
```python
p = (
    WSIGrid(slides=slides, level=0, use_gpu=True)
    .then(PatchExtractor(tile_size=256, stride=256, max_batch_size=800, num_workers=4))
    .then(LowContrastBackgroundFilter(range_threshold=0.2))
    .then(MacenkoNormalizer())
    .to(NumpyMemoryWriter(layout="NCHW"))
)
np_patch_array, np_coords_array, list_of_wsi_ids = p.run(cpu_processes=2)
```

## 4) Check out the currently available components:
#### Core components
- `WSIGrid`: Your starter block!
- `AttachROIs`: Attach an ROI provider class to ensure that only your regions of interest are patched up. There are two basic ROI providers implemented, being a `RectROIProvider`, and a `RectROIfromXMLProvider`. More to come when needed.
- `PatchExtractor`: A necessary component in every pipeline. This will nicely read and batch up all your patches. 
- A sink component to define what the output should be. Currently implemented are:
  - `WebDatasetWriter`: For writing to a webdataset. Comes with a WebDataset Loader for then reading in the the webdataset in a streamed format to a Torch dataloader.
  - `NumpyMemoryWriter`: For obtaining numpy patches as one big np.ndarray. 
  - `TorchMemoryWriter`: For obtaining all your patches as a Torch dataset.

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

## 5) Build your own components
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
from wsi_patching.custom_component import WriterBase, # <PreviousStageOutputType> can also be found here

class CustomWriter(WriterBase):
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

# More will come
- [ ] Retrained MobileNet classifier for tissue detection with proper documentation
