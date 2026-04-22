# Extending

All base classes and relevant types are importable from `wsi_patching.custom_component`.

```python
from wsi_patching.custom_component import (
    Stage,
    StreamWriterBase,
    MaterializeWriterBase,
    PipelineContext,
    # types you may need for annotations:
    CollatedPatchBatch,
    EncodedCollatedPatchBatch,
    Slide,
    SlideWithROIs,
)
```

---

## Custom Stage

A `Stage` is a callable that accepts an iterable and yields transformed items. Implement it as a generator.

```python
from typing import Iterable
from wsi_patching.custom_component import Stage, PipelineContext, CollatedPatchBatch

class MyFilter(Stage):
    input_type = CollatedPatchBatch   # checked against the preceding stage's output_type
    output_type = CollatedPatchBatch  # checked against the succeeding stage's input_type

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def export_context(self, ctx: PipelineContext) -> None:
        # Optional: write values that other stages may need
        ctx["my_filter_threshold"] = self.threshold

    def validate(self) -> None:
        # Optional: assert preconditions using the fully-populated context
        self.ctx.require_key("use_gpu")  # raises KeyError with a clear message if missing

    def __call__(self, it: Iterable[CollatedPatchBatch]) -> Iterable[CollatedPatchBatch]:
        for batch in it:
            filtered = self._apply(batch)
            if filtered is not None:
                yield filtered

    def _apply(self, batch: CollatedPatchBatch):
        ...
```

**Key rules:**

- `input_type` and `output_type` are class attributes used for preflight type checking. Set them to the exact types from `wsi_patching.custom_component` (or `object` to opt out of checking).
- `export_context` and `validate` are optional — omit them if you have nothing to export or validate.
- `validate` runs after the full context is populated, so `self.ctx["key"]` is safe to read there.
- The `__call__` method **must be a generator** (use `yield`). Do not return a list.

---

## Custom StreamWriter

Implement `stream()` as a generator that yields whatever your downstream consumer expects.

```python
from typing import Any, Iterator
from wsi_patching.custom_component import StreamWriterBase, CollatedPatchBatch

class MyStreamWriter(StreamWriterBase):
    input_type = CollatedPatchBatch

    def stream(self, batch: CollatedPatchBatch) -> Iterator[tuple[str, Any]]:
        # Called once per batch produced by the pipeline
        yield batch.wsi_id, batch.patches
```

`stream()` is called once per queue item. The pipeline handles threading and queue management; you only need to focus on the transformation.

---

## Custom MaterializeWriter

Implement `open()`, `write()`, and `close()` — called once, repeatedly, and once respectively.

```python
import csv
from typing import Any, Optional
from wsi_patching.custom_component import MaterializeWriterBase, CollatedPatchBatch

class CsvWriter(MaterializeWriterBase):
    input_type = CollatedPatchBatch

    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self._file = None
        self._writer = None

    def open(self) -> None:
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["wsi_id", "x", "y"])

    def write(self, batch: CollatedPatchBatch) -> None:
        for coord in batch.coords:
            self._writer.writerow([batch.wsi_id, coord[0], coord[1]])

    def close(self) -> None:
        if self._file:
            self._file.close()

    def get_output(self) -> Optional[str]:
        # Optional: return something after materialization completes
        return self.path
```

!!! note
    `close()` is always called, even if an error occurred during `write()`. Use it for cleanup (closing file handles, flushing buffers).

---

## Profiling your component

To appear in the pipeline's profile output, instrument your `__call__` method with the built-in profiler:

```python
import time
from typing import Iterable
from wsi_patching.custom_component import Stage, CollatedPatchBatch

class MyStage(Stage):
    input_type = CollatedPatchBatch
    output_type = CollatedPatchBatch

    def __call__(self, it: Iterable[CollatedPatchBatch]) -> Iterable[CollatedPatchBatch]:
        prof = self.get_current_profiler()  # None when profile=False
        for batch in it:
            t0 = time.perf_counter()

            result = self._process(batch)

            dt = time.perf_counter() - t0
            if result is not None:
                prof.add_time("MyStage", dt, yielded=True)
                yield result
            else:
                prof.add_time("MyStage", dt, yielded=False)
```

- `yielded=True` — this timing window produced output (the `yield` branch).
- `yielded=False` — this timing window filtered the item (no `yield`).
- The profiler is per-slide and per-thread; the pipeline aggregates results automatically.
- When `profile=False`, `get_current_profiler()` returns `None` — guard with `if prof:` if you call methods on it conditionally, or just pass it directly to `add_time` (it no-ops when disabled).

---

## Type flow

The pipeline validates that each stage's `output_type` is a subclass of the next stage's `input_type`. This check runs at pipeline construction (in `.then()` and `.to()`), not at run time.

```
WSIGrid          → output_type = Slide
AttachROIs       → input_type  = Slide,          output_type = SlideWithROIs
PatchExtractor   → input_type  = SlideWithROIs,  output_type = CollatedPatchBatch
PNGEncoder       → input_type  = CollatedPatchBatch, output_type = EncodedCollatedPatchBatch
WebDatasetWriter → input_type  = EncodedCollatedPatchBatch
```

Set `input_type = object` or `output_type = object` on your custom component to opt out of the check for that boundary.
