# Concepts

## Pipeline builder

Pipelines are built with a fluent API:

```python
pipeline = (
    WSIGrid(...)          # source stage
    .then(AttachROIs(...))
    .then(PatchExtractor(...))
    .then(PNGEncoder())   # optional intermediate stages
    .to(WebDatasetWriter(...))  # sink
)
```

Each `.then()` call returns a new `Pipeline` object with the stage appended. `.to()` attaches the writer and seals the pipeline. **Type compatibility between consecutive stages is checked at construction time**, not at run time. Each `Stage` subclass declares `input_type` and `output_type` as class attributes; the pipeline raises a `TypeError` immediately if any adjacent pair is incompatible.

You cannot add stages after `.to()` — the writer is always the last element.

---

## Threading model

Running a pipeline spawns three categories of threads:

```
┌────────────────────────────────────────────────────────┐
│  Supervisor thread                                     │
│  - tracks which slides are done                        │
│  - spawns new producer threads as capacity frees up    │
└────────────┬───────────────────────────────────────────┘
             │ spawns up to num_workers threads
  ┌──────────▼──────────┐  ┌──────────────────────┐
  │ Producer (slide A)  │  │ Producer (slide B)   │  ...
  │ runs all stages     │  │ runs all stages      │
  └──────────┬──────────┘  └──────────┬───────────┘
             │                        │
             └──────────┬─────────────┘
                        │ bounded Queue
                        │ maxsize = writer_prefetch_factor * num_workers
             ┌──────────▼──────────┐
             │ Writer              │
             │ .stream()  → generator in caller's thread  │
             │ .materialize() → dedicated writer thread   │
             └─────────────────────┘
```

- **Producer threads** each run an independent copy of the full stage chain for one slide. When a producer finishes a slide it puts an `EndOfStream` sentinel on the queue to signal the slide boundary. The supervisor then spawns a new producer for the next pending slide.
- **The queue** is bounded (`maxsize = writer_prefetch_factor × num_workers`, default 8). This provides natural backpressure: producers block when the writer is slow rather than accumulating unbounded memory.
- **`EndOfQueue`** is put onto the queue once all slides are done, signalling the writer to stop.
- In **`.stream()` mode** the writer runs as a generator in the caller's thread — each iteration of `for wsi_id, images, ... in p.stream()` drives the writer forward.
- In **`.materialize()` mode** the writer runs in a dedicated thread and the call blocks until all slides complete.

---

## PipelineContext

`PipelineContext` is a shared key-value store (dict-like) that stages use to communicate configuration values without tight coupling.

**Lifecycle:**

1. Before the pipeline starts, `export_context(ctx)` is called on every stage in order. Each stage writes its own keys (e.g. `WSIGrid` writes `ctx["tile_size"]`, `ctx["use_gpu"]`, etc.).
2. Then `attach_context(ctx)` is called on every stage, making the fully-populated context available as `self.ctx`.
3. Then `validate()` is called on every stage so each can assert required keys exist and check cross-stage constraints.

Because `export_context` runs before `validate`, a downstream stage can safely call `self.ctx.require_key("tile_size")` in its `validate()` and be guaranteed the key was set by an upstream stage.

```python
def export_context(self, ctx: PipelineContext) -> None:
    ctx["my_param"] = self.my_param

def validate(self) -> None:
    self.ctx.require_key("use_gpu")  # raises KeyError with a clear message if missing
```

---

## Resolution and fallback modes

`WSIGrid` takes a `resolution` + `unit` pair to select which pyramid level to read from:

| `unit` | `resolution` meaning |
|---|---|
| `"level"` | Pyramid level index (0 = full resolution) |
| `"mpp"` | Microns per pixel |
| `"downsample"` | Downsample factor relative to level 0 |

If the slide does not have an exact match for the requested resolution, `fallback_mode` controls what happens:

| `fallback_mode` | Behaviour |
|---|---|
| `"error"` | Raises an error if no exact match |
| `"nearest"` *(default)* | Picks the closest available level, provided it is within `max_relative_deviation` of the request |
| `"floor"` | Picks the highest resolution level that is ≤ the request |
| `"ceil"` | Picks the lowest resolution level that is ≥ the request |
| `"resample"` | Reads from a finer level, then downsamples with `cv2.resize` to hit the exact target. Not valid with `unit="level"`. |

`max_relative_deviation` guards `fallback_mode="nearest"`: if the closest level deviates from the requested resolution by more than this fraction (default `0.01`, i.e. 1%), an error is raised instead of silently patching at the wrong resolution. Pass `None` to disable the check. The other fallback modes ignore it.

`resample_interpolation` controls the interpolation method when `fallback_mode="resample"` (default `"lanczos"`; options: `"nearest"`, `"linear"`, `"cubic"`, `"area"`, `"lanczos"`).

---

## Profiling

Enable profiling by passing `profile=True` to `.stream()` or `.materialize()`. After the pipeline completes, call `.print_profile()` or `.get_profile()`:

```python
p.materialize(num_workers=4, profile=True)
p.print_profile()
```

Output:

```
=== Pipeline Profile (isolated timings only) ===
Stage                                Yields         Wall (s)   Avg (ms/yield)
PNGEncoder.isolated                    640            1.440s          2.412ms

--- Per slide breakdown ---
[slide_a]
  PNGEncoder.isolated          yields=  320    wall=  0.762s    avg=  2.382ms
```

Only stages that explicitly call `self.get_current_profiler()` and `prof.add_time(...)` appear in the profile. The built-in stages do this for you. See [Extending — Profiling](extending.md#profiling-your-component) for how to add it to custom stages.
