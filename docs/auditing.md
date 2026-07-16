# Auditing & visualization

Tuning a patching pipeline means answering one question: **what does each stage
throw out, and why?** The auditing tools record, per stage per slide, how many
patches came in and out, *which* patches each stage dropped, and the metadata
behind each drop — then show it as a funnel table and an interactive overlay.

## Dry run: audit without writing output

`dry_run()` runs the pipeline's filters **without a writer and without encoding**,
purely to collect the audit. It's the fast way to iterate on parameters. No
writer is required.

```python
from wsi_patching import PatchExtractor, OtsuFilter, RemoveEdgeTiles, WSIGrid

p = (
    WSIGrid(slides=["slide.tiff"], resolution=2, unit="level")
    .then(PatchExtractor(tile_size=224, stride=224))
    .then(OtsuFilter(min_tissue_fraction=0.35))
    .then(RemoveEdgeTiles(depth=1))
)

p.dry_run()
p.print_audit()
```

```
=== Audit: slide ===
Stage                                  In        Out    Dropped   Drop %
PatchExtractor                        252        252          0     0.0%
OtsuFilter                            252         85        167    66.3%
RemoveEdgeTiles                        85         42         43    50.6%
```

## Interactive overlay

`visualize_audit(pipeline, slide)` returns a report that renders inline in
Jupyter (it's the last expression in the cell) and can be saved as a
self-contained HTML file.

```python
from wsi_patching import visualize_audit

report = visualize_audit(p, "slide.tiff")
report                       # renders inline in a notebook
report.save("report.html")   # standalone shareable file
```

- **Green** patches are kept; each other color is a stage that dropped those patches.
- **Hover** a patch to see the stage and its metadata (e.g. `tissue_fraction=0.03`).
- **Sliders** re-tune each filter's threshold live — see below.
- **Unchecking a stage in the legend removes it from the pipeline**: everything is
  re-filtered without it, so the patches it had dropped are re-judged by the
  remaining stages and recolored (kept, or dropped by a later stage). The funnel
  marks it `(off)` and the config box drops it.
- **The ▲▼ arrows reorder the filters**, re-running the whole attribution in the
  new order — see below.

Every legend checkbox means exactly one thing — *is this stage in the pipeline?*
`kept` and `unknown` are outcomes rather than stages, so they are shown as plain
swatches with no checkbox.

## Reordering filters

The ▲▼ arrows move a filter up or down the pipeline. The overlay, funnel and config
box all follow, and the config box is emitted in the new order — paste it straight
back into your code.

**Reordering cannot change which patches survive.** A patch is kept only if it
passes *every* filter, and that is independent of the order they run in. What it
changes is:

- **Who gets the blame.** With `OtsuFilter` last, it looks responsible for only 57
  drops; move it first and you see it is really rejecting 167 patches — the other
  filters were just getting to some of them first.
- **How much work each stage does** — which is a real speed decision. Putting the
  cheap filter first means the expensive one sees far fewer patches:

```
RemoveEdgeTiles     252 -> 190      OtsuFilter          252 -> 85
LowContrast         190 ->  67      RemoveEdgeTiles      85 -> 42
OtsuFilter           67 ->  10      LowContrast          42 -> 10
```

Both keep the same 10 patches, but the left order only runs Otsu on 67 patches
instead of 252. `RemoveEdgeTiles` is pure arithmetic, so it is nearly free to put
first.

A filter will only swap with an adjacent **filter**. It cannot be moved across the
source or a pixel-transforming stage (e.g. a normalizer), because the recorded
metrics were measured on that stage's output — moving past it would invalidate them.
Those arrows are disabled.

Between the two you can answer both tuning questions without ever re-running:
*what threshold?* (slider) and *do I even need this filter?* (checkbox).

!!! note
    Unchecking a filter that has **no slider** (e.g. `CellVitTissueClassifierFilter`)
    turns its patches **grey — "unknown"**, not green. Those patches were dropped in
    Python before the later stages ever measured them, so there is no honest way to
    say what the rest of the pipeline would have done with them. Re-run without that
    stage to find out.

## Live parameter sliders

After a `dry_run()`, the report grows a **slider for every tunable filter
parameter**. Drag one — or type an exact value in the box on the right — and the
overlay, the funnel and a copy-pasteable config update instantly:

```
[▲][▼]  RemoveEdgeTiles.depth           >=  [==O=============]  [   1   ]
[▲][▼]  OtsuFilter.min_tissue_fraction  >=  [======O=========]  [ 0.350 ]

.then(RemoveEdgeTiles(depth=1))
.then(OtsuFilter(min_tissue_fraction=0.350))
```

When you like what you see, paste the config box straight into your pipeline.

The slider track spans the observed range of the metric, since nothing outside it
changes the outcome — but the number box accepts any value, so you can push a
threshold past the extremes if you want to see it drop everything.

**No Python runs while you drag.** Every threshold filter computes a per-patch
metric that doesn't depend on its threshold (`tissue_fraction`, `pen_fraction`,
`gray_range`, `edge_distance`), so the report records the metric once and
re-decides keep/drop in the browser. That's why the sliders keep working in a
saved `.html` file with no kernel attached.

### Which parameters get a slider

| Parameter | Slider |
| --- | --- |
| `OtsuFilter.min_tissue_fraction` | ✅ |
| `PenArtifactFilter.max_pen_fraction` | ✅ |
| `LowContrastBackgroundFilter.range_threshold` | ✅ |
| `RemoveEdgeTiles.depth` | ✅ |
| `OtsuFilter.num_bins` / `tissue_is_darker`, `PenArtifactFilter.diff_thresh` | ❌ changes the metric itself |
| `tile_size` / `stride` | ❌ changes the patch grid |

The ❌ ones can't be recomputed from recorded data — change them in the
constructor and `dry_run()` again.

### How `dry_run` makes this work

A filter only computes its metric for the patches that reach it. With real
thresholds, a patch dropped by `OtsuFilter` never gets a `pen_fraction`, so a
slider could only ever *tighten*. To avoid that, `dry_run()` runs every tunable
filter **permissively** (thresholds neutralized so nothing drops); each patch
then flows through every stage and carries every metric, and any threshold can be
re-decided offline in either direction.

The funnel is unaffected: the drops are re-derived from the recorded metrics at
*your* configured thresholds, so `print_audit()` prints the same table it would
for a real run. The cost is speed — nothing is rejected early, so every filter
sees every patch. To audit many slides fast, use `dry_run(tune=False)`; the
report is then static.

`materialize(audit=True)` / `stream(audit=True)` are never permissive — a real
run writes exactly what its real thresholds say — so those reports are static too.

### Tuning knobs on your own stage

Any stage whose decision is `metric op threshold` can opt in by declaring a
`Knob`, as long as the metric does not depend on the threshold:

```python
from wsi_patching.custom_component import Knob, Stage

class MyFilter(Stage):
    AUDIT_KNOBS = (Knob("max_blur", "blur_score", "<=", permissive=1.0),)

    def __call__(self, it):
        for batch in it:
            batch.add_meta_column("blur_score", scores)   # for every row, before filtering
            batch.filter_on_mask(scores <= self.max_blur)
            yield batch
```

`permissive` is the value at which the stage drops nothing. Attach the metric
column for **all** rows *before* `filter_on_mask`, or the dropped patches won't
have it.

## Auditing a real run

You don't have to dry-run. Pass `audit=True` to a normal run to get the same
funnel and overlay alongside your written output:

```python
p.materialize(audit=True)   # or p.stream(audit=True)
p.print_audit()
visualize_audit(p, "slide.tiff")
```

Auditing is opt-in and adds no overhead to runs that don't request it.

## Programmatic access

`pipeline.get_audit()` returns the raw data if you want to build your own
report:

```python
audit = pipeline.get_audit()["by_slide"]["slide"]
audit["funnel"]      # [{"stage", "in", "out", "dropped"}, ...]
audit["kept"]        # (K, 2) coords surviving the whole pipeline
audit["dropped"]     # [{"stage", "coords": (D, 2)}, ...] per dropping stage
audit["candidates"]  # (N, 2) every coord that entered the pipeline
audit["patch_meta"]  # {(x, y): {col: value}} per-patch metadata
audit["knobs"]       # tunable params of the run, with their configured values
```

After a `dry_run()` you can also re-derive the funnel for other knob values
without re-running — the same computation the sliders do, in Python:

```python
knob = pipeline.get_audit()["by_slide"]["slide"]["knobs"][0]
for v in (0.1, 0.2, 0.3):
    entry = pipeline.get_audit(values={knob["stage_idx"]: v})["by_slide"]["slide"]
    print(v, len(entry["kept"]))
```

!!! note
    Attribution is generic: a stage's drops are the patch coordinates present in
    its input but absent from its output. This works for every stage — current or
    custom — with no per-stage code. Per-patch metadata is captured at the single
    drop point (`CollatedPatchBatch.filter_on_mask`) for dropped patches and at the
    last stage for the survivors, so any columns a filter attaches
    (`tissue_fraction`, `pen_fraction`, …) show up automatically.
