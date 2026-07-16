import base64
import html as _html
import io
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, Union
from typing import cast as _cast

import numpy as np
from PIL import Image, ImageDraw

from wsi_patching.backends.fastslide import get_dimensions_for_level, get_level_for_resolution, read_region

CoordLike = Union[Tuple[int, int], Sequence[int], np.ndarray]

# Stable-ish palette for coloring drops by stage (kept patches are always green).
_STAGE_PALETTE = [
    "#e74c3c", "#3498db", "#e67e22", "#9b59b6", "#1abc9c",
    "#f1c40f", "#e84393", "#2c3e50", "#16a085", "#d35400",
]
_KEPT_COLOR = "#2ecc71"


def visualize_selected_patches(
    wsi_path: str,
    coords: Iterable[CoordLike],
    patch_size: int,
    *,
    stride: Optional[int] = None,
    patch_images: Optional[Union[np.ndarray, Mapping[CoordLike, np.ndarray]]] = None,
    thumb_long_side: int = 2000,
    selected_outline_rgba: Tuple[int, int, int, int] = (0, 220, 0, 100),
    save_path: Union[str, Path, None] = None,
) -> Image.Image:
    """
    Render a thumbnail of the WSI and optionally overlay the actual patch images.

    - Background: WSI thumbnail.
    - Optional: fill every *candidate* tile (stride grid) in bounding region as dark gray.
    - Optional: overlay real patch images at their locations (alpha-blended or replaced).
    - Selected patches are outlined in green.

    Args:
        wsi_path: path to the slide.
        coords: iterable of (x, y) top-left anchors at base resolution.
        patch_size: square patch size at base resolution.
        stride: grid stride at base resolution (defaults to patch_size).
        patch_images:
            numpy array of patch images in order of `coords`
        save_path: if provided, save the output PNG to this path.

    Returns:
        PIL Image object of the composite visualization.
    """
    if stride is None:
        stride = patch_size
    if stride <= 0 or patch_size <= 0:
        raise ValueError("`stride` and `patch_size` must be positive integers.")

    # Normalize coords
    coords_arr: list[tuple[int, int]] = []
    for c in coords:
        x, y = int(c[0]), int(c[1])
        coords_arr.append((x, y))
    if not coords_arr:
        raise ValueError("No coordinates provided.")

    # Optionally snap to stride grid to avoid tiny round-off mismatches
    coords_arr = [((x // stride) * stride, (y // stride) * stride) for x, y in coords_arr]

    # Map coords -> patch image (if provided)
    coord_to_patch: dict[tuple[int, int], np.ndarray] = {}
    if patch_images is not None:
        if isinstance(patch_images, Mapping):
            # dict mapping top-left to image
            for k, v in patch_images.items():
                _k = _cast(Sequence[int], k)
                coord_to_patch[(int(_k[0]), int(_k[1]))] = np.asarray(v)
        else:
            # assume same order as coords
            if len(patch_images) != len(coords_arr):
                raise ValueError("If `patch_images` is a sequence, it must match len(coords).")
            for xy, img in zip(coords_arr, patch_images):
                coord_to_patch[xy] = np.asarray(img)

    # Open slide and make thumbnail
    W0, H0 = get_dimensions_for_level(wsi_path, level=0)
    scale = float(thumb_long_side) / max(W0, H0)
    tw = max(1, int(round(W0 * scale)))
    th = max(1, int(round(H0 * scale)))
    base_thumb = get_thumbnail(wsi_path, (tw, th), (W0, H0)).convert("RGB")
    tw, th = base_thumb.size

    # Overlay layers
    # - gray grid + green outlines are drawn on 'overlay' (RGBA)
    # - patch bitmaps are pasted directly onto 'composite' (start from the base thumbnail)
    composite = base_thumb.convert("RGBA")
    overlay = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Bounding region aligned to stride, covering all selected patches fully
    xs = [x for x, _ in coords_arr]
    ys = [y for _, y in coords_arr]

    min_x_raw, min_y_raw = min(xs), min(ys)
    max_x_end_raw = max(xs) + patch_size
    max_y_end_raw = max(ys) + patch_size

    def ceil_to_stride(v: int, s: int) -> int:
        return ((v + s - 1) // s) * s

    min_x = (min_x_raw // stride) * stride
    min_y = (min_y_raw // stride) * stride
    max_x_end = ceil_to_stride(max_x_end_raw, stride)
    max_y_end = ceil_to_stride(max_y_end_raw, stride)

    # Clamp to slide bounds
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x_end = min(W0, max_x_end)
    max_y_end = min(H0, max_y_end)

    # scale factors
    sx = tw / float(W0)
    sy = th / float(H0)

    # line width for outlines
    line_w = max(1, int(round(max(tw, th) / 1000)))

    # 2) paste/blend actual patch images (if provided)
    if coord_to_patch:
        for (x, y), np_patch in coord_to_patch.items():
            if x + patch_size > W0 or y + patch_size > H0:
                continue  # skip out-of-bounds just in case

            if np_patch.ndim == 3 and np_patch.shape[0] in (3, 4):
                np_patch = np.transpose(np_patch, (1, 2, 0))
            patch_img = _to_pil_rgb(np_patch)

            # resize patch to thumbnail scale
            dst_w = max(1, int(round(patch_size * sx)))
            dst_h = max(1, int(round(patch_size * sy)))
            if patch_img.size != (dst_w, dst_h):
                patch_img = patch_img.resize((dst_w, dst_h), resample=Image.Resampling.BILINEAR)

            # destination location on the thumbnail
            x0_t, y0_t = int(round(x * sx)), int(round(y * sy))

            composite.paste(patch_img, (x0_t, y0_t))

    # 3) draw green outlines for selected patches on top
    for x, y in coords_arr:
        if x + patch_size > W0 or y + patch_size > H0:
            continue
        x0_t, y0_t = int(round(x * sx)), int(round(y * sy))
        x1_t, y1_t = int(round((x + patch_size) * sx)), int(round((y + patch_size) * sy))
        draw.rectangle([x0_t, y0_t, x1_t, y1_t], outline=selected_outline_rgba, width=line_w)

    # composite: base + patch bitmaps + overlay (gray + outlines)
    composite = Image.alpha_composite(composite, overlay)

    if save_path is None:
        return composite.convert("RGB")  # or return RGBA if you want transparency

    save_path = Path(save_path)
    composite.convert("RGB").save(save_path, format="PNG")
    return composite.convert("RGB")


def get_thumbnail(path, max_thumbnail_size: tuple[int, int], slide_dimensions: tuple[int, int]) -> Image.Image:
    downsample = max(dim / thumb for dim, thumb in zip(slide_dimensions, max_thumbnail_size))
    level = get_level_for_resolution(path, resolution=downsample, unit="downsample", fallback_mode="nearest")

    tile: np.ndarray = read_region(path, 0, 0, *get_dimensions_for_level(path, level), level)

    im = Image.fromarray(tile)

    # If RGBA, composite onto white so thumbnail is RGB
    if im.mode == "RGBA":
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
    else:
        im = im.convert("RGB")

    im.thumbnail(max_thumbnail_size, Image.Resampling.LANCZOS)
    return im


def _to_pil_rgb(img: np.ndarray) -> Image.Image:
    a = np.asarray(img)

    # Handle CHW -> HWC
    if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[1] > 4 and a.shape[2] > 4:
        a = np.transpose(a, (1, 2, 0))

    # Squeeze singleton channel-last (H,W,1)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]

    # Grayscale -> RGB
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)

    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported patch array shape after processing: {a.shape}")

    # Convert dtype/range to uint8 before PIL
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        # common case: floats in [0,1]
        if a.max() <= 1.0:
            a *= 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)

    # If RGBA, drop alpha (or composite). Dropping is simplest:
    if a.shape[2] == 4:
        a = a[..., :3]

    return Image.fromarray(a, mode="RGB")


class AuditReport:
    """Interactive keep/drop overlay for a slide.

    Renders inline in Jupyter (via `_repr_html_`) when it is the last expression
    in a cell, and `.save(path)` writes the same self-contained HTML file.
    """

    def __init__(self, html: str) -> None:
        self.html = html

    def _repr_html_(self) -> str:
        return self.html

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        head = "<!doctype html><html><head><meta charset='utf-8'><title>WSI audit</title></head><body>"
        path.write_text(head + self.html + "</body></html>", encoding="utf-8")
        return path


def _fmt_meta(meta: dict) -> str:
    parts = []
    for k, v in meta.items():
        if isinstance(v, (float, np.floating)):
            parts.append(f"{k}={float(v):.4g}")
        elif isinstance(v, (int, np.integer, str, bool, np.bool_)):
            parts.append(f"{k}={v}")
        # skip arrays/lists (e.g. geometry) -- too big for a tooltip
    return ", ".join(parts)


def _num(v: Any) -> Optional[float]:
    """Scalar metric -> plain float for JSON; None if absent or non-scalar."""
    if isinstance(v, (bool, np.bool_)):
        return float(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    return None


def _slider_range(values: Sequence[Optional[float]], init: float, integer: bool) -> Tuple[float, float, float, int]:
    """Derive (lo, hi, step, decimals) from the observed metric spread.

    Everything outside the observed range is a no-op for keep/drop, so the data's
    own min/max already covers every value that changes anything -- no config
    needed. Widened to include the user's current setting so the handle is always
    on the track.

    The step is rounded down to a power of ten so dragging yields round numbers
    (0.35, not 0.3502) -- the report's config box is meant to be pasted into code.
    """
    present = [v for v in values if v is not None]
    lo = min(present + [init]) if present else min(init, 0.0)
    hi = max(present + [init]) if present else max(init, 1.0)
    if integer:
        lo, hi = float(np.floor(lo)), float(np.ceil(hi))
        return lo, max(hi, lo + 1.0), 1.0, 0
    if hi <= lo:
        hi = lo + 1.0
    step = float(10 ** np.floor(np.log10((hi - lo) / 100.0)))
    decimals = max(0, int(-np.floor(np.log10(step))))
    # Align the track to the step grid so every handle position is a round number.
    return float(np.floor(lo / step) * step), float(np.ceil(hi / step) * step), step, decimals


def visualize_audit(
    pipeline: Any,
    wsi_path: str,
    *,
    thumb_long_side: int = 2000,
    box_scale: float = 1.0,
    save_path: Union[str, Path, None] = None,
) -> AuditReport:
    """Build an interactive overlay of kept vs dropped patches for one slide.

    Kept patches are green; dropped patches are colored by the stage that removed
    them. Hovering a patch shows its stage + metadata; the legend toggles each
    layer on/off.

    After `dry_run()` (which runs knobbed filters permissively so every patch
    carries every metric), the report also gets a **slider per tunable parameter**.
    Dragging one re-decides keep/drop for every patch in the browser and updates
    the overlay, the funnel and a copy-pasteable config -- no Python re-run, so it
    keeps working in the saved HTML file too.

    The report is fluid: it fills the width it is given (the notebook cell, the
    browser window) and scales the overlay with it, so it never overflows.

    Args:
        pipeline: a Pipeline that has been audited (`dry_run()` or run with audit=True).
        wsi_path: path to the slide to visualize (must have been in the run).
        thumb_long_side: long side of the background thumbnail in pixels. This is
            its *resolution*, not its display size -- raise it for a sharper image
            on a big screen, lower it for a smaller report file.
        box_scale: scale factor for the drawn tile boxes (visual tuning only).
        save_path: if given, also write a standalone HTML file here.

    Returns:
        An AuditReport (auto-renders in Jupyter; call `.save()` for a file).
    """
    audit = pipeline.get_audit()["by_slide"]
    slide_id = Path(wsi_path).stem
    if slide_id not in audit:
        raise KeyError(
            f"No audit data for '{slide_id}'. Available: {list(audit)}. "
            "Run pipeline.dry_run() or a run with audit=True first."
        )
    entry = audit[slide_id]
    wsi_dims = entry["wsi_dims"]
    if wsi_dims is None:
        raise ValueError("Audit entry has no wsi_dims; nothing to draw.")

    tile_size = int(pipeline.context.get("tile_size", 0))
    if tile_size <= 0:
        raise ValueError("Could not determine tile_size from pipeline context.")

    # Thumbnail (built from level 0; spans the full slide, same extent as wsi_dims).
    W0, H0 = get_dimensions_for_level(wsi_path, level=0)
    scale = float(thumb_long_side) / max(W0, H0)
    thumb = get_thumbnail(wsi_path, (max(1, round(W0 * scale)), max(1, round(H0 * scale))), (W0, H0)).convert("RGB")
    tw, th = thumb.size
    # Overlay geometry is expressed as a percentage of the slide extent, not in
    # thumbnail pixels, so the whole report scales to whatever width it is given.
    slide_w, slide_h = float(wsi_dims[0]), float(wsi_dims[1])
    box_w = tile_size / slide_w * 100.0 * box_scale
    box_h = tile_size / slide_h * 100.0 * box_scale

    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # ---- Build the browser-side data model -------------------------------------
    # Everything the JS needs to re-decide keep/drop for any knob value: the patch
    # grid, each knobbed stage's per-patch metric, and each fixed stage's real drops.
    cand = entry["candidates"]
    coords_t = [(int(x), int(y)) for x, y in cand]
    index_of = {c: i for i, c in enumerate(coords_t)}
    patch_meta = entry["patch_meta"]
    knobs_by_idx = {k["stage_idx"]: k for k in entry["knobs"]}

    metrics: dict = {}
    for col in sorted({k["column"] for k in entry["knobs"]}):
        metrics[col] = [_num(patch_meta.get(c, {}).get(col)) for c in coords_t]

    stages_js = []
    for stage in entry["stage_list"]:
        idx = stage["idx"]
        knob = knobs_by_idx.get(idx)
        knob_js = None
        if knob is not None:
            lo, hi, step, dp = _slider_range(metrics[knob["column"]], float(knob["init"]), knob["integer"])
            knob_js = {
                "param": knob["param"],
                "column": knob["column"],
                "op": knob["op"],
                "init": float(knob["init"]),
                "lo": lo,
                "hi": hi,
                "step": step,
                "dp": dp,
                "integer": bool(knob["integer"]),
            }
        stages_js.append(
            {
                "name": stage["name"],
                "knob": knob_js,
                # A knobbed stage was run permissively, so it dropped nothing in Python.
                "fixed": sorted(index_of[c] for c in entry["py_dropped"].get(idx, set()) if c in index_of),
                # Which patches this stage actually judged. A knobbed stage says the
                # same thing with a null metric, so it needs no list.
                "seen": (
                    None
                    if knob is not None
                    else sorted(index_of[c] for c in entry["seen"].get(idx, set()) if c in index_of)
                ),
            }
        )

    data = {
        "cand": [[int(x), int(y)] for x, y in cand],
        "metrics": metrics,
        "stages": stages_js,
        "tips": [_fmt_meta(patch_meta.get(c, {})) for c in coords_t],
        "palette": _STAGE_PALETTE,
        "kept_color": _KEPT_COLOR,
    }

    # ---- Static markup ---------------------------------------------------------
    # ponytail: one DOM node per patch -- fine at tuning scale; if a slide ever has
    # >~50k patches, subsample the drawn set or throttle recompute on rAF.
    divs = "".join(
        f"<div class='pt' style='left:{x / slide_w * 100:.4f}%;top:{y / slide_h * 100:.4f}%;"
        f"width:{box_w:.4f}%;height:{box_h:.4f}%'></div>"
        for x, y in coords_t
    )
    color_css = f".pt.kept{{background:{_KEPT_COLOR}}}" + "".join(
        f".pt.s{j}{{background:{_STAGE_PALETTE[j % len(_STAGE_PALETTE)]}}}" for j in range(len(stages_js))
    )

    uid = f"wsiaudit_{abs(hash(slide_id)) % 100000}"
    html = (
        _REPORT_HTML.replace("__UID__", uid)
        .replace("__SLIDE__", _html.escape(slide_id))
        .replace("__IMG__", b64)
        .replace("__TW__", str(tw))
        .replace("__TH__", str(th))
        .replace("__DIVS__", divs)
        .replace("__COLORCSS__", color_css)
        .replace("__DATA__", json.dumps(data))
    )
    report = AuditReport(html)
    if save_path is not None:
        report.save(save_path)
    return report


# Self-contained report markup. Its `resolve()` mirrors `audit._resolve` -- walk the
# stages in order, first stage to reject a patch owns the drop -- so keep the two in
# step. On top of that it adds enabling/disabling whole stages, which is report-only.
_REPORT_HTML = """
<div id="__UID__" class="wsiaudit">
  <div class="hdr"><b>__SLIDE__</b></div>
  <div class="knobs"></div>
  <div class="legend"></div>
  <div class="wrap">
    <img src="data:image/png;base64,__IMG__" alt="">
    __DIVS__
  </div>
  <div class="funnel"></div>
  <div class="cfg"></div>
  <div class="tip"></div>
</div>
<style>
  /* Fluid: the report fills whatever width it is given and never overflows the
     cell. All overlay geometry is in % of the slide extent, so it just scales. */
  #__UID__{font-family:system-ui,sans-serif;width:100%;max-width:100%;box-sizing:border-box}
  #__UID__ *{box-sizing:border-box}
  #__UID__ .hdr{font-size:14px;margin-bottom:8px}
  #__UID__ .wrap{position:relative;width:100%;aspect-ratio:__TW__/__TH__;line-height:0}
  #__UID__ .wrap img{display:block;width:100%;height:100%}
  #__UID__ .pt{position:absolute;opacity:.45;border:1px solid #0002}
  #__UID__ .pt.unk{background:#95a5a6}
  #__UID__ __COLORCSS__
  #__UID__ .legend{margin-bottom:8px;font-size:13px}
  #__UID__ .lg{margin-right:12px;white-space:nowrap;display:inline-block}
  #__UID__ .lg label{cursor:pointer}
  #__UID__ .lg.off label{opacity:.45;text-decoration:line-through}
  #__UID__ .sw{display:inline-block;width:11px;height:11px;vertical-align:middle;border:1px solid #0003}
  #__UID__ .knobs{margin-bottom:10px;display:flex;flex-direction:column;gap:3px}
  #__UID__ .knob{display:flex;align-items:center;gap:8px;font-size:13px}
  #__UID__ .knob.off .nm,#__UID__ .knob.off input{opacity:.4}
  #__UID__ .mvs{display:inline-flex;gap:2px;flex:0 0 auto}
  #__UID__ .mv{padding:1px 4px;font-size:9px;line-height:1.2;cursor:pointer;color:inherit;
    border:1px solid #0003;border-radius:3px;background:#0000000d}
  #__UID__ .mv:disabled{opacity:.25;cursor:default}
  /* Name column is sized to the longest label (set as --nmw from JS). It is
     monospace, so 1ch is exact and every slider starts at the same x. */
  #__UID__ .knob .nm{flex:0 0 var(--nmw,260px);font:12px/1.4 ui-monospace,monospace}
  #__UID__ .knob input[type=range]{flex:1 1 160px;min-width:120px;max-width:260px}
  #__UID__ .knob .vl{font:12px ui-monospace,monospace;flex:0 0 76px;text-align:right;
    padding:2px 4px;border:1px solid #0003;border-radius:3px;background:transparent;color:inherit}
  @media (max-width:640px){
    #__UID__ .knob{flex-wrap:wrap}
    #__UID__ .knob .nm{flex-basis:auto}
  }
  /* Wide monospace blocks scroll inside themselves rather than the page. */
  #__UID__ .funnel,#__UID__ .cfg{font:12px/1.5 ui-monospace,monospace;margin-top:10px;
    white-space:pre;overflow-x:auto;max-width:100%}
  #__UID__ .cfg{background:#f6f8fa;color:#111;border:1px solid #0002;border-radius:5px;padding:8px 10px}
  #__UID__ .cfg:empty{display:none}
  #__UID__ .tip{position:fixed;display:none;pointer-events:none;background:#111;color:#fff;
    padding:3px 7px;border-radius:4px;font:12px system-ui;z-index:9999;max-width:340px}
</style>
<script>
(function(){
  var D=__DATA__;
  var root=document.getElementById("__UID__");
  var boxes=root.querySelectorAll(".pt");
  var n=D.cand.length;
  var vals=D.stages.map(function(s){return s.knob?s.knob.init:null;});
  var enabled=D.stages.map(function(){return true;});   // unchecking a stage removes it
  var order=D.stages.map(function(_,i){return i;});      // display/evaluation order
  var knobRows={};
  // Precomputed lookups for the fixed (knob-less) stages.
  var fixedSet=D.stages.map(function(s){return new Set(s.fixed);});
  var seenSet=D.stages.map(function(s){return s.seen?new Set(s.seen):null;});
  var KEPT=-1, UNKNOWN=-2;
  // Reordering filters cannot change which patches survive -- a patch is kept only
  // if it passes all of them. It changes who gets the blame, and how many patches
  // each stage has to process, which is what makes it a speed decision.
  var MOVE_TIP="Move this filter. Changes the funnel (how much work each stage does) "+
    "and which stage gets the blame -- not which patches survive.";
  var owner=new Int32Array(n).fill(KEPT);  // per patch: dropping stage index

  function clsOf(si){return si===KEPT?"kept":(si===UNKNOWN?"unk":"s"+si);}
  function colorOf(si){
    return si===KEPT?D.kept_color:(si===UNKNOWN?"#95a5a6":D.palette[si%D.palette.length]);
  }
  function canToggle(st){return st.knob||st.fixed.length;}

  // Twin of audit._resolve: first stage to reject a patch owns the drop.
  // On top of that, stages can be removed (skipped) or reordered here in the
  // report. Both are safe because every metric is per-patch and independent of
  // where its stage sits: a patch it never measured has no verdict, and shows as
  // "unknown" rather than being silently passed.
  function resolve(){
    owner.fill(KEPT);
    var alive=new Uint8Array(n).fill(1);
    var funnel=[],nIn=n;
    for(var oi=0;oi<order.length;oi++){
      var si=order[oi],st=D.stages[si],drops=0,i;
      if(!enabled[si]){funnel.push([st.name,nIn,nIn,0,true]);continue;}
      if(st.knob){
        var v=vals[si],col=D.metrics[st.knob.column],ge=st.knob.op===">=";
        for(i=0;i<n;i++){
          if(!alive[i])continue;
          var m=col[i];
          if(m===null){alive[i]=0;owner[i]=UNKNOWN;drops++;continue;}
          if(ge?(m<v):(m>v)){alive[i]=0;owner[i]=si;drops++;}
        }
      }else{
        var fs=fixedSet[si],ss=seenSet[si];
        for(i=0;i<n;i++){
          if(!alive[i])continue;
          if(ss&&!ss.has(i)){alive[i]=0;owner[i]=UNKNOWN;drops++;continue;}
          if(fs.has(i)){alive[i]=0;owner[i]=si;drops++;}
        }
      }
      funnel.push([st.name,nIn,nIn-drops,drops,false]);
      nIn-=drops;
    }
    return {funnel:funnel,kept:nIn};
  }

  // A filter may only swap with an adjacent *filter*. Crossing a source or a
  // pixel-transforming stage (a normalizer) would invalidate the metrics we
  // recorded, so those moves are not offered.
  function canMove(si,dir){
    var p=order.indexOf(si),q=p+dir;
    return q>=0 && q<order.length && canToggle(D.stages[order[q]]);
  }
  function move(si,dir){
    var p=order.indexOf(si),q=p+dir,t=order[p];
    order[p]=order[q];order[q]=t;
    render();
  }

  function fmt(si,v){var k=D.stages[si].knob;return k.integer?String(v|0):(+v).toFixed(k.dp);}

  function render(){
    var r=resolve(),i;
    for(i=0;i<n;i++)boxes[i].className="pt "+clsOf(owner[i]);
    var counts={};
    for(i=0;i<n;i++){var k=clsOf(owner[i]);counts[k]=(counts[k]||0)+1;}

    function swatch(si){
      return "<span class='sw' style='background:"+colorOf(si)+"'></span>";
    }
    // Every checkbox means one thing: is this stage in the pipeline? "kept" and
    // "unknown" are outcomes, not stages, so they are swatches with no checkbox.
    // Arrows reorder; the legend lists stages in their current order.
    var html="<span class='lg'>"+swatch(KEPT)+" kept ("+(counts["kept"]||0)+")</span>";
    for(var oi=0;oi<order.length;oi++){
      var si=order[oi];
      if(!canToggle(D.stages[si]))continue;
      html+="<label class='lg"+(enabled[si]?"":" off")+"' title='Uncheck to see the pipeline "+
        "without this stage'><input type='checkbox' data-si='"+si+"' "+(enabled[si]?"checked":"")+"> "+
        swatch(si)+" "+D.stages[si].name+" ("+(counts["s"+si]||0)+")</label>";
    }
    if(counts["unk"]){
      html+="<span class='lg' title='These patches were dropped in Python before the stage that "+
        "would judge them here ever measured them -- re-run with this pipeline to judge them'>"+
        swatch(UNKNOWN)+" unknown ("+counts["unk"]+")</span>";
    }
    root.querySelector(".legend").innerHTML=html;
    root.querySelectorAll(".legend input[data-si]").forEach(function(cb){
      cb.addEventListener("change",function(){
        enabled[+cb.getAttribute("data-si")]=cb.checked;
        render();
      });
    });
    // Filter rows: keep them in pipeline order, grey out removed ones, and only
    // offer a move that stays between filters.
    for(var key in knobRows){
      var kr=knobRows[key],ksi=+key;
      kr.row.className="knob"+(enabled[ksi]?"":" off");
      kr.row.style.order=order.indexOf(ksi);
      if(kr.input){kr.input.disabled=!enabled[ksi];kr.num.disabled=!enabled[ksi];}
      kr.up.disabled=!canMove(ksi,-1);
      kr.down.disabled=!canMove(ksi,1);
    }
    // Funnel
    var pad=function(s,w){s=String(s);while(s.length<w)s=" "+s;return s;};
    var padR=function(s,w){s=String(s);while(s.length<w)s=s+" ";return s;};
    // Size the stage column to the longest name (+ " (off)") so nothing truncates.
    var W=16;
    for(i=0;i<D.stages.length;i++)W=Math.max(W,D.stages[i].name.length+7);
    var rows=r.funnel.map(function(f){
      var pct=f[1]?(100*f[3]/f[1]).toFixed(1):"0.0";
      var nm=padR(f[0]+(f[4]?" (off)":""),W);
      return f[4] ? nm+pad(f[1],7)+pad(f[2],9)+pad("-",9)+pad("-",8)
                  : nm+pad(f[1],7)+pad(f[2],9)+pad(f[3],9)+pad(pct+"%",8);
    });
    root.querySelector(".funnel").textContent=
      padR("Stage",W)+pad("In",7)+pad("Out",9)+pad("Dropped",9)+pad("Drop %",8)+"\\n"+rows.join("\\n");
    // Live config you can paste back: the stages still switched on, in their
    // current order. (Named cfgOrder, not order -- that one is the stage order.)
    var byStage={},cfgOrder=[];
    for(oi=0;oi<order.length;oi++){
      var csi=order[oi];
      if(!D.stages[csi].knob||!enabled[csi])continue;
      var nm2=D.stages[csi].name;
      if(!byStage[nm2]){byStage[nm2]=[];cfgOrder.push(nm2);}
      byStage[nm2].push(D.stages[csi].knob.param+"="+fmt(csi,vals[csi]));
    }
    root.querySelector(".cfg").textContent=cfgOrder.length
      ? cfgOrder.map(function(nm){return ".then("+nm+"("+byStage[nm].join(", ")+"))";}).join("\\n")
      : "";
  }

  // One row per filter, in pipeline order: move arrows, then its slider if it has
  // one. Filters with no tunable parameter still get a row so they can be moved.
  var kb=root.querySelector(".knobs");
  function label(st){return st.name+(st.knob?"."+st.knob.param+" "+st.knob.op:"");}
  // Size the name column to the longest label so the sliders line up.
  var nmw=0;
  D.stages.forEach(function(st){if(canToggle(st))nmw=Math.max(nmw,label(st).length);});
  kb.style.setProperty("--nmw",(nmw+1)+"ch");
  D.stages.forEach(function(st,si){
    if(!canToggle(st))return;
    var row=document.createElement("div");
    row.className="knob";
    var h="<span class='mvs'><button class='mv' data-dir='-1' title='"+MOVE_TIP+"'>&#9650;</button>"+
          "<button class='mv' data-dir='1' title='"+MOVE_TIP+"'>&#9660;</button></span>"+
          "<span class='nm'>"+label(st)+"</span>";
    if(st.knob){
      h+="<input type='range' min='"+st.knob.lo+"' max='"+st.knob.hi+"' step='"+st.knob.step+
         "' value='"+st.knob.init+"'>"+
         "<input type='number' class='vl' step='"+st.knob.step+"' value='"+fmt(si,st.knob.init)+
         "' title='Type an exact value'>";
    }
    row.innerHTML=h;
    var sl=row.querySelector("input[type=range]");
    var num=row.querySelector("input[type=number]");
    if(sl){
      sl.addEventListener("input",function(){
        vals[si]=parseFloat(sl.value);
        num.value=fmt(si,vals[si]);
        render();
      });
      num.addEventListener("input",function(){
        var v=parseFloat(num.value);
        if(isNaN(v))return;   // mid-typing ("", "0.", "-"): wait for a real number
        vals[si]=v;
        sl.value=v;           // the range clamps its handle; the typed value stands
        render();
      });
      // Tidy the formatting once they are done, without fighting them while typing.
      num.addEventListener("change",function(){
        if(!isNaN(parseFloat(num.value)))num.value=fmt(si,vals[si]);
      });
    }
    var mv=row.querySelectorAll(".mv");
    mv[0].addEventListener("click",function(){move(si,-1);});
    mv[1].addEventListener("click",function(){move(si,1);});
    knobRows[si]={row:row,input:sl,num:num,up:mv[0],down:mv[1]};
    kb.appendChild(row);
  });

  // Tooltip
  var tip=root.querySelector(".tip"),wrap=root.querySelector(".wrap");
  wrap.addEventListener("mousemove",function(e){
    var t=e.target;
    if(t.classList.contains("pt")){
      var i=Array.prototype.indexOf.call(boxes,t);
      var si=owner[i];
      var who=si===KEPT?"kept":(si===UNKNOWN?"unknown (never measured after the stage you disabled)"
                                            :"dropped by "+D.stages[si].name);
      tip.textContent=D.tips[i]?who+" \\u2014 "+D.tips[i]:who;
      tip.style.display="block";
      tip.style.left=(e.clientX+12)+"px";
      tip.style.top=(e.clientY+12)+"px";
    }else{tip.style.display="none";}
  });
  wrap.addEventListener("mouseleave",function(){tip.style.display="none";});

  render();
})();
</script>
"""
