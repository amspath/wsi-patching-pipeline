from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import openslide
from PIL import Image, ImageDraw

CoordLike = Union[Tuple[int, int], Sequence[int], np.ndarray]


def visualize_selected_patches(
    wsi_path: Union[str, Path],
    coords: Iterable[CoordLike],
    patch_size: int,
    *,
    stride: Optional[int] = None,
    patch_images: np.ndarray = None,
    thumb_long_side: int = 2000,
    selected_outline_rgba: Tuple[int, int, int, int] = (0, 220, 0, 100),
    save_path: Union[str, Path, None] = None,
) -> Image:
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
        show_unselected: draw dark-gray fill for unselected tiles in bounding region.
        save_path: if provided, save the output PNG to this path.

    Returns:
        PIL Image object of the composite visualization.
    """
    if stride is None:
        stride = patch_size
    if stride <= 0 or patch_size <= 0:
        raise ValueError("`stride` and `patch_size` must be positive integers.")

    wsi_path = Path(wsi_path)

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
                coord_to_patch[(int(k[0]), int(k[1]))] = np.asarray(v)
        else:
            # assume same order as coords
            if len(patch_images) != len(coords_arr):
                raise ValueError("If `patch_images` is a sequence, it must match len(coords).")
            for xy, img in zip(coords_arr, patch_images):
                coord_to_patch[xy] = np.asarray(img)

    # Open slide and make thumbnail
    slide = openslide.OpenSlide(str(wsi_path))
    W0, H0 = slide.level_dimensions[0]
    scale = float(thumb_long_side) / max(W0, H0)
    tw = max(1, int(round(W0 * scale)))
    th = max(1, int(round(H0 * scale)))
    base_thumb = slide.get_thumbnail((tw, th)).convert("RGB")
    slide.close()

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

            patch_img = _to_pil_rgb(np_patch)

            # resize patch to thumbnail scale
            dst_w = max(1, int(round(patch_size * sx)))
            dst_h = max(1, int(round(patch_size * sy)))
            if patch_img.size != (dst_w, dst_h):
                patch_img = patch_img.resize((dst_w, dst_h), resample=Image.BILINEAR)

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
    composite.convert("RGB")

    if save_path is None:
        return composite

    save_path = Path(save_path)
    composite.convert("RGB").save(save_path, format="PNG")
    return composite


def _to_pil_rgb(img: np.ndarray) -> Image.Image:
    """
    Convert a numpy patch to a PIL RGB image.
    Accepts:
      - HWC (uint8 or float)
      - CHW (uint8 or float)
      - Grayscale (H, W) or (H, W, 1) or (1, H, W)
    Float images are scaled to [0, 255] via min-max per patch (robust to negative/>1 values).
    """
    a = np.asarray(img)

    # Move channel last if CHW
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[2] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))  # CHW -> HWC

    # Squeeze singleton channel
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]

    # Convert grayscale to RGB
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    elif a.ndim == 3 and a.shape[2] == 3:
        pass
    else:
        # Unknown shape, try to coerce
        a = np.squeeze(a)
        if a.ndim == 2:
            a = np.stack([a, a, a], axis=-1)
        elif a.ndim == 3 and a.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Unsupported patch array shape after processing: {a.shape}")

    return Image.fromarray(a)
