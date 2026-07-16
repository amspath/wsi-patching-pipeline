from typing import Iterable, Sequence, Tuple

from wsi_patching.backends.cupy_numpy import ensure_numpy, get_xp_backend, validate_xp_backend
from wsi_patching.core.pipeline import Stage
from wsi_patching.core.types.types import CollatedPatchBatch
from wsi_patching.utils.audit import Knob


class PenArtifactFilter(Stage):
    """
    Pure-xp (NumPy/CuPy) pen artifact filter using bitset LUTs.

    Assumptions:
      - Input `patches` are uint8, or float32 scaled in [0, 255].
      - If ctx['use_gpu'] is True, `patches` is a CuPy array; otherwise NumPy.
      - All intermediate/results stay on the active xp backend; host conversion is done
        only where the pipeline explicitly expects NumPy (via ensure_numpy).

    Method:
      - For each color "mode" (red/green/blue), build 3 LUTs (R/G/B). Each LUT (Lookup Table) cell
        contains a bitmask where each bit represents one (r,g,b) threshold tuple.
      - For a pixel, we OR the three channel lookups; if all bits are set (== full_mask),
        the pixel *passes* that mode. A pixel is kept only if it passes all 3 modes,
        or if it is very dark: (R+G+B) < 3*diff_thresh.
      - A patch is dropped when its fraction of *pen* pixels (i.e., pixels not kept) exceeds
        `max_pen_fraction`.
    """

    # `pen_fraction` does not depend on `max_pen_fraction`, so the audit report can
    # re-decide keep/drop for any threshold without re-running. `diff_thresh` feeds
    # the metric itself, so it gets no slider.
    AUDIT_KNOBS = (Knob("max_pen_fraction", "pen_fraction", "<=", permissive=1.0),)

    # Hard-coded threshold parameter sets (ported from original histolab logic)
    _BLUE_PARAMS: Sequence[Tuple[int, int, int]] = [
        (60, 120, 190),
        (120, 170, 200),
        (175, 210, 230),
        (145, 180, 210),
        (37, 95, 160),
        (30, 65, 130),
        (130, 155, 180),
        (40, 35, 85),
        (30, 20, 65),
        (90, 90, 140),
        (60, 60, 120),
        (110, 110, 175),
    ]
    _GREEN_PARAMS: Sequence[Tuple[int, int, int]] = [
        (150, 160, 140),
        (70, 110, 110),
        (45, 115, 100),
        (30, 75, 60),
        (195, 220, 210),
        (225, 230, 225),
        (170, 210, 200),
        (20, 30, 20),
        (50, 60, 40),
        (30, 50, 35),
        (65, 70, 60),
        (100, 110, 105),
        (165, 180, 180),
        (140, 140, 150),
        (185, 195, 195),
    ]
    _RED_PARAMS: Sequence[Tuple[int, int, int]] = [
        (150, 80, 90),
        (110, 20, 30),
        (185, 65, 105),
        (195, 85, 125),
        (220, 115, 145),
        (125, 40, 70),
        (100, 50, 65),
        (85, 25, 45),
    ]

    def __init__(self, diff_thresh: float = 5.0, max_pen_fraction: float = 0.01):
        # Darkness guard and batch-level drop threshold
        self.diff_thresh = float(diff_thresh)
        self.max_pen_fraction = float(max_pen_fraction)

        # Mode-specific LUTs
        self._luts_red = None
        self._luts_green = None
        self._luts_blue = None

    def validate(self):
        """Resolve backend (NumPy/CuPy) and build LUTs once on that backend."""
        self.ctx.require_key("use_gpu")
        use_gpu = bool(self.ctx["use_gpu"])
        if use_gpu:
            # Fail fast if GPU path requested but CuPy unavailable
            validate_xp_backend(True)

        self._build_all_luts_xp(get_xp_backend(self.ctx["use_gpu"]))

    def __call__(self, it: Iterable[CollatedPatchBatch]) -> Iterable[CollatedPatchBatch]:
        """
        For each batch:
          - Ensure dtype uint8 (accept float32 in [0,255] and cast).
          - Compute per-pixel keep mask from LUTs + darkness guard.
          - Derive per-patch pen_fraction; drop patches exceeding `max_pen_fraction`.
          - Attach stats and yield the filtered batch.
        """
        use_gpu = bool(self.ctx["use_gpu"])
        xp = get_xp_backend(use_gpu)
        thr3 = xp.float32(3.0 * self.diff_thresh)

        for collated_patch_batch in it:
            in_sz = len(collated_patch_batch.patches)

            # Normalize dtype: prefer uint8 for LUT indexing
            if collated_patch_batch.patches.dtype != xp.uint8:
                collated_patch_batch.patches = collated_patch_batch.patches.astype(xp.uint8, copy=False)

            # Split channels; shapes: [B,H,W]
            R = collated_patch_batch.patches[..., 0]
            G = collated_patch_batch.patches[..., 1]
            B = collated_patch_batch.patches[..., 2]

            # A pixel is kept if it passes all three modes
            keep = (
                self._mode_pass_xp(R, G, B, self._luts_red)
                & self._mode_pass_xp(R, G, B, self._luts_green)
                & self._mode_pass_xp(R, G, B, self._luts_blue)
            )

            # Darkness guard: keep very dark pixels even if they fail LUTs
            sum_rgb = (R.astype(xp.uint16) + G.astype(xp.uint16) + B.astype(xp.uint16)).astype(xp.float32)
            keep_final = xp.where(keep, True, sum_rgb < thr3)

            # Pen mask is the inverse of keep_final
            pen = xp.logical_not(keep_final)  # [B,H,W] bool

            # Per-patch stats
            Bsz = pen.shape[0]
            pen_pixel_count = pen.reshape(Bsz, -1).sum(axis=1)  # xp integer (per patch)
            total_pixels = pen.shape[1] * pen.shape[2]
            pen_fraction = pen_pixel_count.astype(xp.float32) / float(total_pixels)

            # Drop patches with too much pen content
            keep_batch_mask = pen_fraction <= self.max_pen_fraction

            # Pipeline columns expect NumPy; convert explicitly at the boundary.
            collated_patch_batch.add_meta_column("pen_fraction", ensure_numpy(pen_fraction))
            collated_patch_batch.add_meta_column("pen_pixel_count", ensure_numpy(pen_pixel_count))
            collated_patch_batch.filter_on_mask(ensure_numpy(keep_batch_mask))

            self.log.info(
                f"batch_in={in_sz} batch_out={len(collated_patch_batch.patches)} "
                f"(max_pen_fraction={self.max_pen_fraction})"
            )
            yield collated_patch_batch

    # -------- helpers (xp-native) --------
    def _build_all_luts_xp(self, xp):
        """Build LUT triplets for each mode directly on the active xp backend."""
        self._luts_red = self._build_mode_luts_xp(self._RED_PARAMS, "red", xp)
        self._luts_green = self._build_mode_luts_xp(self._GREEN_PARAMS, "green", xp)
        self._luts_blue = self._build_mode_luts_xp(self._BLUE_PARAMS, "blue", xp)

    @staticmethod
    def _build_mode_luts_xp(params: Sequence[Tuple[int, int, int]], mode: str, xp):
        """
        Build per-channel LUTs for a mode as bitsets:
          - Each LUT is length-256 (one entry per 8-bit channel value).
          - Each bit in the uint32 entry corresponds to one (r,g,b) threshold tuple.
          - For a pixel/channel value, set a bit if the tuple’s condition is met.
        """
        P = len(params)
        if P > 32:
            raise ValueError(f"Bitset LUT supports up to 32 params per mode (got {P}).")

        full_mask = xp.uint32((1 << P) - 1)  # mask with all P bits set
        lutR = xp.zeros(256, dtype=xp.uint32)
        lutG = xp.zeros(256, dtype=xp.uint32)
        lutB = xp.zeros(256, dtype=xp.uint32)

        for i, (r, g, b) in enumerate(params):
            bit = xp.uint32(1 << i)

            # Set bits for *ranges* that satisfy the tuple condition on each channel.
            # See histolab logic: "red", "green", and "blue" modes differ by which side
            # of the threshold is considered passing for each channel.
            if mode == "red":
                if r > 0:
                    lutR[:r] |= bit  # R < r  => pass this tuple
                if g < 255:
                    lutG[g + 1 :] |= bit  # G > g  => pass
                if b < 255:
                    lutB[b + 1 :] |= bit  # B > b  => pass
            elif mode == "green":
                if r < 255:
                    lutR[r + 1 :] |= bit  # R > r
                if g > 0:
                    lutG[:g] |= bit  # G < g
                if b > 0:
                    lutB[:b] |= bit  # B < b
            elif mode == "blue":
                if r < 255:
                    lutR[r + 1 :] |= bit  # R > r
                if g < 255:
                    lutG[g + 1 :] |= bit  # G > g
                if b > 0:
                    lutB[:b] |= bit  # B < b
            else:
                raise ValueError(mode)

        return (lutR, lutG, lutB, full_mask)

    @staticmethod
    def _mode_pass_xp(R, G, B, luts):
        """
        For a given mode, return a boolean mask where each pixel passes that mode.
        Implementation:
          - Gather bitsets from LUTs for the pixel's R/G/B values.
          - OR the three bitsets; a bit is set if any channel satisfies that tuple.
          - Pixel passes the mode if *all* bits are set (i.e., equals full_mask).
        """
        lutR, lutG, lutB, full_mask = luts
        m = lutR[R] | lutG[G] | lutB[B]
        return m == full_mask
