"""
Tests for the fastslide-based slide backend (cucim_openslide_isyntax module).

All slide I/O is mocked via a FakeFastSlide that mirrors the fastslide.FastSlide API,
so no real WSI files are required.
"""
import numpy as np
import pytest

import wsi_patching.backends.cucim_openslide_isyntax as mod


class FakeFastSlideImage:
    """Mimics the image object returned by FastSlide.read_region()."""

    def __init__(self, w, h, fill=7):
        self._arr = np.full((h, w, 3), fill, dtype=np.uint8)

    def numpy(self):
        return self._arr


class FakeFastSlide:
    """
    Fake fastslide.FastSlide object.

    Mimics the attributes used by all backend functions:
      level_count, level_dimensions, level_downsamples, mpp,
      read_region(), convert_level0_to_level_native(), close()
      and context-manager support.
    """

    def __init__(self, path):
        self.path = path
        self.level_dimensions = [(300, 200), (120, 80), (60, 40)]
        self.level_downsamples = [1.0, 2.0, 4.0]
        self.level_count = len(self.level_dimensions)
        self.mpp = (0.5, 0.5)  # (mpp_x, mpp_y) tuple at level 0
        self.calls = []

    def read_region(self, *, location, level, size):
        self.calls.append({"location": location, "level": level, "size": size})
        w, h = size
        return FakeFastSlideImage(w, h, fill=7)

    def convert_level0_to_level_native(self, x, y, level):
        # Mirror fastslide semantics: divide level-0 coords by the downsample factor.
        # round() matches integer rounding for non-integer downsample factors.
        ds = self.level_downsamples[level]
        return round(x / ds), round(y / ds)

    def close(self):
        pass

    # Context-manager support
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class FakeFastSlideNoMPP(FakeFastSlide):
    """Same as FakeFastSlide but without mpp metadata."""

    def __init__(self, path):
        super().__init__(path)
        self.mpp = None  # mpp unavailable


# ---------- helpers ----------


def _make_ctor(cls=FakeFastSlide):
    """Return a _open_slide replacement that constructs the given fake class."""

    def _ctor(path):
        return cls(path)

    return _ctor


# ---------- validate_slide_backend ----------


def test_validate_slide_backend_is_noop():
    """validate_slide_backend should not raise for any value of use_gpu."""
    mod.validate_slide_backend(use_gpu=True)
    mod.validate_slide_backend(use_gpu=False)


# ---------- get_dimensions_for_level ----------


def test_get_dimensions_for_level(monkeypatch):
    monkeypatch.setattr(mod, "_open_slide", _make_ctor(), raising=True)

    W, H = mod.get_dimensions_for_level("dummy.svs", level=1)
    assert (W, H) == (120, 80)
    assert isinstance(W, int) and isinstance(H, int)


def test_get_dimensions_for_level_invalid_level_raises(monkeypatch):
    monkeypatch.setattr(mod, "_open_slide", _make_ctor(), raising=True)

    with pytest.raises(AssertionError):
        mod.get_dimensions_for_level("dummy.svs", level=10)


# ---------- get_level_downsamples ----------


def test_get_level_downsamples(monkeypatch):
    monkeypatch.setattr(mod, "_open_slide", _make_ctor(), raising=True)

    ds = mod.get_level_downsamples("dummy.svs")
    assert ds == [1.0, 2.0, 4.0]
    assert all(isinstance(v, float) for v in ds)


# ---------- get_level_for_resolution: unit == "level" ----------


def test_get_level_for_resolution_unit_level(monkeypatch):
    monkeypatch.setattr(mod, "_open_slide", _make_ctor(), raising=True)

    assert mod.get_level_for_resolution(path="dummy.svs", resolution=1, unit="level", fallback_mode="nearest") == 1

    with pytest.raises(ValueError):
        mod.get_level_for_resolution(path="dummy.svs", resolution=1.5, unit="level", fallback_mode="nearest")

    with pytest.raises(ValueError):
        mod.get_level_for_resolution(path="dummy.svs", resolution=-1, unit="level", fallback_mode="nearest")

    with pytest.raises(ValueError):
        mod.get_level_for_resolution(path="dummy.svs", resolution=99, unit="level", fallback_mode="nearest")


# ---------- get_level_for_resolution: downsample fallback modes ----------


def test_get_level_for_resolution_downsample_fallback_modes(monkeypatch):
    monkeypatch.setattr(mod, "_open_slide", _make_ctor(), raising=True)

    # level_downsamples = [1.0, 2.0, 4.0]

    # nearest: requested 1.3 → nearest to 1.0 vs 2.0 vs 4.0 is 1.0 (idx 0)
    assert (
        mod.get_level_for_resolution(path="dummy.svs", resolution=1.3, unit="downsample", fallback_mode="nearest") == 0
    )

    # floor: coarsest with value >= requested
    # requested 1.1 → eligible 2.0, 4.0 → choose 2.0 (idx 1)
    assert mod.get_level_for_resolution(path="dummy.svs", resolution=1.1, unit="downsample", fallback_mode="floor") == 1

    # floor: requested too large (e.g. 10) → fallback to coarsest (idx 2)
    assert (
        mod.get_level_for_resolution(path="dummy.svs", resolution=10.0, unit="downsample", fallback_mode="floor") == 2
    )

    # ceil: finest with value <= requested
    # requested 1.3 → eligible 1.0 → idx 0
    assert mod.get_level_for_resolution(path="dummy.svs", resolution=1.3, unit="downsample", fallback_mode="ceil") == 0
    # ceil: requested smaller than all (e.g. 0.1) → fallback to level 0
    assert mod.get_level_for_resolution(path="dummy.svs", resolution=0.1, unit="downsample", fallback_mode="ceil") == 0

    # error: exact match required
    assert mod.get_level_for_resolution(path="dummy.svs", resolution=4.0, unit="downsample", fallback_mode="error") == 2
    with pytest.raises(ValueError):
        mod.get_level_for_resolution(path="dummy.svs", resolution=3.0, unit="downsample", fallback_mode="error")


# ---------- get_level_for_resolution: mpp ----------


def test_get_level_for_resolution_mpp_requires_metadata(monkeypatch):
    monkeypatch.setattr(mod, "_open_slide", _make_ctor(FakeFastSlideNoMPP), raising=True)

    with pytest.raises(ValueError):
        mod.get_level_for_resolution(path="dummy.svs", resolution=0.5, unit="mpp", fallback_mode="nearest")


def test_get_level_for_resolution_mpp_nearest(monkeypatch):
    monkeypatch.setattr(mod, "_open_slide", _make_ctor(), raising=True)

    # FakeFastSlide: mpp=0.5, level_downsamples [1.0, 2.0, 4.0]
    # values = [0.5, 1.0, 2.0]
    # request 0.8 → nearest is 1.0 (idx 1)
    assert mod.get_level_for_resolution(path="dummy.svs", resolution=0.8, unit="mpp", fallback_mode="nearest") == 1


def test_get_level_for_resolution_resample_mode(monkeypatch):
    monkeypatch.setattr(mod, "_open_slide", _make_ctor(), raising=True)

    # FakeFastSlide: mpp=0.5, level_downsamples [1.0, 2.0, 4.0]
    # values = [0.5, 1.0, 2.0]
    assert mod.get_level_for_resolution(path="dummy.svs", resolution=0.5, unit="mpp", fallback_mode="resample") == 0
    assert mod.get_level_for_resolution(path="dummy.svs", resolution=0.8, unit="mpp", fallback_mode="resample") == 0
    assert mod.get_level_for_resolution(path="dummy.svs", resolution=1.1, unit="mpp", fallback_mode="resample") == 1
    assert mod.get_level_for_resolution(path="dummy.svs", resolution=0.3, unit="mpp", fallback_mode="resample") == 0


def test_get_level_for_resolution_resample_downsample(monkeypatch):
    monkeypatch.setattr(mod, "_open_slide", _make_ctor(), raising=True)

    # level_downsamples = [1.0, 2.0, 4.0]
    # request downsample 3.0 → eligible [1.0, 2.0] → level with max value = 2.0 → level 1
    assert (
        mod.get_level_for_resolution(path="dummy.svs", resolution=3.0, unit="downsample", fallback_mode="resample")
        == 1
    )


# ---------- get_resample_factor ----------


def test_get_resample_factor_mpp(monkeypatch):
    monkeypatch.setattr(mod, "_open_slide", _make_ctor(), raising=True)

    # FakeFastSlide: mpp=0.5, level_downsamples [1.0, 2.0, 4.0]
    # level 0: actual_mpp = 0.5*1.0 = 0.5, requested = 1.0 → factor = 1.0/0.5 = 2.0
    assert mod.get_resample_factor("dummy.svs", level=0, resolution=1.0, unit="mpp") == pytest.approx(2.0)

    # level 0: requested = 0.5 → factor = 0.5/0.5 = 1.0 (exact match)
    assert mod.get_resample_factor("dummy.svs", level=0, resolution=0.5, unit="mpp") == pytest.approx(1.0)

    # level 0: requested = 0.25 → factor = 0.25/0.5 = 0.5; clamped to 1.0 (no upsampling)
    assert mod.get_resample_factor("dummy.svs", level=0, resolution=0.25, unit="mpp") == pytest.approx(1.0)

    # level 1: actual_mpp = 0.5*2.0 = 1.0, requested = 2.0 → factor = 2.0/1.0 = 2.0
    assert mod.get_resample_factor("dummy.svs", level=1, resolution=2.0, unit="mpp") == pytest.approx(2.0)


def test_get_resample_factor_downsample(monkeypatch):
    monkeypatch.setattr(mod, "_open_slide", _make_ctor(), raising=True)

    assert mod.get_resample_factor("dummy.svs", level=0, resolution=2.0, unit="downsample") == pytest.approx(2.0)
    assert mod.get_resample_factor("dummy.svs", level=1, resolution=2.0, unit="downsample") == pytest.approx(1.0)


def test_get_resample_factor_unsupported_unit(monkeypatch):
    monkeypatch.setattr(mod, "_open_slide", _make_ctor(), raising=True)

    with pytest.raises(ValueError, match="unit"):
        mod.get_resample_factor("dummy.svs", level=0, resolution=0, unit="level")


# ---------- read_region ----------


def test_read_region_returns_numpy_rgb(monkeypatch):
    """read_region should return a (h, w, 3) uint8 NumPy array."""
    fake = FakeFastSlide("slide.svs")
    monkeypatch.setattr(mod, "_open_slide", lambda path: fake, raising=True)

    arr = mod.read_region(path="slide.svs", x=10, y=20, w=7, h=9, level=0, use_gpu=False)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (9, 7, 3)
    assert arr.dtype == np.uint8
    assert int(arr[0, 0, 0]) == 7  # from FakeFastSlideImage fill value


def test_read_region_converts_level0_to_level_native(monkeypatch):
    """read_region must convert level-0 coordinates to level-native before calling fastslide."""
    fake = FakeFastSlide("slide.svs")
    monkeypatch.setattr(mod, "_open_slide", lambda path: fake, raising=True)

    # level 1 has downsample 2.0, so level-native = level-0 / 2
    mod.read_region(path="slide.svs", x=100, y=200, w=4, h=6, level=1, use_gpu=False)

    assert len(fake.calls) == 1
    call = fake.calls[0]
    # x_native = round(100 / 2.0) = 50, y_native = round(200 / 2.0) = 100
    assert call["location"] == (50, 100)
    assert call["level"] == 1
    assert call["size"] == (4, 6)


def test_read_region_use_gpu_and_num_workers_are_accepted(monkeypatch):
    """use_gpu and num_workers_cucim params must be accepted (ignored) without error."""
    fake = FakeFastSlide("slide.svs")
    monkeypatch.setattr(mod, "_open_slide", lambda path: fake, raising=True)

    arr = mod.read_region(path="slide.svs", x=0, y=0, w=4, h=4, level=0, use_gpu=True, num_workers_cucim=16)
    assert isinstance(arr, np.ndarray)
