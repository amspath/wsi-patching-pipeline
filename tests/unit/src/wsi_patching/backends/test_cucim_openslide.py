import numpy as np
import pytest

import wsi_patching.backends.cucim_openslide_isyntax as mod


class FakeCuImage:
    def __init__(self, path):
        self.path = path
        self.calls = []
        # cuCIM exposes dimensions via .resolutions["level_dimensions"]
        self.resolutions = {"level_dimensions": [(100, 50), (25, 10)]}

    def read_region(self, *, location, size, level, num_workers):
        # record the call so we can assert parameters
        self.calls.append({"location": location, "size": size, "level": level, "num_workers": num_workers})
        w, h = size
        return np.full((h, w, 3), 7, dtype=np.uint8)

    def close(self):
        pass


class FakePILImage:
    """Minimal object with .convert('RGB') returning an array-compatible object."""

    def __init__(self, w, h, fill=3):
        self.w, self.h, self.fill = w, h, fill

    def convert(self, mode):
        assert mode == "RGB"
        return np.full((self.h, self.w, 3), self.fill, dtype=np.uint8)


class FakeOpenSlide:
    """
    Fake OpenSlide object.

    It mimics the attributes used by:
      - get_dimensions_for_level  (level_dimensions, level_count)
      - get_level_for_resolution (level_downsamples, properties, level_count)
      - read_region              (for CPU path in read_region)
    """

    def __init__(self, path):
        self.path = path
        self.level_dimensions = [(300, 200), (120, 80), (60, 40)]
        self.level_downsamples = [1.0, 2.0, 4.0]
        self.level_count = len(self.level_dimensions)
        # mpp at level 0
        self.properties = {"openslide.mpp-x": "0.5"}

    def read_region(self, location, level, size):
        # location: (x, y), level: int, size: (w, h)
        w, h = size
        return FakePILImage(w, h, fill=5)

    def close(self):
        pass


class FakeOpenSlideNoMPP(FakeOpenSlide):
    """Same as FakeOpenSlide, but without mpp metadata."""

    def __init__(self, path):
        super().__init__(path)
        self.properties = {}  # no "openslide.mpp-x"


# ---------- tests ----------
def test_validate_slide_backend_raises_when_gpu_requested_but_cucim_missing(monkeypatch):
    monkeypatch.setattr(mod, "_cucim_available", False, raising=True)
    with pytest.raises(ImportError):
        mod.validate_slide_backend(use_gpu=True)

    # CPU path never raises on missing cuCIM
    mod.validate_slide_backend(use_gpu=False)


def test_get_dimensions_for_level(monkeypatch):
    # Force get_dimensions_for_level to use our fake OpenSlide
    def _ctor(path, use_gpu=False):
        assert path == "dummy.svs"
        return FakeOpenSlide(path)

    monkeypatch.setattr(mod, "_open_slide", _ctor, raising=True)

    W, H = mod.get_dimensions_for_level("dummy.svs", level=1)
    assert (W, H) == (120, 80)  # from FakeOpenSlide.level_dimensions[1]
    assert isinstance(W, int) and isinstance(H, int)


def test_get_dimensions_for_level_invalid_level_raises(monkeypatch):
    def _ctor(path, use_gpu=False):
        return FakeOpenSlide(path)

    monkeypatch.setattr(mod, "_open_slide", _ctor, raising=True)

    # level too high
    with pytest.raises(AssertionError):
        mod.get_dimensions_for_level("dummy.svs", level=10)


def test_get_level_for_resolution_unit_level(monkeypatch):
    def _ctor(path, use_gpu=False):
        return FakeOpenSlide(path)

    monkeypatch.setattr(mod, "_open_slide", _ctor, raising=True)

    # valid integer level
    assert mod.get_level_for_resolution(path="dummy.svs", resolution=1, unit="level", fallback_mode="nearest") == 1

    # non-integer resolution should fail for unit="level"
    with pytest.raises(ValueError):
        mod.get_level_for_resolution(path="dummy.svs", resolution=1.5, unit="level", fallback_mode="nearest")

    # negative or out-of-range levels should fail
    with pytest.raises(ValueError):
        mod.get_level_for_resolution(path="dummy.svs", resolution=-1, unit="level", fallback_mode="nearest")

    with pytest.raises(ValueError):
        mod.get_level_for_resolution(path="dummy.svs", resolution=99, unit="level", fallback_mode="nearest")


def test_get_level_for_resolution_downsample_fallback_modes(monkeypatch):
    def _ctor(path, use_gpu=False):
        return FakeOpenSlide(path)

    monkeypatch.setattr(mod, "_open_slide", _ctor, raising=True)

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


def test_get_level_for_resolution_mpp_requires_metadata(monkeypatch):
    def _ctor(path, use_gpu=False):
        return FakeOpenSlideNoMPP(path)

    monkeypatch.setattr(mod, "_open_slide", _ctor, raising=True)

    with pytest.raises(ValueError):
        mod.get_level_for_resolution(path="dummy.svs", resolution=0.5, unit="mpp", fallback_mode="nearest")


def test_get_level_for_resolution_mpp_nearest(monkeypatch):
    def _ctor(path, use_gpu=False):
        return FakeOpenSlide(path)

    monkeypatch.setattr(mod, "_open_slide", _ctor, raising=True)

    # FakeOpenSlide: mpp0 = 0.5, level_downsamples [1.0, 2.0, 4.0]
    # values = [0.5, 1.0, 2.0]
    # request 0.8 → nearest is 1.0 (idx 1)
    assert mod.get_level_for_resolution(path="dummy.svs", resolution=0.8, unit="mpp", fallback_mode="nearest") == 1


def test_get_level_for_resolution_resample_mode(monkeypatch):
    def _ctor(path, use_gpu=False):
        return FakeOpenSlide(path)

    monkeypatch.setattr(mod, "_open_slide", _ctor, raising=True)

    # FakeOpenSlide: mpp0 = 0.5, level_downsamples [1.0, 2.0, 4.0]
    # values = [0.5, 1.0, 2.0]
    # resample: finest level with mpp <= requested
    # request 0.5 → eligible [0.5] → level 0 (exact match, factor 1.0)
    assert mod.get_level_for_resolution(path="dummy.svs", resolution=0.5, unit="mpp", fallback_mode="resample") == 0

    # request 0.8 → eligible [0.5] → level 0 (finest with mpp <= 0.8)
    assert mod.get_level_for_resolution(path="dummy.svs", resolution=0.8, unit="mpp", fallback_mode="resample") == 0

    # request 1.1 → eligible [0.5, 1.0] → level with max eligible mpp = 1.0 → level 1
    assert mod.get_level_for_resolution(path="dummy.svs", resolution=1.1, unit="mpp", fallback_mode="resample") == 1

    # request 0.3 → no eligible levels (all mpp > 0.3) → fallback to level 0
    assert mod.get_level_for_resolution(path="dummy.svs", resolution=0.3, unit="mpp", fallback_mode="resample") == 0


def test_get_level_for_resolution_resample_downsample(monkeypatch):
    def _ctor(path, use_gpu=False):
        return FakeOpenSlide(path)

    monkeypatch.setattr(mod, "_open_slide", _ctor, raising=True)

    # level_downsamples = [1.0, 2.0, 4.0]
    # request downsample 3.0 → eligible [1.0, 2.0] → level with max value = 2.0 → level 1
    assert (
        mod.get_level_for_resolution(path="dummy.svs", resolution=3.0, unit="downsample", fallback_mode="resample")
        == 1
    )


def test_get_resample_factor_mpp(monkeypatch):
    def _ctor(path, use_gpu=False):
        return FakeOpenSlide(path)

    monkeypatch.setattr(mod, "_open_slide", _ctor, raising=True)

    # FakeOpenSlide: mpp0 = 0.5, level_downsamples [1.0, 2.0, 4.0]
    # level 0: actual_mpp = 0.5*1.0 = 0.5, requested = 1.0 → factor = 1.0/0.5 = 2.0
    assert mod.get_resample_factor("dummy.svs", level=0, resolution=1.0, unit="mpp") == pytest.approx(2.0)

    # level 0: requested = 0.5 → factor = 0.5/0.5 = 1.0 (exact match, no resampling)
    assert mod.get_resample_factor("dummy.svs", level=0, resolution=0.5, unit="mpp") == pytest.approx(1.0)

    # level 0: requested = 0.25 → factor = 0.25/0.5 = 0.5; clamped to 1.0 (no upsampling)
    assert mod.get_resample_factor("dummy.svs", level=0, resolution=0.25, unit="mpp") == pytest.approx(1.0)

    # level 1: actual_mpp = 0.5*2.0 = 1.0, requested = 2.0 → factor = 2.0/1.0 = 2.0
    assert mod.get_resample_factor("dummy.svs", level=1, resolution=2.0, unit="mpp") == pytest.approx(2.0)


def test_get_resample_factor_downsample(monkeypatch):
    def _ctor(path, use_gpu=False):
        return FakeOpenSlide(path)

    monkeypatch.setattr(mod, "_open_slide", _ctor, raising=True)

    # level 0: actual_ds = 1.0, requested = 2.0 → factor = 2.0/1.0 = 2.0
    assert mod.get_resample_factor("dummy.svs", level=0, resolution=2.0, unit="downsample") == pytest.approx(2.0)

    # level 1: actual_ds = 2.0, requested = 2.0 → factor = 1.0
    assert mod.get_resample_factor("dummy.svs", level=1, resolution=2.0, unit="downsample") == pytest.approx(1.0)


def test_get_resample_factor_unsupported_unit(monkeypatch):
    def _ctor(path, use_gpu=False):
        return FakeOpenSlide(path)

    monkeypatch.setattr(mod, "_open_slide", _ctor, raising=True)

    with pytest.raises(ValueError, match="unit"):
        mod.get_resample_factor("dummy.svs", level=0, resolution=0, unit="level")


def test_read_region_cpu_returns_numpy_and_uses_openslide(monkeypatch):
    # Force CPU path via _cucim_available=False and OpenSlide→FakeOpenSlide
    monkeypatch.setattr(mod, "_cucim_available", False, raising=True)
    monkeypatch.setattr(mod, "OpenSlide", FakeOpenSlide, raising=True)

    arr = mod.read_region(path="cpu.svs", x=10, y=20, w=7, h=9, level=0, use_gpu=False, num_workers_cucim=99)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (9, 7, 3)
    # filled with 5 per FakePILImage
    assert arr.dtype == np.uint8 and int(arr[0, 0, 0]) == 5


def test_read_region_gpu_returns_numpy_and_passes_num_workers(monkeypatch):
    # Pretend cuCIM is available
    monkeypatch.setattr(mod, "_cucim_available", True, raising=True)

    fake = FakeCuImage("gpu.tif")

    # Replace CuImage constructor to return our single instance (so we can inspect .calls)
    def _ctor(path, use_gpu=False):
        assert path == "gpu.tif"
        return fake

    monkeypatch.setattr(mod, "CuImage", _ctor, raising=False)
    # CuImageType is the type for isinstance(...); fake is an instance of FakeCuImage
    monkeypatch.setattr(mod, "CuImageType", FakeCuImage, raising=False)

    arr = mod.read_region(path="gpu.tif", x=1, y=2, w=4, h=6, level=1, use_gpu=True, num_workers_cucim=13)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (6, 4, 3)
    assert int(arr[0, 0, 0]) == 7  # from FakeCuImage

    # ensure cuCIM path received our parameters
    assert fake.calls and fake.calls[0] == {"location": (1, 2), "size": (4, 6), "level": 1, "num_workers": 13}
