import numpy as np
import pytest

import wsi_patching.backends.cucim_openslide as mod


# ---------- fakes ----------
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
    # OpenSlide exposes .level_dimensions
    level_dimensions = [(300, 200), (120, 80)]

    def __init__(self, path):
        self.path = path

    def read_region(self, location, level, size):
        # location: (x,y), level: int, size: (w,h)
        w, h = size
        return FakePILImage(w, h, fill=5)

    def close(self):
        pass


# ---------- tests ----------
def test_validate_slide_backend_raises_when_gpu_requested_but_cucim_missing(monkeypatch):
    monkeypatch.setattr(mod, "_cucim_available", False, raising=True)
    with pytest.raises(ImportError):
        mod.validate_slide_backend(use_gpu=True)

    # CPU path never raises on missing cuCIM
    mod.validate_slide_backend(use_gpu=False)


def test_get_dimensions_for_level_cpu(monkeypatch):
    monkeypatch.setattr(mod, "_cucim_available", False, raising=True)
    monkeypatch.setattr(mod, "OpenSlide", FakeOpenSlide, raising=True)

    W, H = mod.get_dimensions_for_level("dummy.svs", level=1, use_gpu=False)
    assert (W, H) == (120, 80)  # from FakeOpenSlide.level_dimensions[1]
    assert isinstance(W, int) and isinstance(H, int)


def test_get_dimensions_for_level_gpu_with_cucim(monkeypatch):
    # Pretend cuCIM is available
    monkeypatch.setattr(mod, "_cucim_available", True, raising=True)

    # CuImage is the constructor, CuImageType is the type used in isinstance(...)
    monkeypatch.setattr(mod, "CuImage", FakeCuImage, raising=False)
    monkeypatch.setattr(mod, "CuImageType", FakeCuImage, raising=False)

    W, H = mod.get_dimensions_for_level("dummy.tif", level=1, use_gpu=True)
    assert (W, H) == (25, 10)  # from FakeCuImage.resolutions["level_dimensions"][1]
    assert isinstance(W, int) and isinstance(H, int)


def test_read_region_cpu_returns_numpy_and_uses_openslide(monkeypatch):
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
    def _ctor(path):
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
