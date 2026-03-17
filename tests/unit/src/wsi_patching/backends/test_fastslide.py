import numpy as np
import pytest

import wsi_patching.backends.fastslide as mod

# ---------------------------------------------------------------------------
# Fake fastslide objects
# ---------------------------------------------------------------------------


class FakeRegion:
    """Returned by FakeFastSlide.read_region — mimics fastslide's region object."""

    def __init__(self, w: int, h: int, fill: int = 7):
        self._arr = np.full((h, w, 3), fill, dtype=np.uint8)

    def numpy(self) -> np.ndarray:
        return self._arr


class FakeFastSlide:
    """
    Minimal stand-in for fastslide.FastSlide.

    Supports the context-manager protocol and exposes:
      - level_count
      - level_dimensions
      - level_downsamples
      - mpp
      - convert_level0_to_level_native(x, y, level)
      - read_region(location, level, size)
    """

    # Shared defaults used across most tests
    DEFAULT_LEVEL_DIMENSIONS = [(300, 200), (150, 100), (75, 50)]
    DEFAULT_LEVEL_DOWNSAMPLES = [1.0, 2.0, 4.0]
    DEFAULT_MPP = (0.5, 0.5)

    def __init__(self, path: str, *, level_dimensions=None, level_downsamples=None, mpp=DEFAULT_MPP, fill: int = 7):
        self.path = path
        self.level_dimensions = level_dimensions or self.DEFAULT_LEVEL_DIMENSIONS
        self.level_downsamples = level_downsamples or self.DEFAULT_LEVEL_DOWNSAMPLES
        self.level_count = len(self.level_dimensions)
        self.mpp = mpp
        self._fill = fill

        # Records every call to read_region for inspection in tests
        self.read_region_calls: list[dict] = []

    # -- Context-manager protocol ------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    # -- API used by the module under test ---------------------------------
    def convert_level0_to_level_native(self, x: int, y: int, level: int):
        """Scale level-0 coordinates down by the level's downsample factor."""
        ds = self.level_downsamples[level]
        return int(x / ds), int(y / ds)

    def read_region(self, *, location, level, size):
        self.read_region_calls.append({"location": location, "level": level, "size": size})
        w, h = size
        return FakeRegion(w, h, fill=self._fill)


class FakePILRegion:
    def __init__(self, w: int, h: int, fill: int = 9):
        self._arr = np.full((h, w, 3), fill, dtype=np.uint8)

    def convert(self, mode: str):
        assert mode == "RGB"
        return self

    def __array__(self):
        return self._arr


class FakeOpenSlide:
    def __init__(self, *, level_dimensions=None, level_downsamples=None, properties=None, fill: int = 9):
        self.level_dimensions = level_dimensions or [(300, 200), (150, 100), (75, 50)]
        self.level_downsamples = level_downsamples or [1.0, 2.0, 4.0]
        self.level_count = len(self.level_dimensions)
        self.properties = properties or {"openslide.mpp-x": "0.5"}
        self._fill = fill
        self.read_region_calls: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read_region(self, *, location, level, size):
        self.read_region_calls.append({"location": location, "level": level, "size": size})
        w, h = size
        return FakePILRegion(w, h, fill=self._fill)


def _make_patcher(fake: FakeFastSlide):
    """Return a drop-in replacement for mod._open_slide that always yields *fake*."""

    def _open_slide(path: str):
        return fake

    return _open_slide


# ---------------------------------------------------------------------------
# get_dimensions_for_level
# ---------------------------------------------------------------------------


class TestGetDimensionsForLevel:
    def test_returns_correct_dimensions(self, monkeypatch):
        fake = FakeFastSlide("dummy.svs")
        monkeypatch.setattr(mod, "_open_slide", _make_patcher(fake))

        W, H = mod.get_dimensions_for_level("dummy.svs", level=1)

        assert (W, H) == (150, 100)

    def test_returns_ints(self, monkeypatch):
        fake = FakeFastSlide("dummy.svs")
        monkeypatch.setattr(mod, "_open_slide", _make_patcher(fake))

        W, H = mod.get_dimensions_for_level("dummy.svs", level=0)

        assert isinstance(W, int) and isinstance(H, int)

    def test_level_zero(self, monkeypatch):
        fake = FakeFastSlide("dummy.svs")
        monkeypatch.setattr(mod, "_open_slide", _make_patcher(fake))

        W, H = mod.get_dimensions_for_level("dummy.svs", level=0)

        assert (W, H) == (300, 200)

    def test_invalid_level_raises(self, monkeypatch):
        fake = FakeFastSlide("dummy.svs")
        monkeypatch.setattr(mod, "_open_slide", _make_patcher(fake))

        with pytest.raises(AssertionError):
            mod.get_dimensions_for_level("dummy.svs", level=99)

    def test_negative_level_raises(self, monkeypatch):
        fake = FakeFastSlide("dummy.svs")
        monkeypatch.setattr(mod, "_open_slide", _make_patcher(fake))

        with pytest.raises(AssertionError):
            mod.get_dimensions_for_level("dummy.svs", level=-1)


# ---------------------------------------------------------------------------
# get_level_downsamples
# ---------------------------------------------------------------------------


class TestGetLevelDownsamples:
    def test_returns_expected_values(self, monkeypatch):
        fake = FakeFastSlide("dummy.svs")
        monkeypatch.setattr(mod, "_open_slide", _make_patcher(fake))

        ds = mod.get_level_downsamples("dummy.svs")

        assert ds == [1.0, 2.0, 4.0]

    def test_returns_floats(self, monkeypatch):
        fake = FakeFastSlide("dummy.svs")
        monkeypatch.setattr(mod, "_open_slide", _make_patcher(fake))

        ds = mod.get_level_downsamples("dummy.svs")

        assert all(isinstance(v, float) for v in ds)

    def test_single_level_slide(self, monkeypatch):
        fake = FakeFastSlide("single.svs", level_dimensions=[(512, 512)], level_downsamples=[1.0])
        monkeypatch.setattr(mod, "_open_slide", _make_patcher(fake))

        assert mod.get_level_downsamples("single.svs") == [1.0]


# ---------------------------------------------------------------------------
# get_level_for_resolution — unit="level"
# ---------------------------------------------------------------------------


class TestGetLevelForResolutionUnitLevel:
    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        fake = FakeFastSlide("dummy.svs")
        monkeypatch.setattr(mod, "_open_slide", _make_patcher(fake))

    def test_valid_integer_level(self):
        assert mod.get_level_for_resolution("dummy.svs", resolution=0, unit="level", fallback_mode="nearest") == 0
        assert mod.get_level_for_resolution("dummy.svs", resolution=2, unit="level", fallback_mode="nearest") == 2

    def test_float_resolution_raises(self):
        with pytest.raises(ValueError, match="integer"):
            mod.get_level_for_resolution("dummy.svs", resolution=1.5, unit="level", fallback_mode="nearest")

    def test_negative_level_raises(self):
        with pytest.raises(ValueError):
            mod.get_level_for_resolution("dummy.svs", resolution=-1, unit="level", fallback_mode="nearest")

    def test_out_of_range_level_raises(self):
        with pytest.raises(ValueError):
            mod.get_level_for_resolution("dummy.svs", resolution=99, unit="level", fallback_mode="nearest")


# ---------------------------------------------------------------------------
# get_level_for_resolution — unit="downsample"
# ---------------------------------------------------------------------------


class TestGetLevelForResolutionUnitDownsample:
    # level_downsamples = [1.0, 2.0, 4.0]

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        fake = FakeFastSlide("dummy.svs")
        monkeypatch.setattr(mod, "_open_slide", _make_patcher(fake))

    def test_nearest_picks_closest(self):
        # 1.3 is closer to 1.0 than to 2.0
        assert (
            mod.get_level_for_resolution("dummy.svs", resolution=1.3, unit="downsample", fallback_mode="nearest") == 0
        )
        # 1.6 is closer to 2.0
        assert (
            mod.get_level_for_resolution("dummy.svs", resolution=1.6, unit="downsample", fallback_mode="nearest") == 1
        )

    def test_nearest_exact_match(self):
        assert (
            mod.get_level_for_resolution("dummy.svs", resolution=4.0, unit="downsample", fallback_mode="nearest") == 2
        )

    def test_floor_picks_coarsest_above(self):
        # requested=1.1 → eligible [2.0, 4.0] → coarsest with value ≥ requested but closest = 2.0 (idx 1)
        assert mod.get_level_for_resolution("dummy.svs", resolution=1.1, unit="downsample", fallback_mode="floor") == 1

    def test_floor_fallback_to_coarsest(self):
        # nothing ≥ 10.0 → use last level
        assert mod.get_level_for_resolution("dummy.svs", resolution=10.0, unit="downsample", fallback_mode="floor") == 2

    def test_ceil_picks_finest_below(self):
        # requested=3.0 → eligible [1.0, 2.0] → finest with value ≤ 3.0 = 2.0 (idx 1)
        assert mod.get_level_for_resolution("dummy.svs", resolution=3.0, unit="downsample", fallback_mode="ceil") == 1

    def test_ceil_fallback_to_finest(self):
        # nothing ≤ 0.1 → fall back to level 0
        assert mod.get_level_for_resolution("dummy.svs", resolution=0.1, unit="downsample", fallback_mode="ceil") == 0

    def test_error_exact_match(self):
        assert mod.get_level_for_resolution("dummy.svs", resolution=2.0, unit="downsample", fallback_mode="error") == 1

    def test_error_no_match_raises(self):
        with pytest.raises(ValueError):
            mod.get_level_for_resolution("dummy.svs", resolution=3.0, unit="downsample", fallback_mode="error")


# ---------------------------------------------------------------------------
# get_level_for_resolution — unit="mpp"
# ---------------------------------------------------------------------------


class TestGetLevelForResolutionUnitMpp:
    # mpp0=0.5, downsamples=[1.0, 2.0, 4.0] → mpp per level = [0.5, 1.0, 2.0]

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        fake = FakeFastSlide("dummy.svs", mpp=(0.5, 0.5))
        monkeypatch.setattr(mod, "_open_slide", _make_patcher(fake))

    def test_nearest_mpp(self):
        # 0.8 → nearest to 1.0 (idx 1)
        assert mod.get_level_for_resolution("dummy.svs", resolution=0.8, unit="mpp", fallback_mode="nearest") == 1

    def test_exact_mpp_match(self):
        assert mod.get_level_for_resolution("dummy.svs", resolution=0.5, unit="mpp", fallback_mode="nearest") == 0
        assert mod.get_level_for_resolution("dummy.svs", resolution=2.0, unit="mpp", fallback_mode="nearest") == 2

    def test_missing_mpp_metadata_raises(self, monkeypatch):
        fake_no_mpp = FakeFastSlide("dummy.svs", mpp=(None, None))
        monkeypatch.setattr(mod, "_open_slide", _make_patcher(fake_no_mpp))

        with pytest.raises(ValueError, match="mpp"):
            mod.get_level_for_resolution("dummy.svs", resolution=0.5, unit="mpp", fallback_mode="nearest")


# ---------------------------------------------------------------------------
# get_level_for_resolution — invalid inputs
# ---------------------------------------------------------------------------


class TestGetLevelForResolutionInvalidInputs:
    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        fake = FakeFastSlide("dummy.svs")
        monkeypatch.setattr(mod, "_open_slide", _make_patcher(fake))

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown unit"):
            mod.get_level_for_resolution("dummy.svs", resolution=1.0, unit="pixels", fallback_mode="nearest")

    def test_unknown_fallback_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown fallback_mode"):
            mod.get_level_for_resolution("dummy.svs", resolution=1.0, unit="downsample", fallback_mode="fuzzy")


# ---------------------------------------------------------------------------
# read_region
# ---------------------------------------------------------------------------


class TestReadRegion:
    def test_returns_numpy_array_with_correct_shape(self, monkeypatch):
        fake = FakeFastSlide("test.svs", fill=42)
        monkeypatch.setattr(mod, "_open_slide", _make_patcher(fake))

        arr = mod.read_region("test.svs", x=0, y=0, w=16, h=8, level=0)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (8, 16, 3)
        assert arr.dtype == np.uint8

    def test_pixel_values_come_from_slide(self, monkeypatch):
        fake = FakeFastSlide("test.svs", fill=42)
        monkeypatch.setattr(mod, "_open_slide", _make_patcher(fake))

        arr = mod.read_region("test.svs", x=0, y=0, w=4, h=4, level=0)

        assert int(arr[0, 0, 0]) == 42

    def test_level0_coordinates_are_converted(self, monkeypatch):
        """
        At level 1 the downsample is 2×, so level-0 coords (20, 40)
        should become native coords (10, 20).
        """
        fake = FakeFastSlide("test.svs")
        monkeypatch.setattr(mod, "_open_slide", _make_patcher(fake))

        mod.read_region("test.svs", x=20, y=40, w=8, h=6, level=1)

        call = fake.read_region_calls[0]
        assert call["location"] == (10, 20)

    def test_requested_size_is_passed_through(self, monkeypatch):
        fake = FakeFastSlide("test.svs")
        monkeypatch.setattr(mod, "_open_slide", _make_patcher(fake))

        mod.read_region("test.svs", x=0, y=0, w=32, h=24, level=0)

        call = fake.read_region_calls[0]
        assert call["size"] == (32, 24)

    def test_correct_level_is_passed_to_slide(self, monkeypatch):
        fake = FakeFastSlide("test.svs")
        monkeypatch.setattr(mod, "_open_slide", _make_patcher(fake))

        mod.read_region("test.svs", x=0, y=0, w=4, h=4, level=2)

        assert fake.read_region_calls[0]["level"] == 2

    def test_dicom_uses_openslide_coordinates(self, monkeypatch):
        fake = FakeOpenSlide(fill=24)
        monkeypatch.setattr(mod, "_open_slide_openslide", lambda _: fake)

        arr = mod.read_region("test.dcm", x=20, y=40, w=8, h=6, level=1)

        assert arr.shape == (6, 8, 3)
        assert int(arr[0, 0, 0]) == 24
        call = fake.read_region_calls[0]
        assert call["location"] == (20, 40)
        assert call["level"] == 1
        assert call["size"] == (8, 6)


# ---------------------------------------------------------------------------
# get_level_for_resolution — fallback_mode="resample"
# ---------------------------------------------------------------------------


class TestGetLevelForResolutionResampleMpp:
    # mpp0=0.5, downsamples=[1.0, 2.0, 4.0] → mpp per level = [0.5, 1.0, 2.0]

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        fake = FakeFastSlide("dummy.svs", mpp=(0.5, 0.5))
        monkeypatch.setattr(mod, "_open_slide", _make_patcher(fake))

    def test_resample_picks_finest_finer_level(self):
        # requested=1.5 mpp → eligible levels with mpp <= 1.5: [0.5 (0), 1.0 (1)] → finest = level 1
        assert mod.get_level_for_resolution("dummy.svs", resolution=1.5, unit="mpp", fallback_mode="resample") == 1

    def test_resample_falls_back_to_level0_if_all_coarser(self):
        # requested=0.3 mpp → no level with mpp <= 0.3 → fall back to level 0
        assert mod.get_level_for_resolution("dummy.svs", resolution=0.3, unit="mpp", fallback_mode="resample") == 0

    def test_resample_exact_match(self):
        # requested=1.0 mpp → exactly matches level 1
        assert mod.get_level_for_resolution("dummy.svs", resolution=1.0, unit="mpp", fallback_mode="resample") == 1


class TestGetLevelForResolutionResampleDownsample:
    # level_downsamples = [1.0, 2.0, 4.0]

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        fake = FakeFastSlide("dummy.svs")
        monkeypatch.setattr(mod, "_open_slide", _make_patcher(fake))

    def test_resample_picks_finest_finer_level(self):
        # requested=3.0 → eligible [1.0 (0), 2.0 (1)] → finest = level 1
        assert (
            mod.get_level_for_resolution("dummy.svs", resolution=3.0, unit="downsample", fallback_mode="resample") == 1
        )

    def test_resample_falls_back_to_level0_if_all_coarser(self):
        # requested=0.5 → no level with ds <= 0.5 → fall back to level 0
        assert (
            mod.get_level_for_resolution("dummy.svs", resolution=0.5, unit="downsample", fallback_mode="resample") == 0
        )

    def test_resample_raises_for_unit_level(self):
        with pytest.raises(ValueError, match="resample"):
            mod.get_level_for_resolution("dummy.svs", resolution=1, unit="level", fallback_mode="resample")


# ---------------------------------------------------------------------------
# get_resample_factor
# ---------------------------------------------------------------------------


class TestGetResampleFactor:
    # mpp0=0.5, downsamples=[1.0, 2.0, 4.0] → mpp per level = [0.5, 1.0, 2.0]

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        fake = FakeFastSlide("dummy.svs", mpp=(0.5, 0.5))
        monkeypatch.setattr(mod, "_open_slide", _make_patcher(fake))

    def test_mpp_factor_above_one(self):
        # requested=1.5, level=1 (mpp=1.0) → factor = 1.5 / 1.0 = 1.5
        rf = mod.get_resample_factor("dummy.svs", resolution=1.5, unit="mpp", selected_level=1)
        assert abs(rf - 1.5) < 1e-9

    def test_mpp_factor_exact_match(self):
        # requested=1.0, level=1 (mpp=1.0) → factor = 1.0
        rf = mod.get_resample_factor("dummy.svs", resolution=1.0, unit="mpp", selected_level=1)
        assert abs(rf - 1.0) < 1e-9

    def test_downsample_factor(self):
        # requested=3.0, level=1 (ds=2.0) → factor = 3.0 / 2.0 = 1.5
        rf = mod.get_resample_factor("dummy.svs", resolution=3.0, unit="downsample", selected_level=1)
        assert abs(rf - 1.5) < 1e-9

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown unit"):
            mod.get_resample_factor("dummy.svs", resolution=1.0, unit="level", selected_level=0)

    def test_missing_mpp_raises(self, monkeypatch):
        fake_no_mpp = FakeFastSlide("dummy.svs", mpp=(None, None))
        monkeypatch.setattr(mod, "_open_slide", _make_patcher(fake_no_mpp))
        with pytest.raises(ValueError, match="mpp"):
            mod.get_resample_factor("dummy.svs", resolution=1.5, unit="mpp", selected_level=1)


class TestDicomRouting:
    def test_open_slide_uses_openslide_for_dicom(self, monkeypatch):
        fake = FakeOpenSlide()
        monkeypatch.setattr(mod, "_open_slide_openslide", lambda _: fake)

        assert mod._open_slide("example.dcm") is fake

    def test_get_level_for_resolution_reads_mpp_from_openslide_properties(self, monkeypatch):
        fake = FakeOpenSlide(properties={"openslide.mpp-x": "0.5"})
        monkeypatch.setattr(mod, "_open_slide_openslide", lambda _: fake)

        level = mod.get_level_for_resolution("example.dcm", resolution=1.0, unit="mpp", fallback_mode="nearest")

        assert level == 1
