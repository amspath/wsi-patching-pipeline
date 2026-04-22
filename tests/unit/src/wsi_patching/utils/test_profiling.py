import math
import re

from wsi_patching.utils.profiling import PipelineProfileAggregator, Profiler, get_current_profiler, set_current_profiler


def test_profiler_disabled_no_stats():
    """Disabled profiler should not record any stats."""
    p = Profiler(enabled=False, slide_id="s1")
    p.add_time("load", 1.23, yielded=True)
    assert p.serialize() == {"slide_id": "s1", "stages": {}}


def test_profiler_accumulates_wall_and_yields():
    """Test that multiple add_time calls accumulate correctly."""
    p = Profiler(enabled=True, slide_id="slide-A")
    p.add_time("stageX", 0.2, yielded=True)  # counts yield
    p.add_time("stageX", 0.3, yielded=False)  # no yield increment

    ser = p.serialize()
    assert ser["slide_id"] == "slide-A"
    assert "stageX" in ser["stages"]
    assert math.isclose(ser["stages"]["stageX"]["wall_time_sec"], 0.5, rel_tol=1e-9)
    assert ser["stages"]["stageX"]["yields"] == 1


def test_aggregator_ingest_and_get_profile_single():
    """Test ingestion and profile retrieval for a single slide."""
    # Build a fake producer summary like Profiler.serialize()
    msg = {
        "slide_id": "s1",
        "stages": {"load": {"wall_time_sec": 2.0, "yields": 4}, "proc": {"wall_time_sec": 1.0, "yields": 0}},
    }
    agg = PipelineProfileAggregator()
    agg.ingest_msg(msg)

    prof = agg.get_profile()

    # by_slide mirrors message + computed avg_ms_per_yield
    assert "s1" in prof["by_slide"]
    assert prof["by_slide"]["s1"]["load"]["yields"] == 4
    assert math.isclose(prof["by_slide"]["s1"]["load"]["avg_ms_per_yield"], 2.0 / 4 * 1000.0, rel_tol=1e-9)

    # when yields == 0, avg is 0.0
    assert prof["by_slide"]["s1"]["proc"]["yields"] == 0
    assert prof["by_slide"]["s1"]["proc"]["avg_ms_per_yield"] == 0.0

    # by_stage aggregates (here identical to single slide)
    assert prof["by_stage"]["load"]["yields"] == 4
    assert math.isclose(prof["by_stage"]["load"]["wall_time_sec"], 2.0, rel_tol=1e-9)
    assert math.isclose(prof["by_stage"]["load"]["avg_ms_per_yield"], 2.0 / 4 * 1000.0, rel_tol=1e-9)


def test_aggregator_multiple_slides_and_stages():
    """Test aggregation across multiple slides and stages."""
    agg = PipelineProfileAggregator()
    agg.ingest_msg(
        {
            "slide_id": "A",
            "stages": {"load": {"wall_time_sec": 1.0, "yields": 2}, "prep": {"wall_time_sec": 0.5, "yields": 1}},
        }
    )
    agg.ingest_msg(
        {
            "slide_id": "B",
            "stages": {"load": {"wall_time_sec": 3.0, "yields": 3}, "proc": {"wall_time_sec": 2.0, "yields": 4}},
        }
    )

    prof = agg.get_profile()

    # by_stage sums across slides
    assert math.isclose(prof["by_stage"]["load"]["wall_time_sec"], 4.0, rel_tol=1e-9)
    assert prof["by_stage"]["load"]["yields"] == 5  # 2 + 3
    assert math.isclose(prof["by_stage"]["load"]["avg_ms_per_yield"], 4.0 / 5 * 1000.0, rel_tol=1e-9)

    # ensure unique stages are present
    assert "prep" in prof["by_stage"]
    assert "proc" in prof["by_stage"]

    # by_slide has both slides with computed avgs
    for slide_id in ("A", "B"):
        assert slide_id in prof["by_slide"]


def test_print_profile_output_contains_headers_and_sorted(caplog):
    """Test that print_profile logs expected headers and sorted stages."""
    import logging

    agg = PipelineProfileAggregator()
    # Make 'big' stage to appear first (sorted by wall_time_sec desc)
    agg.ingest_msg(
        {
            "slide_id": "S1",
            "stages": {"big": {"wall_time_sec": 5.0, "yields": 5}, "small": {"wall_time_sec": 1.0, "yields": 2}},
        }
    )
    with caplog.at_level(logging.INFO, logger="wsi_patching.utils.profiling"):
        agg.print_profile()
    out = "\n".join(caplog.messages)

    # Headers
    assert "=== Pipeline Profile (isolated timings only) ===" in out
    assert re.search(r"\bStage\b\s+Yields\s+Wall \(s\)\s+Avg \(ms/yield\)", out)

    # Per-slide section
    assert "--- Per slide breakdown ---" in out
    assert "[S1]" in out
    assert "big" in out and "small" in out

    # Order: 'big' should appear before 'small' in the by_stage table
    idx_big = out.find("big")
    idx_small = out.find("small")
    assert idx_big != -1 and idx_small != -1 and idx_big < idx_small


def test_global_profiler_set_get_and_clear():
    """Test setting, getting, and clearing the global current profiler."""
    p = Profiler(enabled=True, slide_id="G1")
    set_current_profiler(p)
    assert get_current_profiler() is p

    set_current_profiler(None)
    assert get_current_profiler() is None


def test_aggregator_reset_clears_state():
    """Test that resetting the aggregator clears its internal state."""
    agg = PipelineProfileAggregator()
    agg.ingest_msg({"slide_id": "S", "stages": {"x": {"wall_time_sec": 1.0, "yields": 1}}})
    assert agg.get_profile()["by_stage"]  # not empty
    agg.reset()
    prof = agg.get_profile()
    assert prof["by_stage"] == {}
    assert prof["by_slide"] == {}
