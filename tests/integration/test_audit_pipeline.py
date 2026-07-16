
from wsi_patching import OtsuFilter, PatchExtractor, RemoveEdgeTiles, WSIGrid, visualize_audit
from wsi_patching.encoders import PNGEncoder
from wsi_patching.writers import WebDatasetWriter


def _pipe(path):
    # 512x512 slide, tile 128 -> 4x4 = 16 candidate patches.
    return (
        WSIGrid(slides=[path], resolution=0, unit="level", use_gpu=False)
        .then(PatchExtractor(tile_size=128, stride=128, max_batch_size=100))
        .then(RemoveEdgeTiles(depth=1))
    )


def _coords(arr):
    return {(int(x), int(y)) for x, y in arr}


def test_dry_run_audit_funnel(synthetic_slide):
    path, wsi_id = synthetic_slide
    p = _pipe(path)  # no writer needed for a dry run
    p.dry_run()

    entry = p.get_audit()["by_slide"][wsi_id]
    funnel = {row["stage"]: row for row in entry["funnel"]}

    # depth=1 on a 4x4 grid keeps the inner 2x2 (4) and drops the border ring (12).
    assert funnel["PatchExtractor"]["out"] == 16
    assert funnel["RemoveEdgeTiles"]["out"] == 4
    assert funnel["RemoveEdgeTiles"]["dropped"] == 12
    assert len(entry["kept"]) == 4

    # Drops are attributed to the right stage, and every drop is accounted for.
    total_dropped = sum(len(d["coords"]) for d in entry["dropped"])
    assert total_dropped == 16 - len(entry["kept"])
    assert entry["dropped"][0]["stage"] == "RemoveEdgeTiles"


def test_dry_run_skips_encoder(synthetic_slide):
    # A trailing PNGEncoder must not break dry_run (it's dropped, not run).
    path, wsi_id = synthetic_slide
    p = _pipe(path).then(PNGEncoder())
    p.dry_run()
    assert p.get_audit()["by_slide"][wsi_id]["funnel"][-1]["stage"] == "RemoveEdgeTiles"


def test_visualize_audit_html(synthetic_slide):
    path, wsi_id = synthetic_slide
    p = _pipe(path)
    p.dry_run()

    report = visualize_audit(p, path)
    html = report._repr_html_()
    assert wsi_id in html
    assert "RemoveEdgeTiles" in html
    assert "data:image/png;base64," in html


def test_permissive_tuning_run_matches_an_honest_run(synthetic_slide):
    """The load-bearing guarantee of tune=True.

    A permissive run neutralizes every knob and re-derives the drops from the
    recorded metrics. That reconstruction must be indistinguishable from actually
    running the filters at their configured thresholds -- otherwise the sliders lie.
    """
    path, wsi_id = synthetic_slide

    honest = _pipe(path).dry_run(tune=False).get_audit()["by_slide"][wsi_id]
    tuned = _pipe(path).dry_run(tune=True).get_audit()["by_slide"][wsi_id]

    assert tuned["funnel"] == honest["funnel"]
    assert _coords(tuned["kept"]) == _coords(honest["kept"])
    assert [(d["stage"], _coords(d["coords"])) for d in tuned["dropped"]] == [
        (d["stage"], _coords(d["coords"])) for d in honest["dropped"]
    ]
    assert tuned["tuning"] is True and honest["tuning"] is False


def test_tuning_run_exposes_slider_data_for_every_knob(synthetic_slide):
    path, wsi_id = synthetic_slide
    p = (
        WSIGrid(slides=[path], resolution=0, unit="level", use_gpu=False)
        .then(PatchExtractor(tile_size=128, stride=128, max_batch_size=100))
        .then(OtsuFilter(min_tissue_fraction=0.35))
        .then(RemoveEdgeTiles(depth=1))
    )
    entry = p.dry_run().get_audit()["by_slide"][wsi_id]

    knobs = {k["param"]: k for k in entry["knobs"]}
    assert set(knobs) == {"min_tissue_fraction", "depth"}
    assert knobs["min_tissue_fraction"]["init"] == 0.35
    assert knobs["depth"]["init"] == 1

    # Every candidate must carry every knob's metric -- that is what makes the
    # sliders bidirectional. A gap here means a filter dropped despite permissive mode.
    for c in _coords(entry["candidates"]):
        meta = entry["patch_meta"][c]
        assert "tissue_fraction" in meta and "edge_distance" in meta

    # And the recorded metrics reproduce the funnel for other values, live.
    loose = p.get_audit(values={k["stage_idx"]: 0.0 for k in entry["knobs"]})["by_slide"][wsi_id]
    assert len(loose["kept"]) == len(entry["candidates"])


def test_sliders_only_when_the_run_can_support_them(synthetic_slide):
    """A non-tuning run has no metrics to slide on, so it must embed no knobs.

    The report builds a slider per knob in the embedded data, so the data is what
    decides -- the markup for a slider is in the template either way.
    """
    path, _ = synthetic_slide

    static = visualize_audit(_pipe(path).dry_run(tune=False), path)._repr_html_()
    assert '"knob": null' in static and '"param": "depth"' not in static

    tuned = visualize_audit(_pipe(path).dry_run(), path)._repr_html_()
    assert '"param": "depth"' in tuned and '"column": "edge_distance"' in tuned


def test_edge_distance_metric_matches_the_stages_own_drops(synthetic_slide):
    """The metric behind the depth slider must agree with what the stage really did."""
    path, wsi_id = synthetic_slide
    entry = _pipe(path).dry_run(tune=False).get_audit()["by_slide"][wsi_id]

    for c in _coords(entry["kept"]):
        assert entry["patch_meta"][c]["edge_distance"] >= 1
    for drop in entry["dropped"]:
        for c in _coords(drop["coords"]):
            assert entry["patch_meta"][c]["edge_distance"] < 1


def test_no_audit_by_default(synthetic_slide, tmp_path):
    # A normal run without audit=True records nothing (zero overhead path).
    path, _ = synthetic_slide
    out = tmp_path / "patches"
    out.mkdir()
    p = _pipe(path).then(PNGEncoder()).to(WebDatasetWriter(outdir=out, shard_size=100))
    p.materialize()
    assert p.get_audit() == {"by_slide": {}}
