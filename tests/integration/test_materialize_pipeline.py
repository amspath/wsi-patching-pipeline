from __future__ import annotations

import tarfile

from wsi_patching.core import PatchExtractor, WSIGrid
from wsi_patching.encoders import PNGEncoder
from wsi_patching.writers import WebDatasetWriter


def test_webdataset_output(synthetic_slide, tmp_path):
    path, _ = synthetic_slide
    out_dir = tmp_path / "patches"
    out_dir.mkdir()
    p = (
        WSIGrid(slides=[path], resolution=0, unit="level", use_gpu=False)
        .then(PatchExtractor(tile_size=128, stride=128, max_batch_size=100))
        .then(PNGEncoder())
        .to(WebDatasetWriter(outdir=out_dir, shard_size=100))
    )
    p.materialize()
    tars = list(out_dir.glob("*.tar"))
    assert len(tars) >= 1
    total = sum(sum(1 for m in tarfile.open(t).getmembers() if m.name.endswith(".png")) for t in tars)
    assert total == 16
