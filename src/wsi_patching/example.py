# example.py
# A minimal, single-file skeleton that matches your desired pipeline:
#   WSIGrid → FilterByROI → Regionize → RegionReadAndBatch → GPUOps → PNGEncoder → ToWebDataset
#
# Design goals:
# - Region-prefetch with cuCIM: read a big region once, then slice into tiles (cheap).
# - Multi-WSI parallelism via multiprocessing (one producer process per WSI).
# - Micro-batched GPU ops (e.g., 200 tiles) in a dedicated GPU process.
# - Single, continuous WebDataset writer draining a bounded queue ("write when ready").
#
# Notes:
# - This file is intentionally compact and pragmatic. Many parts are simplified stubs you can extend.
# - cuCIM, torch, and webdataset are optional. If they are not installed, certain parts will no-op or raise
#   a helpful error. Wire in your real kernels and readers where marked with TODOs.
# - ROI format: rectangles in level coordinates [(x, y, w, h), ...] per WSI.
#
# Run style (pseudo):
#   pipeline = (
#       WSIGrid(slides, tile_size=256, stride=256, level=0)
#       .then(FilterByROI(roi_by_wsi={"slideA": [(0,0,4096,4096)]}))
#       .then(Regionize(max_region_mp=96))
#       .then(RegionReadAndBatch(cucim_workers=8))
#       .then(GPUOps(device=0, batch_size=200, batch_timeout_ms=75))
#       .then(PNGEncoder())
#       .then(ToWebDataset(pattern="/out/train-%06d.tar", maxcount=25000))
#   )
#   pipeline.run(max_producers=4, gpu_devices=[0])
#
# Author: you + ChatGPT (quick-start skeleton)

from __future__ import annotations

import time
from ast import List
from pathlib import Path

from wsi_patching.core import Pipeline, WSIGrid
from wsi_patching.encoders import PNGEncoder
from wsi_patching.filters import FilterByROI
from wsi_patching.gpustuff import GPUOps
from wsi_patching.patchers import Regionize, RegionReadAndBatch
from wsi_patching.writers import RandomizedShardWriter


def _demo_build_pipeline(slides: List[str]) -> Pipeline:
    """
    Small helper to show how you'd wire the stages together.
    """
    rois = {Path(s).stem: [(0, 0, 4000, 4000)] for s in slides}

    pipeline = (
        WSIGrid(slides=slides, tile_size=256, stride=256, level=0)
        .then(FilterByROI(roi_by_wsi=rois))
        .then(Regionize(max_region_mp=96))
        .then(RegionReadAndBatch(cucim_workers=8))
        .then(GPUOps(device=0, batch_size=200, batch_timeout_ms=75))  # placement = gpu
        .then(PNGEncoder())  # placement = gpu
        .then(
            RandomizedShardWriter(pattern="./output/train-%06d.tar", shard_size=500, buffer_multiplier=2)
        )  # placement = writer
    )
    return pipeline


def main(argv=None):
    # Example usage (adjust 'slides' to your real paths)
    slides = [
        "./data/RBIO-GC072-HE-01.tiff",
        "./data/RBIO-GC072-HE-02.tiff",
        "./data/RBIO-GC072-HE-03.tiff",
        "./data/RBIO-GC072-HE-04.tiff",
        "./data/RBIO-GC072-HE-05.tiff",
        "./data/RBIO-GC072-HE-06.tiff",
        "./data/RBIO-GC072-HE-07.tiff",
        "./data/RBIO-GC072-HE-08.tiff",
        "./data/RBIO-GC072-HE-09.tiff",
        "./data/RBIO-GC072-HE-10.tiff",
    ]
    if not slides:
        print("Populate 'slides' with real WSI paths before running.")
        exit(0)

    p = _demo_build_pipeline(slides)
    # Tune concurrency as needed
    start_time = time.time()
    p.run(max_producers=10, gpu_devices=[0])
    print(f"Done in {time.time() - start_time:.1f} seconds.")
