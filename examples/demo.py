#!/usr/bin/env python3
"""
Minimal streaming WSI patch pipeline with region-prefetch (cuCIM if available),
per-WSI multiprocessing producers, and a single async WebDataset writer.

Pipeline (as used in main()):
    WSIGrid -> FilterByROI -> Regionize -> RegionReadAndBatch -> DummyTissueClassifier -> PNGEncoder -> WebDatasetWriter

Notes
-----
- This is a barebones, runnable skeleton designed to match the requested API and flow.
- cuCIM is optional at runtime. If unavailable, we fall back to Pillow for small images (level=0 only).
  For real WSIs, install cuCIM and pass real slide paths.
- Multiprocessing model:
    * One producer process per slide (or up to cpu_processes concurrently)
    * One writer process drains a bounded MP queue and writes tar shards continuously
- GPU ops:
    * DummyTissueClassifier simulates a batched GPU step if torch+CUDA are available.
    * It waits for batches (default size 200) emitted by RegionReadAndBatch, then returns the batch.
- WebDataset writer:
    * The writer process owns the only ShardWriter.
    * Samples are written as they arrive; no ordering guarantees.

Profiling
---------
- Enable via Pipeline.run(..., profile=True).
- Each producer process profiles its stages (writer excluded) and sends a summary
  to the parent via a dedicated queue.
- After run(), call Pipeline.get_profile() for a dict or Pipeline.print_profile() for a summary.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from wsi_patching.core.chunking_and_batching import ReadWindowChunker, RegionReadAndBatch, TilePlanner
from wsi_patching.core.regions_of_interest import AttachROIs, RectROIProvider
from wsi_patching.core.wsi_grid import WSIGrid
from wsi_patching.filtering.cellvit_tissue_classifier_filter import CellVitTissueClassifierFilter
from wsi_patching.pngencoder import PNGEncoder
from wsi_patching.writers.webdataset.webdataset_writer import WebDatasetWriter


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Minimal streaming WSI patcher with region-prefetch and WebDataset sink."
    )
    parser.add_argument("--out", type=str, default="./output/train-%06d.tar", help="Shard pattern for WebDataset.")
    parser.add_argument("--procs", type=int, default=4, help="Max producer processes (one per slide concurrently).")
    parser.add_argument("--batch", type=int, default=200, help="Batch size for GPU micro-batching.")
    parser.add_argument("--num-workers", type=int, default=8, help="cuCIM num_workers per region read.")
    parser.add_argument(
        "--profile", action="store_true", help="Enable per-stage profiling for producers.", default=True
    )
    args = parser.parse_args(argv)

    # Example usage (adjust 'slides' to your real paths)
    slides = [
        "./data/RBIO-GC072-HE-01.tiff",
        "./data/RBIO-GC072-HE-02.tiff",
        "./data/RBIO-GC072-HE-03.tiff",
        "./data/RBIO-GC072-HE-04.tiff",
    ]

    # Example ROI dict (compat with old code)
    rois_dict = {Path(s).stem: [(0, 0, 8000, 8000)] for s in slides}

    p = (
        WSIGrid(slides=slides, tile_size=224, stride=224, level=0, use_gpu=True)
        .then(AttachROIs(providers=[RectROIProvider(rois_dict)]))
        .then(TilePlanner())
        .then(ReadWindowChunker(max_window_size=224 * 40))
        .then(RegionReadAndBatch(batch_size=args.batch, num_workers=args.num_workers))
        .then(CellVitTissueClassifierFilter())
        .then(PNGEncoder())
        .to(WebDatasetWriter())
    )

    start_time = time.time()
    p.run(cpu_processes=args.procs, profile=args.profile, verbosity_level="INFO")
    logging.info(f"Done in {time.time() - start_time:.1f} seconds.")

    if args.profile:
        # Print a summary on completion if requested
        p.print_profile()


if __name__ == "__main__":
    main()
