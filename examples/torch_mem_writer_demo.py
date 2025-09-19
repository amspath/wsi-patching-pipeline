from __future__ import annotations

import logging
import time
from pathlib import Path

from wsi_patching.core.chunking_and_batching import ReadWindowChunker, RegionReadAndBatch, TilePlanner
from wsi_patching.core.regions_of_interest import AttachROIs, RectROIProvider
from wsi_patching.core.wsi_grid import WSIGrid
from wsi_patching.tissue_classifier import DummyTissueClassifier
from wsi_patching.writers.torch_mem_writer import TorchMemoryWriter


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
    slides = ["./data/RBIO-GC072-HE-01.tiff", "./data/RBIO-GC072-HE-02.tiff"]

    # Example ROI dict (compat with old code)
    rois_dict = {Path(s).stem: [(0, 0, 4000, 4000)] for s in slides}

    p = (
        WSIGrid(slides=slides, tile_size=256, stride=256, level=0, use_gpu=True)
        .then(AttachROIs(providers=[RectROIProvider(rois_dict)]))
        .then(TilePlanner())
        .then(ReadWindowChunker(max_window_size=4096))
        .then(RegionReadAndBatch(batch_size=args.batch, num_workers=args.num_workers))
        .then(DummyTissueClassifier())
        .to(TorchMemoryWriter(layout="NCHW"))
    )

    start_time = time.time()
    torch_dataset = p.run(cpu_processes=args.procs, profile=args.profile)
    print(torch_dataset)
    logging.info(f"Done in {time.time() - start_time:.1f} seconds.")

    if args.profile:
        # Print a summary on completion if requested
        p.print_profile()


if __name__ == "__main__":
    main()
