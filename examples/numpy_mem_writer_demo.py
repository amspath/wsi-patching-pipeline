from __future__ import annotations

import numpy as np

from wsi_patching.core import ReadWindowChunker, RegionReadAndBatch, TilePlanner, WSIGrid
from wsi_patching.filtering import LowContrastBackgroundFilter, OtsuFilter, PenArtifactFilter
from wsi_patching.transforms import MacenkoNormalizer
from wsi_patching.utils import visualize_selected_patches
from wsi_patching.writers import NumpyMemoryWriter


def main():
    # Example usage (adjust 'slides' to your real paths)
    slides = ["./data/RBIO-GC072-HE-01.tiff"]

    p = (
        WSIGrid(slides=slides, tile_size=256, stride=256, level=0, use_gpu=True)
        .then(TilePlanner())
        .then(ReadWindowChunker())
        .then(RegionReadAndBatch(batch_size=800, num_workers=4))
        .then(LowContrastBackgroundFilter(range_threshold=0.2))
        .then(PenArtifactFilter())
        .then(OtsuFilter(min_tissue_fraction=0.1, tissue_is_darker=True))
        .then(MacenkoNormalizer())
        .to(NumpyMemoryWriter(layout="NCHW"))
    )

    final_images, final_coords, wsi_ids = p.run(cpu_processes=2, profile=False, verbosity_level="INFO")
    visualize_selected_patches(
        slides[0],
        coords=final_coords,
        patch_size=256,
        patch_images=final_images.astype(np.uint8),
        stride=256,
        save_path="selected_patches_demo.png",
    )


if __name__ == "__main__":
    main()
