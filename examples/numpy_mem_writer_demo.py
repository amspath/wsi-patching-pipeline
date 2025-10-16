from __future__ import annotations

import numpy as np

from wsi_patching.core import PatchExtractor, WSIGrid
from wsi_patching.filtering import LowContrastBackgroundFilter, OtsuFilter, PenArtifactFilter
from wsi_patching.transforms import MacenkoNormalizer
from wsi_patching.writers import NumpyStreamWriter


def main():
    # Example usage (adjust 'slides' to your real paths)
    slides = ["./data/RBIO-GC072-HE-01.tiff"]

    p = (
        WSIGrid(slides=slides, level=0, use_gpu=True)
        .then(PatchExtractor(tile_size=256, stride=256, max_batch_size=800, num_workers=4))
        .then(LowContrastBackgroundFilter(range_threshold=0.2))
        .then(PenArtifactFilter())
        .then(OtsuFilter(min_tissue_fraction=0.1, tissue_is_darker=True))
        .then(MacenkoNormalizer())
        .to(NumpyStreamWriter(layout="NCHW"))
    )

    stream = p.stream(cpu_processes=2, profile=False, verbosity_level="INFO")
    for wsi_ids, final_images, final_coords, meta in stream:
        ...


if __name__ == "__main__":
    main()
