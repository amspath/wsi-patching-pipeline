import logging
import time
from pathlib import Path

from wsi_patching.core import PatchExtractor, WSIGrid
from wsi_patching.regions_of_interest.attach_rois import AttachROIs
from wsi_patching.regions_of_interest.roi_providers import RectROIProvider
from wsi_patching.writers import NumpyStreamWriter


def main():
    # Example usage (adjust 'slides' to your real paths)
    slides = [
        "./data/RBIO-GC072-HE-01.tiff",
        "./data/RBIO-GC072-HE-02.tiff",
        "./data/RBIO-GC072-HE-03.tiff",
        "./data/RBIO-GC072-HE-04.tiff",
    ]

    rois_dict = {Path(s).stem: [(0, 0, 18000, 10000)] for s in slides}

    p = (
        WSIGrid(slides=slides, resolution=0, unit="level")
        .then(AttachROIs(providers=[RectROIProvider(rois_dict)]))
        .then(PatchExtractor(tile_size=256, stride=256))
        .to(NumpyStreamWriter(layout="NCHW"))
    )

    start_time = time.time()
    stream = p.stream(num_workers=4, verbosity_level="INFO")
    for wsi_id, final_images, final_coords, meta in stream:
        ...

    logging.warning(f"Done in {time.time() - start_time:.1f} seconds.")


if __name__ == "__main__":
    main()
