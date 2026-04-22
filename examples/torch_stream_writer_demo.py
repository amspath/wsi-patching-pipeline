import logging
import time

from wsi_patching.core import PatchExtractor, WSIGrid
from wsi_patching.filtering import PenArtifactFilter
from wsi_patching.regions_of_interest import AttachROIs, RectROIfromXMLProvider
from wsi_patching.transforms import MacenkoNormalizer
from wsi_patching.writers import TorchStreamWriter


def main():
    # Example usage (adjust 'slides' to your real paths)
    slides = ["./data/RT14-09099_HE.tiff"]

    roi_xml = {"RT14-09099_HE": "./data/RT14-09099_HE.xml"}

    p = (
        # Since both macenko and PenArtifaceFilter can run on GPU, we set use_gpu=True here. Adjust as needed.
        WSIGrid(slides=slides, resolution=2, unit="level", use_gpu=True)
        .then(AttachROIs(providers=[RectROIfromXMLProvider(rois=roi_xml, annotation_level=0)]))
        .then(PatchExtractor(tile_size=256, stride=256, max_batch_size=800))
        .then(PenArtifactFilter())
        .then(MacenkoNormalizer())
        .to(TorchStreamWriter(layout="NCHW"))
    )

    start_time = time.time()
    stream = p.stream(num_workers=2)

    for wsi_id, images_t, coords_t, metadata in stream:
        ...

    logging.warning(f"Done in {time.time() - start_time:.1f} seconds.")


if __name__ == "__main__":
    main()
