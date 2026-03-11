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
        WSIGrid(slides=slides, resolution=2, unit="level", use_gpu=True)
        .then(AttachROIs(providers=[RectROIfromXMLProvider(rois=roi_xml, annotation_level=0)]))
        .then(PatchExtractor(tile_size=256, stride=256, max_batch_size=800, num_workers=4))
        .then(PenArtifactFilter())
        .then(MacenkoNormalizer())
        .to(TorchStreamWriter(layout="NCHW"))
    )

    stream = p.stream(cpu_processes=2, profile=False, verbosity_level="INFO")

    for wsi_id, images_t, coords_t, metadata in stream:
        ...


if __name__ == "__main__":
    main()
