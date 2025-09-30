from wsi_patching.core import ReadWindowChunker, RegionReadAndBatch, TilePlanner, WSIGrid
from wsi_patching.regions_of_interest import AttachROIs, RectROIfromXMLProvider
from wsi_patching.transforms import MacenkoNormalizer
from wsi_patching.writers import TorchMemoryWriter


def main():
    # Example usage (adjust 'slides' to your real paths)
    slides = ["./data/RT14-09099_HE.tiff"]

    roi_xml = {"RT14-09099_HE": "./data/RT14-09099_HE.xml"}

    p = (
        WSIGrid(slides=slides, tile_size=256, stride=256, level=0, use_gpu=True)
        .then(AttachROIs(RectROIfromXMLProvider(rois=roi_xml)))
        .then(TilePlanner())
        .then(ReadWindowChunker())
        .then(RegionReadAndBatch(batch_size=800, num_workers=4))
        .then(MacenkoNormalizer())
        .to(TorchMemoryWriter(layout="NCHW"))
    )

    torch_dataset = p.run(cpu_processes=2, profile=False, verbosity_level="INFO")
    print(torch_dataset)


if __name__ == "__main__":
    main()
