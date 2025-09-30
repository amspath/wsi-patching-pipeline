from __future__ import annotations

from wsi_patching.core import ReadWindowChunker, RegionReadAndBatch, TilePlanner, WSIGrid
from wsi_patching.writers import TorchMemoryWriter


def main():
    # Example usage (adjust 'slides' to your real paths)
    slides = ["./data/RBIO-GC072-HE-01.tiff", "./data/RBIO-GC072-HE-02.tiff"]

    p = (
        WSIGrid(slides=slides, tile_size=256, stride=256, level=0, use_gpu=True)
        .then(TilePlanner())
        .then(ReadWindowChunker())
        .then(RegionReadAndBatch(batch_size=800, num_workers=4))
        .to(TorchMemoryWriter(layout="NCHW"))
    )

    torch_dataset = p.run(cpu_processes=2, profile=False, verbosity_level="INFO")
    print(torch_dataset)


if __name__ == "__main__":
    main()
