from __future__ import annotations

from wsi_patching.core import ReadWindowChunker, RegionReadAndBatch, TilePlanner, WSIGrid
from wsi_patching.writers import NumpyMemoryWriter


def main():
    # Example usage (adjust 'slides' to your real paths)
    slides = ["./data/RBIO-GC072-HE-01.tiff", "./data/RBIO-GC072-HE-02.tiff"]

    p = (
        WSIGrid(slides=slides, tile_size=256, stride=256, level=0, use_gpu=True)
        .then(TilePlanner())
        .then(ReadWindowChunker())
        .then(RegionReadAndBatch(batch_size=800, num_workers=4))
        .to(NumpyMemoryWriter(layout="NCHW"))
    )

    np_patch_array, np_coords_array, list_of_wsi_ids = p.run(cpu_processes=2, profile=False, verbosity_level="INFO")
    print(
        (
            f"NumPy dataset: patches shape={np_patch_array.shape}, "
            f"coords shape={np_coords_array.shape}, wsi_ids count={len(list_of_wsi_ids)}"
        )
    )


if __name__ == "__main__":
    main()
