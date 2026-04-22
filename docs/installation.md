# Installation

Python **≥ 3.10, < 3.14** is required. Python 3.14 is not yet supported.

## CPU

```bash
pip install wsi-patching
```

## GPU

```bash
pip install wsi-patching[gpu]
```

The `[gpu]` extra adds `torch`, `torchvision`, and `cupy-cuda12x` (Linux only). The following components require it:

| Component | Reason |
|---|---|
| `PenArtifactFilter` | Batched pen-colour detection runs on CUDA |
| `CellVitTissueClassifierFilter` | MobileNetV3 classifier runs on GPU |
| `MacenkoNormalizer` | Stain normalisation uses CuPy for speed |
| `TorchStreamWriter` | Output type is `torch.Tensor` |

All GPU components fail clearly at import time with an `ImportError` if the extras are missing.

## Development

```bash
git clone https://github.com/amspath/wsi-patching-pipeline.git
cd wsi-patching-pipeline
uv sync --extra dev,gpu
```

[uv](https://github.com/astral-sh/uv) is recommended. Plain pip also works:

```bash
pip install -e ".[dev,gpu]"
```

---

!!! note "Pinned `fastslide` version"
    The slide-reading backend `fastslide==0.5.2` is a hard-pinned dependency. If you see version conflicts from another package requiring a different `fastslide` version, isolate `wsi-patching` in its own virtual environment.
