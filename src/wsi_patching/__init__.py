# src/wsi_patching/__init__.py
from importlib.metadata import version as _version

__all__ = []

# version (from your installed package metadata)
try:
    __version__ = _version("wsi_patching")
except Exception:  # during editable installs before metadata exists
    __version__ = "0.1.0"
