# src/wsi_patching/__init__.py
from importlib.metadata import version as _version

from .annotations import AddAnnotationFromGeoJSON, AddCellAnnotationFromCSV
from .core import PatchExtractor, ReadWindowChunker, RegionReadAndBatch, TilePlanner, WSIGrid
from .encoders import PNGEncoder
from .filtering import LowContrastBackgroundFilter, OtsuFilter, PenArtifactFilter, RemoveEdgeTiles
from .regions_of_interest import AttachROIs, RectROIfromXMLProvider, RectROIProvider, WholeSlideProvider
from .utils import visualize_audit, visualize_selected_patches
from .writers import NumpyStreamWriter, TorchStreamWriter, WebDatasetWriter

__all__ = [
    # Core pipeline
    "WSIGrid",
    "TilePlanner",
    "PatchExtractor",
    "ReadWindowChunker",
    "RegionReadAndBatch",
    # Regions of interest
    "AttachROIs",
    "RectROIProvider",
    "RectROIfromXMLProvider",
    "WholeSlideProvider",
    # Annotations
    "AddCellAnnotationFromCSV",
    "AddAnnotationFromGeoJSON",
    # Encoders
    "PNGEncoder",
    # Filters
    "LowContrastBackgroundFilter",
    "OtsuFilter",
    "PenArtifactFilter",
    "RemoveEdgeTiles",
    # Writers
    "NumpyStreamWriter",
    "WebDatasetWriter",
    "TorchStreamWriter",
    # Visualization / auditing
    "visualize_audit",
    "visualize_selected_patches",
]

# version (from your installed package metadata)
try:
    __version__ = _version("wsi_patching")
except Exception:  # during editable installs before metadata exists
    __version__ = "0.1.0"
