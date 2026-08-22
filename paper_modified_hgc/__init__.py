"""Paper-modified 3D-FPN arm for the Issue #59 Justin HGC trainer."""

from .data import PaperModifiedCanonicalDataset
from .encoder import ConvBlock3D, ThreeDFPNEncoder
from .model import PaperModifiedHGC

__all__ = [
    "ConvBlock3D",
    "PaperModifiedCanonicalDataset",
    "PaperModifiedHGC",
    "ThreeDFPNEncoder",
]
