"""Issue #59 canonical-v3 adapter for the official HGC-Net architecture.

The official files remain immutable.  This package makes the Justin-specific
input, output, and runtime boundary explicit and reviewable.
"""

from .data import JustinCanonicalDataset
from .model import JustinHGCOutputHead, JustinPointNet2

__all__ = ["JustinCanonicalDataset", "JustinHGCOutputHead", "JustinPointNet2"]
