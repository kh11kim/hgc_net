"""Paper-modified 3D-FPN arm for the Issue #59 Justin HGC trainer."""

from .data import PaperModifiedCanonicalDataset
from .encoder import ConvBlock3D, ThreeDFPNEncoder
from .model import (
    PaperModifiedHGC,
    PaperModifiedHGCHead,
    PaperModifiedPrediction,
    sigmoid_focal_loss,
)
from .pose import (
    ORIENTATION_BINS,
    bin_and_residual_to_mat,
    mat_to_bin_and_residual,
    pose9d_to_targets,
    targets_to_pose9d,
)
from .runtime import decode_paper_candidates

__all__ = [
    "ConvBlock3D",
    "PaperModifiedCanonicalDataset",
    "PaperModifiedHGC",
    "PaperModifiedHGCHead",
    "PaperModifiedPrediction",
    "ThreeDFPNEncoder",
    "ORIENTATION_BINS",
    "bin_and_residual_to_mat",
    "decode_paper_candidates",
    "mat_to_bin_and_residual",
    "pose9d_to_targets",
    "sigmoid_focal_loss",
    "targets_to_pose9d",
]
