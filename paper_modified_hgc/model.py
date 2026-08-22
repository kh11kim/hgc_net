"""Paper-modified 3D-FPN encoder with the shared Justin HGC output head."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from justin_hgc.bin_pose import DEFAULT_BIN_SPEC, BinPoseSpec
from justin_hgc.model import JustinHGCOutputHead

from .encoder import ThreeDFPNEncoder


class PaperModifiedHGC(nn.Module):
    """Voxel arm whose only arm-specific part is encoder/spatial projection.

    The ``JustinHGCOutputHead`` instance is the exact same head used by the
    PointNet++ arm.  It predicts three template-wise graspability, pose
    bin/residual, ``q_contact`` and ``q_squeeze`` outputs; ``q_open`` remains a
    fixed template datum and is never a learned output.
    """

    def __init__(
        self,
        *,
        in_channels: int = 2,
        encoder_channels: Sequence[int] = (32, 64, 128, 256),
        encoder_scales: Sequence[int] = (2, 2, 2, 4),
        encoder_out_channels: int = 128,
        align_corners: bool = True,
        num_templates: int = 3,
        bin_spec: BinPoseSpec = DEFAULT_BIN_SPEC,
    ) -> None:
        super().__init__()
        self.encoder = ThreeDFPNEncoder(
            in_channels=in_channels,
            channels=encoder_channels,
            scales=encoder_scales,
            out_channels=encoder_out_channels,
            align_corners=align_corners,
        )
        self.local_projection = (
            nn.Identity()
            if int(encoder_out_channels) == 128
            else nn.Conv1d(int(encoder_out_channels), 128, kernel_size=1)
        )
        self.head = JustinHGCOutputHead(num_templates=num_templates, bin_spec=bin_spec)

    def encode(self, input_grid: torch.Tensor) -> torch.Tensor:
        """Return the full-resolution ``B,128,D,H,W`` local feature grid."""

        features = self.encoder(input_grid.float())
        if isinstance(self.local_projection, nn.Identity):
            return features
        # Projection is pointwise over the voxel grid and keeps DHW alignment.
        batch, channels, depth, height, width = features.shape
        projected = self.local_projection(features.reshape(batch, channels, -1))
        return projected.reshape(batch, 128, depth, height, width)

    def forward_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Run the shared head at the batch's approach-point voxel locations."""

        if "input_grid" not in batch:
            raise KeyError("paper-modified batch must contain input_grid")
        if "feature_indices" not in batch:
            raise KeyError("paper-modified batch must contain feature_indices")
        feature_grid = self.encode(batch["input_grid"])
        local = self.encoder.select(feature_grid, batch["feature_indices"])
        # JustinHGCOutputHead accepts B,128,P while the FPN gather naturally
        # returns B,P,128.
        return self.head(local.transpose(1, 2).contiguous())

    def forward(
        self,
        input_grid: torch.Tensor | dict[str, torch.Tensor],
        feature_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if isinstance(input_grid, dict):
            if feature_indices is not None:
                raise ValueError("feature_indices must be omitted when passing a batch dictionary")
            return self.forward_batch(input_grid)
        if feature_indices is None:
            raise ValueError("feature_indices is required for tensor forward")
        return self.forward_batch({"input_grid": input_grid, "feature_indices": feature_indices})


__all__ = ["PaperModifiedHGC"]
