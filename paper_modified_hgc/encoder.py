"""Independent 3D-FPN encoder used by the paper-modified HGC arm.

The block topology follows the pinned ``scdm_final@a344dcf`` reference, but
the implementation is local to HGC-Net and has no runtime dependency on that
checkout.  It deliberately stops at a 128D voxel feature grid; the
paper-modified dense quality/orientation/contact heads are defined in the
local ``paper_modified_hgc.model`` module.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    """Match reference ``out_channels // 8`` GroupNorm groups.

    The production channels (32, 64, 128, 256) therefore use 4, 8, 16 and
    32 groups respectively.  Tiny CPU fixtures may use fewer than eight
    channels; in that case the reference expression would produce zero, so
    clamp to one and walk down until the group count divides the channels.
    """

    channels = int(channels)
    if channels <= 0:
        raise ValueError("GroupNorm channels must be positive")
    groups = max(1, channels // 8)
    while groups > 1 and channels % groups:
        groups -= 1
    return groups


class ConvBlock3D(nn.Module):
    """Reference downsampling block with residual skip and replicate padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        scale: int = 2,
        align_corners: bool = True,
        skip_connection: bool = True,
        padding_mode: str = "replicate",
    ) -> None:
        super().__init__()
        self.skip_connection = bool(skip_connection)
        scale = int(scale)
        if scale <= 0 or (not align_corners and scale < 2):
            raise ValueError("3D-FPN scales must be positive; align_corners=False requires scale >= 2")
        self.use_pooling = not bool(align_corners)
        stride = scale if not self.use_pooling else scale // 2
        self.down = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            padding_mode=padding_mode,
        )
        self.conv1 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
        )
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
        )
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        if self.use_pooling:
            # This is the reference ConvBlock path for align_corners=False:
            # a reduced-stride convolution followed by 2x max pooling.
            x = F.max_pool3d(x, kernel_size=2, stride=2)
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.skip_connection:
            out = x + out
        return self.activation(out)


class ThreeDFPNEncoder(nn.Module):
    """Four-level 3D-FPN returning a full-resolution local feature grid."""

    def __init__(
        self,
        *,
        in_channels: int = 2,
        channels: Sequence[int] = (32, 64, 128, 256),
        scales: Sequence[int] = (2, 2, 2, 4),
        out_channels: int = 128,
        align_corners: bool = True,
        skip_connection: bool = True,
        padding_mode: str = "replicate",
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in channels)
        scales = tuple(int(value) for value in scales)
        if len(channels) != 4 or len(scales) != 4:
            raise ValueError("3D-FPN requires four channel and scale values")
        if any(value <= 0 for value in channels + scales) or int(out_channels) <= 0:
            raise ValueError("3D-FPN channels, scales, and output channels must be positive")
        self.in_channels = int(in_channels)
        self.channels = channels
        self.scales = scales
        self.out_channels = int(out_channels)
        self.align_corners = bool(align_corners)
        down_kwargs = {
            "align_corners": self.align_corners,
            "skip_connection": bool(skip_connection),
            "padding_mode": padding_mode,
        }
        self.down1 = ConvBlock3D(self.in_channels, channels[0], scale=scales[0], **down_kwargs)
        self.down2 = ConvBlock3D(channels[0], channels[1], scale=scales[1], **down_kwargs)
        self.down3 = ConvBlock3D(channels[1], channels[2], scale=scales[2], **down_kwargs)
        self.down4 = ConvBlock3D(channels[2], channels[3], scale=scales[3], **down_kwargs)

        self.latlayer4 = nn.Conv3d(channels[3], self.out_channels, kernel_size=1)
        self.latlayer3 = nn.Conv3d(channels[2], self.out_channels, kernel_size=1)
        self.latlayer2 = nn.Conv3d(channels[1], self.out_channels, kernel_size=1)
        self.latlayer1 = nn.Conv3d(channels[0], self.out_channels, kernel_size=1)
        self.latlayer0 = nn.Conv3d(self.in_channels, self.out_channels, kernel_size=1)
        self.smooth0 = nn.Conv3d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
        )
        self.smooth1 = nn.Conv3d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
        )
        self.smooth2 = nn.Conv3d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
        )
        self.smooth3 = nn.Conv3d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
        )
        self.smooth4 = nn.Conv3d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        if grid.ndim != 5 or grid.shape[1] != self.in_channels:
            raise ValueError(
                f"expected voxel input Bx{self.in_channels}xDxHxW, got {tuple(grid.shape)}"
            )
        f1 = self.down1(grid)
        f2 = self.down2(f1)
        f3 = self.down3(f2)
        f4 = self.down4(f3)
        g4 = self.latlayer4(f4)
        g3 = F.interpolate(
            g4, size=f3.shape[-3:], mode="trilinear", align_corners=self.align_corners
        ) + self.latlayer3(f3)
        g2 = F.interpolate(
            g3, size=f2.shape[-3:], mode="trilinear", align_corners=self.align_corners
        ) + self.latlayer2(f2)
        g1 = F.interpolate(
            g2, size=f1.shape[-3:], mode="trilinear", align_corners=self.align_corners
        ) + self.latlayer1(f1)
        # The smooth layers are intentionally retained at each FPN level as in
        # the source reference; h0 is the full-resolution local feature grid.
        self.smooth1(g1)
        self.smooth2(g2)
        self.smooth3(g3)
        self.smooth4(g4)
        g0 = F.interpolate(
            g1, size=grid.shape[-3:], mode="trilinear", align_corners=self.align_corners
        ) + self.latlayer0(grid)
        return self.smooth0(g0)

    def select(self, feature_grid: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Gather local features at DHW/ZYX integer indices as ``B,P,C``."""

        if feature_grid.ndim != 5:
            raise ValueError(f"feature_grid must be BCDHW, got {tuple(feature_grid.shape)}")
        if indices.ndim != 3 or indices.shape[-1] != 3:
            raise ValueError(f"indices must be BP3, got {tuple(indices.shape)}")
        if indices.shape[0] != feature_grid.shape[0]:
            raise ValueError("feature_grid and indices batch dimensions differ")
        size_d, size_h, size_w = feature_grid.shape[-3:]
        indices = indices.to(device=feature_grid.device, dtype=torch.long)
        if torch.any(indices[..., 0] < 0) or torch.any(indices[..., 0] >= size_d):
            raise ValueError("feature indices exceed depth bounds")
        if torch.any(indices[..., 1] < 0) or torch.any(indices[..., 1] >= size_h):
            raise ValueError("feature indices exceed height bounds")
        if torch.any(indices[..., 2] < 0) or torch.any(indices[..., 2] >= size_w):
            raise ValueError("feature indices exceed width bounds")
        linear = (
            indices[..., 0] * size_h * size_w
            + indices[..., 1] * size_w
            + indices[..., 2]
        )
        flat = feature_grid.flatten(start_dim=2).transpose(1, 2)
        channels = flat.shape[-1]
        selected = torch.gather(flat, 1, linear.unsqueeze(-1).expand(-1, -1, channels))
        return selected


__all__ = ["ConvBlock3D", "ThreeDFPNEncoder"]
