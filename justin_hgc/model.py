"""Official PointNet++ body with a native Justin-right output boundary.

This file intentionally does not edit ``model.py``.  The set-abstraction and
feature-propagation stack below is structurally the official upstream stack; the
only learned boundary change is three Justin template heads with 12+12 joint
targets instead of five DLR taxonomies with one 20-DoF target.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bin_pose import DEFAULT_BIN_SPEC, BinPoseSpec, bin_regression_loss, target_range_counts


def _pointnet2_components() -> tuple[Any, Any, Any]:
    """Load the pinned legacy PointNet++ package only when a full model is built."""
    package_root = Path(__file__).resolve().parents[1] / "pointnet2"
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    try:
        from pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModule
    except ModuleNotFoundError as error:
        if error.name == "pointnet2_cuda":
            raise RuntimeError(
                "The pinned PointNet++ CUDA extension (pointnet2_cuda) is not built for this Python/Torch/CUDA environment. "
                "The official repository bundles only Python 3.6/3.7 wheels; rebuild its extension in a compatible isolated environment before GPU forward/training."
            ) from error
        raise
    return PointnetFPModule, PointnetSAModule, PointnetSAModule


class JustinHGCOutputHead(nn.Module):
    """Pointwise official-HGC head with three native Justin grasp templates."""

    def __init__(self, *, num_templates: int = 3, bin_spec: BinPoseSpec = DEFAULT_BIN_SPEC) -> None:
        super().__init__()
        self.num_templates = int(num_templates)
        self.bin_spec = bin_spec
        self.channels_per_template = 2 + bin_spec.channels + 24  # gp + pose + q_contact/q_squeeze
        self.conv1 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(128, self.channels_per_template * self.num_templates, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return gp, pose, q_contact, q_squeeze as B,N,*,template tensors."""
        if feature.ndim != 3 or feature.shape[1] != 128:
            raise ValueError(f"expected PointNet++ feature Bx128xN, got {tuple(feature.shape)}")
        feature = F.leaky_relu(self.bn1(self.conv1(feature)), negative_slope=0.2)
        feature = self.drop1(feature)
        feature = F.leaky_relu(self.bn2(self.conv2(feature)), negative_slope=0.2)
        feature = self.drop2(feature)
        batch, _, points = feature.shape
        prediction = self.conv3(feature).permute(0, 2, 1).reshape(
            batch, points, self.channels_per_template, self.num_templates
        )
        gp = prediction[:, :, :2, :]
        pose_end = 2 + self.bin_spec.channels
        pose = prediction[:, :, 2:pose_end, :]
        q_contact = prediction[:, :, pose_end : pose_end + 12, :]
        q_squeeze = prediction[:, :, pose_end + 12 : pose_end + 24, :]
        return gp, pose, q_contact, q_squeeze

    def loss(
        self,
        prediction: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch: dict[str, torch.Tensor],
        *,
        graspable_weight: tuple[float, float] = (1.0, 10.0),
    ) -> dict[str, torch.Tensor]:
        """Use upstream sparse CE + bin/residual loss per Justin template."""
        gp, pose, q_contact, q_squeeze = prediction
        labels = batch["template_graspable"].long()
        pose_target = batch["template_pose"].to(dtype=pose.dtype)
        contact_target = batch["template_q_contact"].to(dtype=q_contact.dtype)
        squeeze_target = batch["template_q_squeeze"].to(dtype=q_squeeze.dtype)
        if labels.shape != (gp.shape[0], gp.shape[1], self.num_templates):
            raise ValueError("template_graspable shape does not match prediction")
        weight = torch.tensor(graspable_weight, dtype=gp.dtype, device=gp.device)
        total = gp.new_zeros(())
        result: dict[str, torch.Tensor] = {}
        for template in range(self.num_templates):
            label = labels[:, :, template]
            supervised = label >= 0
            positive = label > 0
            if not torch.any(supervised):
                continue
            gp_loss = F.cross_entropy(gp[:, :, :, template][supervised], label[supervised], weight=weight)
            template_loss = gp_loss
            result[f"template{template}/gp_loss"] = gp_loss
            if torch.any(positive):
                selected_target = pose_target[:, :, template][positive]
                range_counts = target_range_counts(selected_target, self.bin_spec)
                if any(range_counts.values()):
                    raise ValueError(
                        f"Justin pose targets outside configured bin range for template {template}: {range_counts}"
                    )
                _, pose_loss = bin_regression_loss(pose[:, :, :, template][positive], selected_target, self.bin_spec)
                contact_loss = F.mse_loss(q_contact[:, :, :, template][positive], contact_target[:, :, template][positive])
                squeeze_loss = F.mse_loss(q_squeeze[:, :, :, template][positive], squeeze_target[:, :, template][positive])
                template_loss = template_loss + pose_loss + contact_loss + squeeze_loss
                result[f"template{template}/pose_loss"] = pose_loss
                result[f"template{template}/q_contact_loss"] = contact_loss
                result[f"template{template}/q_squeeze_loss"] = squeeze_loss
            result[f"template{template}/loss"] = template_loss
            total = total + template_loss
        result["total_loss"] = total
        return result


class JustinPointNet2(nn.Module):
    """Pinned PointNet++ encoder and a native Justin HGC output head.

    Instantiation intentionally requires the official CUDA extension.  Head/loss
    structural tests remain CPU-safe through :class:`JustinHGCOutputHead`.
    """

    def __init__(self, *, num_templates: int = 3, bin_spec: BinPoseSpec = DEFAULT_BIN_SPEC) -> None:
        super().__init__()
        PointnetFPModule, PointnetSAModule, _ = _pointnet2_components()
        # These layer parameters are copied verbatim from the immutable upstream
        # backbone_pointnet2 with use_norm_points=True.
        self.sa1 = PointnetSAModule(mlp=[3, 32, 32, 64], npoint=1024, radius=0.1, nsample=32, bn=True)
        self.sa2 = PointnetSAModule(mlp=[64, 64, 64, 128], npoint=256, radius=0.2, nsample=64, bn=True)
        self.sa3 = PointnetSAModule(mlp=[128, 128, 128, 256], npoint=64, radius=0.4, nsample=128, bn=True)
        self.sa4 = PointnetSAModule(mlp=[256, 256, 256, 512], npoint=None, radius=None, nsample=None, bn=True)
        self.fp4 = PointnetFPModule(mlp=[768, 256, 256])
        self.fp3 = PointnetFPModule(mlp=[384, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[320, 256, 128])
        self.fp1 = PointnetFPModule(mlp=[134, 128, 128])
        self.head = JustinHGCOutputHead(num_templates=num_templates, bin_spec=bin_spec)

    def encode(self, xyz: torch.Tensor, normalized_points: torch.Tensor) -> torch.Tensor:
        l1_xyz, l1_points = self.sa1(xyz.contiguous(), normalized_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        return self.fp1(xyz.contiguous(), l1_xyz, torch.cat((xyz.transpose(1, 2), normalized_points), dim=1), l1_points)

    def forward(self, xyz: torch.Tensor, normalized_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.head(self.encode(xyz, normalized_points))
