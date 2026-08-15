"""Device-agnostic copy of the upstream HGC bin/residual pose objective.

Only the depth interval changes: canonical Justin palms are anchored at a visible
approach point rather than at the official DLR hand's fixed 20 cm offset.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BinPoseSpec:
    # The upstream angular bins are retained verbatim.  The upstream depth target
    # was (depth_cm - 20) in [0, 8]; Justin's observed physical depth is 0--26.03
    # cm on the 100-view contract sample, so use direct centimetres in [0, 28].
    depth_scope_cm: float = 28.0
    depth_bin_cm: float = 1.0
    azimuth_scope_deg: float = 360.0
    azimuth_bin_deg: float = 60.0
    elevation_scope_deg: float = 90.0
    elevation_bin_deg: float = 15.0
    grasp_angle_scope_deg: float = 360.0
    grasp_angle_bin_deg: float = 30.0

    @property
    def group_specs(self) -> tuple[tuple[str, float, float], ...]:
        return (
            ("depth", self.depth_scope_cm, self.depth_bin_cm),
            ("azimuth", self.azimuth_scope_deg, self.azimuth_bin_deg),
            ("elevation", self.elevation_scope_deg, self.elevation_bin_deg),
            ("grasp_angle", self.grasp_angle_scope_deg, self.grasp_angle_bin_deg),
        )

    @property
    def channels(self) -> int:
        return 2 * sum(int(scope / size) for _, scope, size in self.group_specs)


DEFAULT_BIN_SPEC = BinPoseSpec()


def target_range_counts(target: torch.Tensor, spec: BinPoseSpec = DEFAULT_BIN_SPEC) -> dict[str, int]:
    """Report, rather than hide, targets outside the configured bin intervals."""
    if target.shape[-1] != 4:
        raise ValueError(f"expected pose target (...,4), got {tuple(target.shape)}")
    return {
        name: int(((target[..., index] < 0) | (target[..., index] >= scope)).sum().item())
        for index, (name, scope, _) in enumerate(spec.group_specs)
    }


def bin_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    spec: BinPoseSpec = DEFAULT_BIN_SPEC,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Match upstream CE-bin plus sigmoid-residual SmoothL1 loss on any device."""
    if prediction.ndim != 2 or prediction.shape[1] != spec.channels:
        raise ValueError(f"expected prediction (N,{spec.channels}), got {tuple(prediction.shape)}")
    if target.shape != (len(prediction), 4):
        raise ValueError("target must be Nx4 and match prediction rows")
    total = prediction.new_zeros(())
    result: dict[str, torch.Tensor] = {}
    start = 0
    for target_index, (name, scope, size) in enumerate(spec.group_specs):
        bins = int(scope / size)
        logits = prediction[:, start : start + bins]
        residual_logits = prediction[:, start + bins : start + 2 * bins]
        start += 2 * bins
        value = target[:, target_index].clamp(0, scope - 1e-4)
        label = torch.floor(value / size).long()
        residual = (value - label.float() * size) / size
        one_hot = F.one_hot(label, num_classes=bins).to(dtype=prediction.dtype)
        classification = F.cross_entropy(logits, label)
        regression = F.smooth_l1_loss((torch.sigmoid(residual_logits) * one_hot).sum(dim=1), residual)
        result[f"{name}_bin_loss"] = classification
        result[f"{name}_res_loss"] = regression
        total = total + classification + regression
    result["pose_loss"] = total
    return result, total


def _decode_group(prediction: torch.Tensor, start: int, scope: float, size: float) -> tuple[torch.Tensor, int]:
    bins = int(scope / size)
    logits = prediction[:, start : start + bins]
    residual_logits = prediction[:, start + bins : start + 2 * bins]
    index = torch.argmax(logits, dim=1)
    # This deliberately preserves the official decoder's raw-residual behavior.
    residual = residual_logits.gather(1, index[:, None]).squeeze(1) * size
    return index.float() * size + size / 2.0 + residual, start + 2 * bins


def decode_pose_bins(prediction: torch.Tensor, spec: BinPoseSpec = DEFAULT_BIN_SPEC) -> torch.Tensor:
    """Decode four bin/residual fields, retaining the official raw residual decode."""
    if prediction.ndim != 2 or prediction.shape[1] != spec.channels:
        raise ValueError(f"expected prediction (N,{spec.channels}), got {tuple(prediction.shape)}")
    values = []
    start = 0
    for _, scope, size in spec.group_specs:
        value, start = _decode_group(prediction, start, scope, size)
        values.append(value)
    return torch.stack(values, dim=-1)


def bin_target_to_rotation(target: torch.Tensor) -> torch.Tensor:
    """Build R=[-cross(axis, closing), closing, surface_to_palm_axis]."""
    _, azimuth_deg, elevation_deg, grasp_angle_deg = target.unbind(dim=-1)
    azimuth = torch.deg2rad(azimuth_deg)
    elevation = torch.deg2rad(elevation_deg)
    x = torch.cos(azimuth)
    y = torch.sin(azimuth)
    z = -torch.tan(elevation - math.pi / 2.0) * torch.sqrt(x.square() + y.square())
    axis = F.normalize(torch.stack((x, y, z), dim=-1), dim=-1)
    closing_angle = torch.deg2rad(grasp_angle_deg)
    raw_closing = torch.stack((torch.cos(closing_angle), torch.sin(closing_angle), torch.zeros_like(closing_angle)), dim=-1)
    closing = raw_closing - (raw_closing * axis).sum(dim=-1, keepdim=True) * axis
    singular = closing.norm(dim=-1, keepdim=True) < 1e-6
    fallback = torch.stack((torch.zeros_like(x), torch.ones_like(x), torch.zeros_like(x)), dim=-1)
    closing = torch.where(singular, fallback - (fallback * axis).sum(dim=-1, keepdim=True) * axis, closing)
    closing = F.normalize(closing, dim=-1)
    first = -torch.cross(axis, closing, dim=-1)
    return torch.stack((first, closing, axis), dim=-1)
