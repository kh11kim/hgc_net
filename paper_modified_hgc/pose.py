"""Pose representation used by the paper-modified 3D-FPN HGC arm.

This is the small pose codec from the paper-modified reference lineage.  It
is deliberately independent of :mod:`justin_hgc.bin_pose`: that codec belongs
to the upstream PointNet++ arm and predicts depth/angular bins, whereas this
arm predicts one 12*6*12 orientation bin and a continuous four-vector
``(depth, azimuth residual, elevation residual, roll residual)``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


BINS_CONFIG = {
    "azimuth": 12,
    "elevation": 6,
    "roll": 12,
}
ORIENTATION_BINS = 12 * 6 * 12


def mat_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    """Flatten the first two rotation columns in the reference order."""

    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must end in (3,3), got {tuple(matrix.shape)}")
    return matrix[..., :3, :2].transpose(-1, -2).reshape(*matrix.shape[:-2], 6)


def rot6d_to_mat(rot6d: torch.Tensor) -> torch.Tensor:
    """Restore an orthonormal rotation matrix from the continuous 6D form."""

    if rot6d.shape[-1] != 6:
        raise ValueError(f"rot6d must end in 6 values, got {tuple(rot6d.shape)}")
    x_raw, y_raw = rot6d[..., :3], rot6d[..., 3:]
    x = F.normalize(x_raw, p=2, dim=-1)
    y_ortho = y_raw - (x * y_raw).sum(dim=-1, keepdim=True) * x
    y_norm = torch.linalg.vector_norm(y_ortho, dim=-1, keepdim=True)
    singular = y_norm < 1e-8
    z_axis = torch.zeros_like(x)
    z_axis[..., 2] = 1.0
    x_axis = torch.zeros_like(x)
    x_axis[..., 0] = 1.0
    fallback = torch.where((x[..., 2:3].abs() > 0.99), x_axis, z_axis)
    y_safe = torch.where(singular, fallback, y_raw)
    y_safe = y_safe - (x * y_safe).sum(dim=-1, keepdim=True) * x
    y = F.normalize(y_safe, p=2, dim=-1)
    z = torch.cross(x, y, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def rotation_matrix_to_angles(rotation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ZYX azimuth/elevation/roll angles in radians."""

    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"rotation must end in (3,3), got {tuple(rotation.shape)}")
    batch_shape = rotation.shape[:-2]
    flat = rotation.reshape(-1, 3, 3)
    device = rotation.device
    sy = -flat[:, 2, 0]
    elevation = torch.asin(sy.clamp(-1.0, 1.0))
    gimbal = sy.abs() > 0.99999
    azimuth = torch.zeros_like(sy, device=device)
    roll = torch.zeros_like(sy, device=device)
    regular = ~gimbal
    if torch.any(regular):
        azimuth[regular] = torch.atan2(flat[regular, 1, 0], flat[regular, 0, 0])
        roll[regular] = torch.atan2(flat[regular, 2, 1], flat[regular, 2, 2])
    if torch.any(gimbal):
        # Same singular branch as the established reference implementation.
        azimuth[gimbal] = torch.atan2(-flat[gimbal, 0, 1], flat[gimbal, 1, 1])
    return (
        azimuth.reshape(batch_shape),
        elevation.reshape(batch_shape),
        roll.reshape(batch_shape),
    )


def mat_to_bin_and_residual(rotation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode rotations as one 864-way bin and three continuous residuals."""

    azimuth, elevation, roll = rotation_matrix_to_angles(rotation)
    num_az = BINS_CONFIG["azimuth"]
    num_el = BINS_CONFIG["elevation"]
    num_roll = BINS_CONFIG["roll"]
    bin_az = torch.clamp(((azimuth + math.pi) / (2.0 * math.pi) * num_az).long(), 0, num_az - 1)
    bin_el = torch.clamp(((elevation + math.pi / 2.0) / math.pi * num_el).long(), 0, num_el - 1)
    bin_roll = torch.clamp(((roll + math.pi) / (2.0 * math.pi) * num_roll).long(), 0, num_roll - 1)
    single_bin = bin_roll * (num_az * num_el) + bin_el * num_az + bin_az
    az_center = (bin_az.float() + 0.5) * (2.0 * math.pi / num_az) - math.pi
    el_center = (bin_el.float() + 0.5) * (math.pi / num_el) - math.pi / 2.0
    roll_center = (bin_roll.float() + 0.5) * (2.0 * math.pi / num_roll) - math.pi
    residual = torch.stack(
        (azimuth - az_center, elevation - el_center, roll - roll_center), dim=-1
    )
    return single_bin, residual


def bin_and_residual_to_mat(single_bin: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Decode one orientation bin and three angular residuals to ``(...,3,3)``."""

    if residual.shape[-1] != 3 or single_bin.shape != residual.shape[:-1]:
        raise ValueError("single_bin and residual must have matching leading dimensions")
    shape = single_bin.shape
    flat_bin = single_bin.reshape(-1).long()
    flat_residual = residual.reshape(-1, 3)
    num_az = BINS_CONFIG["azimuth"]
    num_el = BINS_CONFIG["elevation"]
    num_roll = BINS_CONFIG["roll"]
    bin_az = flat_bin % num_az
    tmp = flat_bin // num_az
    bin_el = tmp % num_el
    bin_roll = tmp // num_el
    azimuth = (bin_az.float() + 0.5) * (2.0 * math.pi / num_az) - math.pi + flat_residual[:, 0]
    elevation = (bin_el.float() + 0.5) * (math.pi / num_el) - math.pi / 2.0 + flat_residual[:, 1]
    roll = (bin_roll.float() + 0.5) * (2.0 * math.pi / num_roll) - math.pi + flat_residual[:, 2]
    zeros = torch.zeros_like(azimuth)
    ones = torch.ones_like(azimuth)
    ca, sa = torch.cos(azimuth), torch.sin(azimuth)
    ce, se = torch.cos(elevation), torch.sin(elevation)
    cr, sr = torch.cos(roll), torch.sin(roll)
    r_z = torch.stack(
        (
            torch.stack((ca, -sa, zeros), dim=1),
            torch.stack((sa, ca, zeros), dim=1),
            torch.stack((zeros, zeros, ones), dim=1),
        ),
        dim=1,
    )
    r_y = torch.stack(
        (
            torch.stack((ce, zeros, se), dim=1),
            torch.stack((zeros, ones, zeros), dim=1),
            torch.stack((-se, zeros, ce), dim=1),
        ),
        dim=1,
    )
    r_x = torch.stack(
        (
            torch.stack((ones, zeros, zeros), dim=1),
            torch.stack((zeros, cr, -sr), dim=1),
            torch.stack((zeros, sr, cr), dim=1),
        ),
        dim=1,
    )
    return torch.bmm(r_z, torch.bmm(r_y, r_x)).reshape(*shape, 3, 3)


def pose9d_to_targets(
    palm_pose9d: torch.Tensor, grasp_point: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode canonical palm pose and surface anchor to depth/bin targets."""

    if palm_pose9d.shape[:-1] != grasp_point.shape[:-1] or palm_pose9d.shape[-1] != 9 or grasp_point.shape[-1] != 3:
        raise ValueError("palm_pose9d and grasp_point must have matching (...,9)/(..,3) shapes")
    relative_position = palm_pose9d[..., :3] - grasp_point
    depth = torch.linalg.vector_norm(relative_position, dim=-1)
    rotation = rot6d_to_mat(palm_pose9d[..., 3:])
    orientation_bin, angular_residual = mat_to_bin_and_residual(rotation)
    return orientation_bin, torch.cat((depth.unsqueeze(-1), angular_residual), dim=-1)


def targets_to_pose9d(
    orientation_bin: torch.Tensor,
    residual: torch.Tensor,
    grasp_point: torch.Tensor,
) -> torch.Tensor:
    """Decode depth/orientation targets back to canonical palm ``pose9d``."""

    if residual.shape[-1] != 4 or orientation_bin.shape != residual.shape[:-1] or grasp_point.shape != (*residual.shape[:-1], 3):
        raise ValueError("orientation_bin, residual and grasp_point shapes do not match")
    rotation = bin_and_residual_to_mat(orientation_bin, residual[..., 1:])
    palm_position = grasp_point - rotation[..., :, 2] * residual[..., :1]
    return torch.cat((palm_position, mat_to_rot6d(rotation)), dim=-1)


__all__ = [
    "BINS_CONFIG",
    "ORIENTATION_BINS",
    "bin_and_residual_to_mat",
    "mat_to_bin_and_residual",
    "mat_to_rot6d",
    "pose9d_to_targets",
    "rot6d_to_mat",
    "rotation_matrix_to_angles",
    "targets_to_pose9d",
]
