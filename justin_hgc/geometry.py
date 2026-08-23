"""Frame-explicit geometry for canonical unified-v4 views."""

from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np


GRID_EDGE_M = 0.5
GRID_HALF_EDGE_M = GRID_EDGE_M / 2.0
POINT_COUNT = 25_000


def deproject_depth_m(
    depth: np.ndarray, intrinsics: Mapping[str, float], *, depth_scale: float
) -> np.ndarray:
    """Deproject positive depth pixels to the camera frame in metres.

    Unified-v3 depth PNG values are unsigned millimetres and its manifest pins
    ``depth_scale: 0.001``.  Zero remains the only invalid-depth convention.
    """
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth.shape}")
    for key in ("fx", "fy", "cx", "cy"):
        if key not in intrinsics:
            raise KeyError(f"camera intrinsics missing {key!r}")
    v, u = np.nonzero(depth > 0)
    z = depth[v, u].astype(np.float64, copy=False) * float(depth_scale)
    x = (u.astype(np.float64) - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    y = (v.astype(np.float64) - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
    return np.stack((x, y, z), axis=-1).astype(np.float32)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a camera-to-reconstruction-grid homogeneous transform."""
    points = np.asarray(points, dtype=np.float32)
    transform = np.asarray(transform, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be Nx3, got {points.shape}")
    if transform.shape != (4, 4):
        raise ValueError(f"transform must be 4x4, got {transform.shape}")
    return (points @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)


def crop_centered_grid(points: np.ndarray, *, edge_length_m: float = GRID_EDGE_M) -> tuple[np.ndarray, np.ndarray]:
    """Keep the canonical reconstruction grid: a 0.5 m cube centred at zero."""
    points = np.asarray(points, dtype=np.float32)
    half = float(edge_length_m) / 2.0
    mask = np.all((points >= -half) & (points <= half), axis=1)
    return points[mask], mask


def stable_seed(key: str, *, salt: str = "issue59") -> int:
    payload = f"{salt}:{key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def deterministic_fixed_sample(points: np.ndarray, *, count: int = POINT_COUNT, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Sample a fixed cloud without fabricating geometry.

    A long view gets distinct points only.  A short view keeps every real point
    first, then repeats deterministically selected existing points for the exact
    shortfall.  The returned indices make that policy auditable in a smoke run.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be Nx3, got {points.shape}")
    if len(points) == 0:
        raise ValueError("no valid depth points remain in the centred 0.5 m grid")
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    rng = np.random.default_rng(stable_seed(key))
    if len(points) >= count:
        indices = rng.choice(len(points), size=count, replace=False)
    else:
        repeated = rng.choice(len(points), size=count - len(points), replace=True)
        indices = np.concatenate((np.arange(len(points), dtype=np.int64), repeated))
    return points[indices], indices.astype(np.int64, copy=False)


def deterministic_seed_sample(
    points: np.ndarray, *, count: int = POINT_COUNT, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sample exactly ``count`` existing points using an evaluation seed.

    This is the runtime counterpart of :func:`deterministic_fixed_sample`.
    Every returned row is an existing cropped point; when a view has fewer
    than ``count`` points, deterministic replacement repeats real rows rather
    than fabricating geometry.  The explicit integer seed is part of the
    evaluation request and is intentionally not derived from a sample ID.
    """

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be Nx3, got {points.shape}")
    if len(points) == 0:
        raise ValueError("no valid depth points remain in the centred 0.5 m grid")
    if int(seed) != seed or int(seed) < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    if int(count) <= 0:
        raise ValueError(f"count must be positive, got {count}")
    rng = np.random.default_rng(int(seed))
    if len(points) >= int(count):
        indices = rng.choice(len(points), size=int(count), replace=False)
    else:
        repeated = rng.choice(len(points), size=int(count) - len(points), replace=True)
        indices = np.concatenate((np.arange(len(points), dtype=np.int64), repeated))
    indices = np.asarray(indices, dtype=np.int64)
    return np.ascontiguousarray(points[indices]), indices


def pose9d_to_matrix(pose9d: np.ndarray) -> np.ndarray:
    """Convert canonical pose9d (translation + first two rotation columns)."""
    pose9d = np.asarray(pose9d, dtype=np.float32)
    if pose9d.shape[-1] != 9:
        raise ValueError(f"palm_pose9d must end in 9 values, got {pose9d.shape}")
    x = pose9d[..., 3:6]
    y_raw = pose9d[..., 6:9]
    x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    y_ortho = y_raw - np.sum(x * y_raw, axis=-1, keepdims=True) * x
    y = y_ortho / np.maximum(np.linalg.norm(y_ortho, axis=-1, keepdims=True), 1e-12)
    z = np.cross(x, y)
    return np.stack((x, y, z), axis=-1)


def pose9d_to_bin_target(palm_pose9d: np.ndarray, approach_point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode canonical Justin palms in upstream HGC's four bin/residual fields.

    Canonical ``palm_pose9d`` local +z points from palm to approach point.  HGC
    predicts from a visible approach-point anchor back to the palm, hence the
    returned axis is ``-R[:, 2]``.  Depth is centimetres from that anchor.
    """
    palm_pose9d = np.asarray(palm_pose9d, dtype=np.float32)
    approach_point = np.asarray(approach_point, dtype=np.float32)
    if palm_pose9d.shape != (*approach_point.shape[:-1], 9) or approach_point.shape[-1] != 3:
        raise ValueError("palm_pose9d and approach_point must have matching Nx9/Nx3 shapes")
    rotation = pose9d_to_matrix(palm_pose9d)
    surface_to_palm = -rotation[..., :, 2]
    offset = palm_pose9d[..., :3] - approach_point
    depth_m = np.sum(offset * surface_to_palm, axis=-1)
    if np.any(depth_m < -1e-4):
        raise ValueError("canonical approach point lies behind its declared palm axis")
    depth_cm = np.maximum(depth_m, 0.0) * 100.0
    azimuth = np.degrees(np.arctan2(surface_to_palm[..., 1], surface_to_palm[..., 0])) % 360.0
    elevation = np.degrees(
        np.arctan2(-surface_to_palm[..., 2], np.hypot(surface_to_palm[..., 0], surface_to_palm[..., 1]))
        + np.pi / 2.0
    )
    closing = rotation[..., :, 1]
    grasp_angle = np.degrees(np.arctan2(closing[..., 1], closing[..., 0])) % 360.0
    target = np.stack((depth_cm, azimuth, elevation, grasp_angle), axis=-1).astype(
        np.float32
    )
    # Values infinitesimally below 360 degrees can round to exactly 360 during
    # float32 conversion. Re-wrap the two circular fields in their stored dtype
    # so every valid direction remains inside the half-open [0, 360) bins.
    target[..., 1] = np.remainder(target[..., 1], np.float32(360.0))
    target[..., 3] = np.remainder(target[..., 3], np.float32(360.0))
    return target, surface_to_palm.astype(np.float32)


def palm_axis_aligned_mask(
    palm_pose9d: np.ndarray,
    approach_point: np.ndarray,
    *,
    minimum_cosine: float = 0.9999,
) -> np.ndarray:
    """Select anchors representable by HGC's scalar approach-depth pose.

    DFC may fall back to a ray tilted by exactly 10 degrees when the direct
    palm-z ray misses the object.  HGC predicts only one depth scalar, so those
    tilted anchors cannot reconstruct the labelled palm translation and must
    not become contradictory positive pose labels.
    """
    palm_pose9d = np.asarray(palm_pose9d, dtype=np.float32)
    approach_point = np.asarray(approach_point, dtype=np.float32)
    if palm_pose9d.shape != (*approach_point.shape[:-1], 9):
        raise ValueError("palm_pose9d and approach_point must have matching Nx9/Nx3 shapes")
    offset = approach_point - palm_pose9d[..., :3]
    length = np.linalg.norm(offset, axis=-1)
    if np.any(length <= 1.0e-8):
        raise ValueError("approach point must differ from the palm translation")
    palm_z = pose9d_to_matrix(palm_pose9d)[..., :, 2]
    cosine = np.sum(offset * palm_z, axis=-1) / length
    return cosine >= float(minimum_cosine)
