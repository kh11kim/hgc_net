"""Justin runtime candidate filtering with explicit canonical-frame semantics."""

from __future__ import annotations

import numpy as np


WORLD_UP = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)


def top_side_mask(grasp_point: np.ndarray, palm_position: np.ndarray) -> np.ndarray:
    """Keep candidates with their palm on the +world-z side of the surface point."""
    grasp_point = np.asarray(grasp_point, dtype=np.float32)
    palm_position = np.asarray(palm_position, dtype=np.float32)
    if grasp_point.shape != palm_position.shape or grasp_point.shape[-1:] != (3,):
        raise ValueError("grasp_point and palm_position must be matching Nx3 arrays")
    return np.einsum("...i,i->...", palm_position - grasp_point, WORLD_UP) > 0.0


def rotation_distance_rad(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first) @ np.asarray(second).T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def aggressive_nms(
    *,
    positions: np.ndarray,
    rotations: np.ndarray,
    scores: np.ndarray,
    distance_threshold_m: float = 0.03,
    angle_threshold_deg: float = 30.0,
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply the upstream HGC OR suppression rule exactly.

    A lower-scored candidate is suppressed when it is close in *either* position
    or orientation.  Equivalently, it survives only when both its position is
    farther than 3 cm and its rotation differs by more than 30 degrees.
    """
    positions = np.asarray(positions, dtype=np.float32)
    rotations = np.asarray(rotations, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be Nx3")
    if rotations.shape != (len(positions), 3, 3) or scores.shape != (len(positions),):
        raise ValueError("rotations/scores do not match positions")
    # Keep the official implementation's score ordering and iterative indexing,
    # including its behavior for equal scores, rather than substituting a library
    # NMS routine.
    remaining = np.argsort(scores)
    kept: list[int] = []
    angle_threshold = np.deg2rad(angle_threshold_deg)
    while len(remaining):
        current = int(remaining[-1])
        kept.append(current)
        survivors: list[int] = []
        for other in remaining[:-1]:
            far_enough = np.linalg.norm(positions[current] - positions[other]) > distance_threshold_m
            different_enough = rotation_distance_rad(rotations[current], rotations[other]) > angle_threshold
            if far_enough and different_enough:
                survivors.append(int(other))
        remaining = np.asarray(survivors, dtype=np.int64)
    return np.asarray(kept, dtype=np.int64), {"pre_nms": len(positions), "post_nms": len(kept)}
