"""Deterministic HGC negative-point derivation for canonical scene views."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import stable_seed


NEGATIVE_SEED_SALT = "issue59-negatives"


@dataclass(frozen=True)
class NegativePointSelection:
    """Negative rows in the deterministic fixed-size HGC point cloud."""

    negative_point_indices: np.ndarray
    positive_point_count: int
    remaining_point_count: int


def select_negative_point_indices(
    *,
    points: np.ndarray,
    approach_points: np.ndarray,
    negative_fraction: float,
    key: str,
    radius_m: float = 0.005,
) -> NegativePointSelection:
    """Select a deterministic subset of points outside every positive anchor.

    ``approach_points`` must contain all canonical positive grasps for the
    scene/view.  Visibility metadata is deliberately not used: the geometric
    radius test decides whether a positive anchor is represented in the view.
    Returned indices address the deterministic fixed-size HGC point cloud, not
    pixels or the pre-sampling cropped cloud.
    """
    points = np.asarray(points, dtype=np.float32)
    approach_points = np.asarray(approach_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be Nx3, got {points.shape}")
    if approach_points.ndim != 2 or approach_points.shape[1] != 3:
        raise ValueError(f"approach_points must be Mx3, got {approach_points.shape}")
    if not 0.0 <= negative_fraction <= 1.0:
        raise ValueError("negative_fraction must be in [0, 1]")
    if radius_m < 0.0:
        raise ValueError("radius_m must be non-negative")

    positive_rows = np.zeros(len(points), dtype=bool)
    radius_squared = float(radius_m) ** 2
    for approach_point in approach_points:
        delta = points - approach_point
        positive_rows |= np.einsum("ij,ij->i", delta, delta) <= radius_squared

    remaining = np.flatnonzero(~positive_rows)
    negative_count = int(np.floor(float(negative_fraction) * len(remaining)))
    if negative_count:
        rng = np.random.default_rng(stable_seed(key, salt=NEGATIVE_SEED_SALT))
        negative_indices = rng.choice(remaining, size=negative_count, replace=False)
        negative_indices.sort()
    else:
        negative_indices = np.empty(0, dtype=np.int64)
    return NegativePointSelection(
        negative_point_indices=negative_indices.astype(np.int64, copy=False),
        positive_point_count=int(np.count_nonzero(positive_rows)),
        remaining_point_count=len(remaining),
    )
