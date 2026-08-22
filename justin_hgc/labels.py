"""Sparse point supervision for canonical unified-v4 views."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import pose9d_to_bin_target, stable_seed


@dataclass(frozen=True)
class SparseTemplateLabels:
    """Per-point labels using upstream's {-1 ignore, 0 negative, 1 positive}."""

    graspable: np.ndarray  # (N, templates), int64
    pose: np.ndarray  # (N, templates, 4), float32
    q_contact: np.ndarray  # (N, templates, 12), float32
    q_squeeze: np.ndarray  # (N, templates, 12), float32
    canonical_positive_count: int
    matched_positive_count: int  # canonical approach anchors with a 5 mm surface match


def make_sparse_template_labels(
    *,
    points: np.ndarray,
    palm_pose9d: np.ndarray,
    approach_point: np.ndarray,
    q_contact: np.ndarray,
    q_squeeze: np.ndarray,
    template_index: np.ndarray,
    source_grasp_index: np.ndarray,
    num_templates: int,
    negative_fraction: float,
    key: str,
    radius_m: float = 0.005,
    negative_point_indices: np.ndarray | None = None,
) -> SparseTemplateLabels:
    """Attach geometric positives and either cached or generated negatives.

    Unified-v4 stores ``negative_palm_pose9d`` and ``negative_q_contact`` but no
    negative approach point.  They cannot truthfully be attached to a visible
    surface point, so they are intentionally *not* converted into point labels.
    The remaining visible points receive the upstream-style sparse 10% negatives.
    """
    points = np.asarray(points, dtype=np.float32)
    palm_pose9d = np.asarray(palm_pose9d, dtype=np.float32)
    approach_point = np.asarray(approach_point, dtype=np.float32)
    q_contact = np.asarray(q_contact, dtype=np.float32)
    q_squeeze = np.asarray(q_squeeze, dtype=np.float32)
    template_index = np.asarray(template_index, dtype=np.int64)
    source_grasp_index = np.asarray(source_grasp_index, dtype=np.int64)
    n = len(points)
    if points.shape != (n, 3):
        raise ValueError(f"points must be Nx3, got {points.shape}")
    if not (len(palm_pose9d) == len(approach_point) == len(q_contact) == len(q_squeeze) == len(template_index) == len(source_grasp_index)):
        raise ValueError("positive canonical grasp arrays must have the same length")
    if q_contact.shape[-1:] != (12,) or q_squeeze.shape != q_contact.shape:
        raise ValueError("Justin joint targets must be matching Nx12 arrays")
    if np.any((template_index < 0) | (template_index >= num_templates)):
        raise ValueError("grasp_type_idx is outside the Justin template set")
    if not 0.0 <= negative_fraction <= 1.0:
        raise ValueError("negative_fraction must be in [0, 1]")

    labels = np.full((n, num_templates), -1, dtype=np.int64)
    pose = np.zeros((n, num_templates, 4), dtype=np.float32)
    contact = np.zeros((n, num_templates, 12), dtype=np.float32)
    squeeze = np.zeros((n, num_templates, 12), dtype=np.float32)
    matched_grasps = np.zeros(len(approach_point), dtype=bool)
    if len(approach_point):
        pose_target, _ = pose9d_to_bin_target(palm_pose9d, approach_point)
        # Do not materialize an N-by-number-of-grasps distance matrix.  A full
        # scene can carry many positives; this bounded per-grasp pass keeps only
        # one N-vector for the winning anchor of each Justin template.
        best_distance = np.full((n, num_templates), np.inf, dtype=np.float32)
        best_source = np.full((n, num_templates), np.iinfo(np.int64).max, dtype=np.int64)
        for grasp_idx in range(len(approach_point)):
            template = int(template_index[grasp_idx])
            distance = np.linalg.norm(points - approach_point[grasp_idx], axis=1)
            nearby = distance <= radius_m
            matched_grasps[grasp_idx] = bool(np.any(nearby))
            better = nearby & (
                (distance < best_distance[:, template])
                | ((distance == best_distance[:, template]) & (source_grasp_index[grasp_idx] < best_source[:, template]))
            )
            if not np.any(better):
                continue
            labels[better, template] = 1
            pose[better, template] = pose_target[grasp_idx]
            contact[better, template] = q_contact[grasp_idx]
            squeeze[better, template] = q_squeeze[grasp_idx]
            best_distance[better, template] = distance[better]
            best_source[better, template] = source_grasp_index[grasp_idx]

    positive_rows = np.any(labels == 1, axis=1)
    remaining = np.flatnonzero(~positive_rows)
    if negative_point_indices is not None:
        negative_rows = np.asarray(negative_point_indices, dtype=np.int64)
        if negative_rows.ndim != 1:
            raise ValueError("negative_point_indices must be one-dimensional")
        if np.any((negative_rows < 0) | (negative_rows >= n)):
            raise ValueError("negative_point_indices contains an out-of-range row")
        if len(np.unique(negative_rows)) != len(negative_rows):
            raise ValueError("negative_point_indices contains duplicates")
        if np.any(positive_rows[negative_rows]):
            raise ValueError("cached negative point overlaps a geometric positive")
        labels[negative_rows, :] = 0
    else:
        negative_count = int(np.floor(negative_fraction * len(remaining)))
        if not negative_count:
            negative_rows = np.empty(0, dtype=np.int64)
        else:
            rng = np.random.default_rng(stable_seed(key, salt="issue59-negatives"))
            negative_rows = rng.choice(remaining, size=negative_count, replace=False)
        labels[negative_rows, :] = 0
    matched_positive_count = int(np.count_nonzero(matched_grasps))
    return SparseTemplateLabels(
        graspable=labels,
        pose=pose,
        q_contact=contact,
        q_squeeze=squeeze,
        canonical_positive_count=len(approach_point),
        matched_positive_count=matched_positive_count,
    )
