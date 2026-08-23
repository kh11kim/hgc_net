"""Unified-v4 full-occupancy voxel adapter for the paper-modified HGC arm.

The canonical files remain immutable.  This adapter reads the per-view
``reconstruction_full_seg_grid`` and projects the same Justin grasp targets
used by the point arm onto voxel locations.  The input representation is
explicitly ``[object_full_occupancy, ground]`` in DHW/ZYX order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from justin_hgc.geometry import GRID_EDGE_M, stable_seed
from justin_hgc.templates import TEMPLATE_NAMES, load_template_q_open

from .pose import pose9d_to_targets, rot6d_to_mat


DEFAULT_ROOT = Path(
    "/home/irsl/datasets/dlr/compiled/"
    "scdm_justin_right_vgn_train_10000_reconstruction_view_aligned_v5"
)
DEFAULT_GRIPPER_CONFIG = Path(
    "/home/irsl/datasets/dlr/grippers/justin_hand/justin_right_hand_simple.yaml"
)
GRID_SIZE = 64
GRID_EDGE = GRID_EDGE_M


_BATCH_DENSE_KEYS = ("input_grid", "quality_target")
_BATCH_ROW_KEYS = (
    "feature_indices",
    "positive_mask",
    "orientation_bin",
    "pose_target",
    "q_contact",
)


def collate_paper_modified(
    items: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Batch trainer tensors while padding variable selected-feature rows."""

    if not items:
        raise ValueError("cannot collate an empty paper-modified batch")
    max_rows = max(int(item["feature_indices"].shape[0]) for item in items)
    output = {
        key: torch.stack([item[key] for item in items])
        for key in _BATCH_DENSE_KEYS
    }
    for key in _BATCH_ROW_KEYS:
        exemplar = items[0][key]
        padded = exemplar.new_zeros((len(items), max_rows, *exemplar.shape[1:]))
        if key == "orientation_bin":
            padded.fill_(-1)
        for batch_index, item in enumerate(items):
            rows = int(item[key].shape[0])
            padded[batch_index, :rows] = item[key]
        output[key] = padded
    output["tilted_excluded_count"] = torch.stack(
        [item["tilted_excluded_count"] for item in items]
    )
    return output


def _json_lines(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _grid_point_to_index(points: np.ndarray, *, size: int = GRID_SIZE, edge: float = GRID_EDGE) -> np.ndarray:
    """Map xyz grid-frame points to integer DHW/ZYX indices.

    The canonical reconstruction grid follows the reference Grid
    ``align_corners=True`` convention: voxel centers span
    ``[-(edge-edge/size)/2, +(edge-edge/size)/2]`` and output indices are
    ordered ``(z, y, x)``.
    """

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be Nx3, got {points.shape}")
    voxel = float(edge) / int(size)
    norm_scale = (float(edge) - voxel) / 2.0
    normalized = points / norm_scale
    xyz = np.rint((normalized + 1.0) * 0.5 * (int(size) - 1)).astype(np.int64)
    xyz = np.clip(xyz, 0, int(size) - 1)
    return xyz[:, [2, 1, 0]]


def _grid_index_to_point(indices: np.ndarray, *, size: int = GRID_SIZE, edge: float = GRID_EDGE) -> np.ndarray:
    """Map DHW/ZYX voxel indices to xyz grid-frame voxel-center points."""

    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError(f"indices must be Nx3, got {indices.shape}")
    voxel = float(edge) / int(size)
    norm_scale = (float(edge) - voxel) / 2.0
    zyx = indices.astype(np.float32, copy=False)
    xyz_index = zyx[:, [2, 1, 0]]
    normalized = (xyz_index / float(int(size) - 1)) * 2.0 - 1.0
    return (normalized * norm_scale).astype(np.float32, copy=False)


class PaperModifiedCanonicalDataset(Dataset[dict[str, torch.Tensor]]):
    """One GT full-occupancy/ground voxel scene and a fixed grasp budget.

    The reference arm uses a dense quality target over all 64^3 voxels and
    evaluates pose/contact targets only at the selected positive grasp rows.
    Deterministic object-occupancy rows are retained in the adapter for
    provenance and feature-shape checks, but are ignored by the paper pose and
    finger losses.  ``template_graspable`` is kept as a compatibility view of
    the canonical three grasp-type labels; the paper head itself has no
    template axis.  ``q_open``/``q_squeeze`` are execution waypoints derived
    by the grasp-sim client from learned ``q_contact``; neither is a
    paper-arm training target.
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        *,
        split: str = "train",
        num_grasps: int = 100,
        negative_voxel_count: int | None = None,
        negative_voxel_fraction: float | None = None,
        seed: int = 0,
        gripper_config: str | Path = DEFAULT_GRIPPER_CONFIG,
        sample_ids: Iterable[str] | None = None,
        grid_size: int = GRID_SIZE,
        grid_edge_m: float = GRID_EDGE,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train, val, or test; got {split!r}")
        if int(num_grasps) <= 0:
            raise ValueError("num_grasps must be positive")
        self.root = Path(root).resolve()
        self.split = split
        self.num_grasps = int(num_grasps)
        if negative_voxel_count is None and negative_voxel_fraction is None:
            negative_voxel_count = 100
        if negative_voxel_count is not None and int(negative_voxel_count) < 0:
            raise ValueError("negative_voxel_count must be non-negative or None")
        if negative_voxel_fraction is not None and not 0.0 <= float(negative_voxel_fraction) <= 1.0:
            raise ValueError("negative_voxel_fraction must be in [0, 1]")
        if negative_voxel_count is not None and negative_voxel_fraction is not None:
            raise ValueError("choose negative_voxel_count or negative_voxel_fraction, not both")
        self.negative_voxel_count = (
            None if negative_voxel_count is None else int(negative_voxel_count)
        )
        self.negative_voxel_fraction = (
            None if negative_voxel_fraction is None else float(negative_voxel_fraction)
        )
        self.seed = int(seed)
        self.grid_size = int(grid_size)
        self.grid_edge_m = float(grid_edge_m)
        if self.grid_size != GRID_SIZE:
            raise ValueError(f"paper-modified HGC requires a 64^3 grid, got {self.grid_size}")
        if not np.isclose(self.grid_edge_m, GRID_EDGE):
            raise ValueError(f"paper-modified HGC requires a 0.5m grid, got {self.grid_edge_m}")
        self._validate_manifest()
        self.template_q_open = np.asarray(load_template_q_open(gripper_config), dtype=np.float32)
        with (self.root / "index" / "splits.json").open(encoding="utf-8") as source:
            scene_splits = json.load(source)
        all_records = _json_lines(self.root / "index" / "samples.jsonl")
        records = [row for row in all_records if scene_splits.get(row["scene_id"]) == split]
        if sample_ids is not None:
            requested = list(sample_ids)
            by_id = {row["sample_id"]: row for row in records}
            missing = [sample_id for sample_id in requested if sample_id not in by_id]
            if missing:
                raise ValueError(f"canonical split {split!r} lacks requested sample IDs: {missing}")
            records = [by_id[sample_id] for sample_id in requested]
        if not records:
            raise ValueError(f"canonical split {split!r} has no samples")
        self.records = records
        self.views = {row["view_id"]: row for row in _json_lines(self.root / "index" / "views.jsonl")}

    def _validate_manifest(self) -> None:
        path = self.root / "dataset.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"canonical dataset manifest not found: {path}")
        with path.open(encoding="utf-8") as source:
            manifest = yaml.safe_load(source) or {}
        grid = manifest.get("grid", {})
        shape = tuple(grid.get("shape", ()))
        axis_order = tuple(grid.get("axis_order", ()))
        edge = grid.get("edge_length")
        if shape and shape != (GRID_SIZE, GRID_SIZE, GRID_SIZE):
            raise ValueError(f"{path}: expected 64^3 grid, got {shape}")
        if axis_order and axis_order != ("z", "y", "x"):
            raise ValueError(f"{path}: expected DHW/ZYX axis order, got {axis_order}")
        if edge is not None and not np.isclose(float(edge), GRID_EDGE):
            raise ValueError(f"{path}: expected 0.5m grid edge, got {edge}")
        hand = manifest.get("hand", {})
        if hand.get("dof", 12) != 12:
            raise ValueError(f"{path}: expected Justin 12-DoF labels")

    def __len__(self) -> int:
        return len(self.records)

    def _select_grasps(self, record: dict[str, Any]) -> dict[str, np.ndarray]:
        with np.load(self.root / record["grasp_path"], allow_pickle=False) as payload:
            required = (
                "palm_pose9d",
                "approach_point",
                "q_contact",
                "grasp_type_idx",
            )
            missing = [key for key in required if key not in payload]
            if missing:
                raise KeyError(f"{record['grasp_path']} missing {missing}")
            palm_pose = np.asarray(payload["palm_pose9d"], dtype=np.float32)
            approach = np.asarray(payload["approach_point"], dtype=np.float32)
            q_contact = np.asarray(payload["q_contact"], dtype=np.float32)
            template = np.asarray(payload["grasp_type_idx"], dtype=np.int64)
        if palm_pose.ndim != 2 or palm_pose.shape[1] != 9:
            raise ValueError(f"{record['sample_id']}: palm_pose9d must be Nx9")
        if approach.shape != (len(palm_pose), 3):
            raise ValueError(f"{record['sample_id']}: approach_point shape mismatch")
        if q_contact.shape != (len(palm_pose), 12):
            raise ValueError(f"{record['sample_id']}: Justin joint target shape mismatch")
        if template.shape != (len(palm_pose),):
            raise ValueError(f"{record['sample_id']}: grasp_type_idx shape mismatch")
        inside = np.all(np.abs(approach) <= self.grid_edge_m / 2.0 + 1e-6, axis=-1)
        if not np.any(inside):
            raise ValueError(f"{record['sample_id']}: no positive grasp approach point inside grid")
        palm_pose = palm_pose[inside]
        approach = approach[inside]
        q_contact = q_contact[inside]
        template = template[inside]
        # The paper codec has one scalar depth along palm local +z.  Canonical
        # DFC postprocess stores a 10-degree tilt-ray fallback for a minority
        # of grasps; those anchors cannot be reconstructed by this scalar
        # depth representation and must not enter supervision.
        rotation = rot6d_to_mat(torch.from_numpy(palm_pose[:, 3:])).numpy()
        approach_vector = approach - palm_pose[:, :3]
        approach_norm = np.linalg.norm(approach_vector, axis=1)
        palm_z = rotation[:, :, 2]
        axis_cosine = np.divide(
            np.sum(approach_vector * palm_z, axis=1),
            np.maximum(approach_norm, 1.0e-12),
        )
        direct_axis = (approach_norm > 1.0e-8) & (axis_cosine >= 0.9999)
        tilted_excluded_count = int((~direct_axis).sum())
        palm_pose = palm_pose[direct_axis]
        approach = approach[direct_axis]
        q_contact = q_contact[direct_axis]
        template = template[direct_axis]
        if np.any((template < 0) | (template >= len(TEMPLATE_NAMES))):
            raise ValueError(f"{record['sample_id']}: grasp_type_idx outside 0..2")
        rng = np.random.default_rng(stable_seed(record["sample_id"], salt=f"paper-modified:{self.seed}"))
        if len(palm_pose) == 0:
            selected = np.empty((0,), dtype=np.int64)
        elif len(palm_pose) >= self.num_grasps:
            selected = rng.choice(len(palm_pose), size=self.num_grasps, replace=False)
        else:
            selected = np.concatenate(
                (
                    rng.permutation(len(palm_pose)),
                    rng.choice(len(palm_pose), size=self.num_grasps - len(palm_pose), replace=True),
                )
            )
        return {
            "palm_pose9d": palm_pose[selected],
            "approach_point": approach[selected],
            "q_contact": q_contact[selected],
            "template": template[selected],
            # Keep the complete inside-grid canonical positive pool separate
            # from the selected training budget.  Every member of this pool
            # must be excluded from GT-voxel negative sampling.
            "all_inside_approach_point": approach,
            "all_inside_count": np.asarray(len(palm_pose), dtype=np.int64),
            "tilted_excluded_count": np.asarray(tilted_excluded_count, dtype=np.int64),
        }

    def load_numpy(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        view = self.views.get(record["view_id"], record)
        grid_path = view.get("reconstruction_full_seg_grid_path") or record.get(
            "reconstruction_full_seg_grid_path"
        )
        if not grid_path:
            raise KeyError(f"{record['sample_id']}: missing reconstruction_full_seg_grid_path")
        with np.load(self.root / grid_path, allow_pickle=False) as payload:
            if "full_seg_grid" not in payload:
                raise KeyError(f"{grid_path} missing full_seg_grid")
            labels = np.asarray(payload["full_seg_grid"])
        if labels.shape != (self.grid_size,) * 3 or labels.dtype != np.uint8:
            raise ValueError(f"{record['sample_id']}: expected uint8 64^3 full_seg_grid, got {labels.shape} {labels.dtype}")
        object_grid = labels >= 2
        ground_grid = labels == 1
        input_grid = np.stack((object_grid, ground_grid), axis=0).astype(np.float32, copy=False)
        selected = self._select_grasps(record)
        positive_approach = selected["approach_point"]
        positive_indices = _grid_point_to_index(
            positive_approach, size=self.grid_size, edge=self.grid_edge_m
        )
        all_positive_indices = np.unique(
            _grid_point_to_index(
                selected["all_inside_approach_point"],
                size=self.grid_size,
                edge=self.grid_edge_m,
            ),
            axis=0,
        )
        # Sample only occupied/object voxels.  The positive approach anchors
        # are removed by exact DHW index, so a negative feature can never share
        # a voxel with a positive feature.  Sampling is independent of grasp
        # order and therefore stable for the same sample_id and seed.
        negative_mask = object_grid.copy()
        if len(all_positive_indices):
            negative_mask[
                all_positive_indices[:, 0], all_positive_indices[:, 1], all_positive_indices[:, 2]
            ] = False
        negative_candidates = np.argwhere(negative_mask).astype(np.int64, copy=False)
        if self.negative_voxel_fraction is not None:
            requested_negative_count = int(
                np.ceil(self.negative_voxel_fraction * len(negative_candidates))
            )
            if self.negative_voxel_fraction > 0.0:
                requested_negative_count = max(1, requested_negative_count)
        else:
            requested_negative_count = int(self.negative_voxel_count or 0)
        if requested_negative_count > 0 and not len(negative_candidates):
            raise ValueError(
                f"{record['sample_id']}: no object voxel remains for negative supervision"
            )
        if requested_negative_count > len(negative_candidates):
            requested_negative_count = len(negative_candidates)
        negative_rng = np.random.default_rng(
            stable_seed(record["sample_id"], salt=f"paper-modified-negative:{self.seed}")
        )
        if requested_negative_count:
            negative_selection = negative_rng.choice(
                len(negative_candidates), size=requested_negative_count, replace=False
            )
            negative_indices = negative_candidates[negative_selection]
        else:
            negative_indices = np.empty((0, 3), dtype=np.int64)
        positive_count = len(positive_indices)
        negative_count = len(negative_indices)
        indices = np.concatenate((positive_indices, negative_indices), axis=0)
        feature_points = np.concatenate(
            (
                positive_approach.astype(np.float32, copy=False),
                _grid_index_to_point(
                    negative_indices, size=self.grid_size, edge=self.grid_edge_m
                ),
            ),
            axis=0,
        )
        template = selected["template"]
        count = positive_count + negative_count
        graspable = np.full((count, len(TEMPLATE_NAMES)), -1, dtype=np.int64)
        if positive_count:
            orientation_bin_t, paper_pose_target_t = pose9d_to_targets(
                torch.from_numpy(selected["palm_pose9d"]),
                torch.from_numpy(positive_approach),
            )
            orientation_bin = orientation_bin_t.numpy().astype(np.int64, copy=False)
            pose_target = paper_pose_target_t.numpy().astype(np.float32, copy=False)
        else:
            orientation_bin = np.empty((0,), dtype=np.int64)
            pose_target = np.empty((0, 4), dtype=np.float32)
        pose = np.zeros((count, len(TEMPLATE_NAMES), 4), dtype=np.float32)
        contact = np.zeros((count, len(TEMPLATE_NAMES), 12), dtype=np.float32)
        for row, grasp_template in enumerate(template):
            graspable[row, int(grasp_template)] = 1
            pose[row, int(grasp_template)] = pose_target[row]
            contact[row, int(grasp_template)] = selected["q_contact"][row]
        if negative_count:
            graspable[positive_count:, :] = 0
        q_open = np.broadcast_to(self.template_q_open[None, :, :], (count, len(TEMPLATE_NAMES), 12)).copy()
        palm_pose9d = np.concatenate(
            (
                selected["palm_pose9d"],
                np.zeros((negative_count, 9), dtype=np.float32),
            ),
            axis=0,
        )
        grasp_type_idx = np.concatenate(
            (
                template,
                np.full(negative_count, -1, dtype=np.int64),
            ),
            axis=0,
        )
        q_open_selected = np.concatenate(
            (
                self.template_q_open[template],
                np.zeros((negative_count, 12), dtype=np.float32),
            ),
            axis=0,
        )
        positive_mask = np.concatenate(
            (
                np.ones((positive_count,), dtype=np.bool_),
                np.zeros((negative_count,), dtype=np.bool_),
            ),
            axis=0,
        )
        orientation_bin_selected = np.concatenate(
            (
                orientation_bin,
                np.full((negative_count,), -1, dtype=np.int64),
            ),
            axis=0,
        )
        pose_target_selected = np.concatenate(
            (
                pose_target,
                np.zeros((negative_count, 4), dtype=np.float32),
            ),
            axis=0,
        )
        # The reference dense quality label marks the selected feasible scene
        # anchors.  Every unselected voxel remains the focal-loss negative;
        # the auxiliary negative feature rows above are not used to fabricate
        # an additional quality target.
        quality_target = np.zeros((1, self.grid_size, self.grid_size, self.grid_size), dtype=np.float32)
        if positive_count:
            quality_target[
                0,
                positive_indices[:, 0],
                positive_indices[:, 1],
                positive_indices[:, 2],
            ] = 1.0
        row_q_contact = np.concatenate(
            (
                selected["q_contact"].astype(np.float32, copy=False),
                np.zeros((negative_count, 12), dtype=np.float32),
            ),
            axis=0,
        )
        return {
            "input_grid": input_grid,
            # These explicit channel views retain the vocabulary used by the
            # canonical reconstruction loader while ``input_grid`` is the
            # trainer-facing two-channel tensor.
            "full_grid": object_grid[None, ...].astype(np.bool_, copy=False),
            "ground_grid": ground_grid[None, ...].astype(np.bool_, copy=False),
            "feature_indices": indices,
            "positive_feature_indices": positive_indices,
            "all_positive_feature_indices": all_positive_indices,
            "negative_feature_indices": negative_indices,
            "feature_points": feature_points,
            "positive_approach_point": positive_approach.astype(np.float32, copy=False),
            "grasp_point": feature_points,
            "template_graspable": graspable,
            "template_pose": pose,
            "template_q_contact": contact,
            "template_q_open": q_open,
            "positive_mask": positive_mask,
            "quality_target": quality_target,
            "orientation_bin": orientation_bin_selected,
            "pose_target": pose_target_selected,
            "q_contact": row_q_contact,
            "q_open": q_open_selected,
            "palm_pose9d": palm_pose9d,
            "grasp_type_idx": grasp_type_idx,
            "canonical_positive_count": selected["all_inside_count"],
            "tilted_excluded_count": selected["tilted_excluded_count"],
            "positive_feature_count": np.asarray(positive_count, dtype=np.int64),
            "all_positive_feature_count": np.asarray(len(all_positive_indices), dtype=np.int64),
            "negative_feature_count": np.asarray(negative_count, dtype=np.int64),
            "negative_candidate_count": np.asarray(len(negative_candidates), dtype=np.int64),
            "sample_id": record["sample_id"],
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.load_numpy(index)
        return {
            key: torch.from_numpy(value)
            for key, value in item.items()
            if key != "sample_id"
        }


__all__ = ["PaperModifiedCanonicalDataset", "collate_paper_modified"]
