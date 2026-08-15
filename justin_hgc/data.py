"""Canonical unified-v3 depth, frame, sampling, and sparse-label adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .geometry import (
    GRID_EDGE_M,
    POINT_COUNT,
    crop_centered_grid,
    deproject_depth_m,
    deterministic_fixed_sample,
    transform_points,
)
from .labels import make_sparse_template_labels
from .templates import TEMPLATE_NAMES, load_template_q_open


DEFAULT_ROOT = Path("/home/irsl/datasets/dlr/compiled/scdm_justin_right_vgn_train_10000_reconstruction_view_aligned_v3")
DEFAULT_GRIPPER_CONFIG = Path("/home/irsl/datasets/dlr/grippers/justin_hand/justin_right_hand_simple.yaml")


def _json_lines(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


class JustinCanonicalDataset(Dataset[dict[str, torch.Tensor]]):
    """One deterministic 25,000-point input per canonical scene/view.

    The dataset never converts canonical negative grasps into a fake surface
    anchor: those payloads lack ``negative_approach_point``.  It instead exposes
    the documented sparse 10% remaining-visible-points negative policy.
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        *,
        split: str = "train",
        point_count: int = POINT_COUNT,
        negative_fraction: float = 0.10,
        gripper_config: str | Path = DEFAULT_GRIPPER_CONFIG,
    ) -> None:
        self.root = Path(root)
        self.point_count = int(point_count)
        self.negative_fraction = float(negative_fraction)
        # This also verifies that stored grasp_type_idx 0/1/2 means exactly
        # finger2/finger3/finger4, in the same order as the KMK config.
        self.template_q_open = load_template_q_open(gripper_config)
        manifest = self.root / "dataset.yaml"
        if not manifest.is_file():
            raise FileNotFoundError(f"canonical dataset manifest not found: {manifest}")
        # The immutable contract pins this scalar; do not rely on an image default.
        self.depth_scale = 0.001
        with (self.root / "index" / "splits.json").open(encoding="utf-8") as source:
            scene_splits = json.load(source)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train, val, or test; got {split!r}")
        self.records = [
            record
            for record in _json_lines(self.root / "index" / "samples.jsonl")
            if scene_splits.get(record["scene_id"]) == split
        ]
        self.views = {record["view_id"]: record for record in _json_lines(self.root / "index" / "views.jsonl")}
        if not self.records:
            raise ValueError(f"canonical split {split!r} has no samples")

    def __len__(self) -> int:
        return len(self.records)

    def _positive_payload(self, record: dict[str, Any]) -> dict[str, np.ndarray]:
        with np.load(self.root / record["grasp_path"], allow_pickle=False) as payload:
            required = ("palm_pose9d", "approach_point", "q_contact", "q_squeeze", "grasp_type_idx", "source_grasp_index", "thumb_visible_mask")
            missing = [key for key in required if key not in payload]
            if missing:
                raise KeyError(f"{record['grasp_path']} missing {missing}")
            visible = np.asarray(payload["thumb_visible_mask"], dtype=bool)
            view_index = int(record["view_index"])
            if visible.ndim != 2 or view_index >= visible.shape[0]:
                raise ValueError(f"{record['sample_id']}: invalid thumb_visible_mask for view {view_index}")
            mask = visible[view_index]
            return {
                "palm_pose9d": np.asarray(payload["palm_pose9d"], dtype=np.float32)[mask],
                "approach_point": np.asarray(payload["approach_point"], dtype=np.float32)[mask],
                "q_contact": np.asarray(payload["q_contact"], dtype=np.float32)[mask],
                "q_squeeze": np.asarray(payload["q_squeeze"], dtype=np.float32)[mask],
                "template_index": np.asarray(payload["grasp_type_idx"], dtype=np.int64)[mask],
                "source_grasp_index": np.asarray(payload["source_grasp_index"], dtype=np.int64)[mask],
            }

    def load_numpy(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        view = self.views[record["view_id"]]
        with (self.root / view["view_meta_path"]).open(encoding="utf-8") as source:
            meta = json.load(source)
        if "T_reconstruction_grid_camera" not in meta:
            raise KeyError(f"{view['view_meta_path']} lacks T_reconstruction_grid_camera")
        depth = np.asarray(Image.open(self.root / view["depth_path"]))
        camera_points = deproject_depth_m(depth, meta["camera_intrinsics"], depth_scale=self.depth_scale)
        grid_points = transform_points(camera_points, np.asarray(meta["T_reconstruction_grid_camera"]))
        cropped, crop_mask = crop_centered_grid(grid_points, edge_length_m=GRID_EDGE_M)
        point, point_indices = deterministic_fixed_sample(cropped, count=self.point_count, key=record["sample_id"])
        labels = make_sparse_template_labels(
            points=point,
            **self._positive_payload(record),
            num_templates=len(TEMPLATE_NAMES),
            negative_fraction=self.negative_fraction,
            key=record["sample_id"],
        )
        return {
            "point": point,
            "norm_point": point - point.mean(axis=0, keepdims=True),
            "template_graspable": labels.graspable,
            "template_pose": labels.pose,
            "template_q_contact": labels.q_contact,
            "template_q_squeeze": labels.q_squeeze,
            "sample_indices": point_indices,
            "visible_crop_count": np.asarray(len(cropped), dtype=np.int64),
            "visible_positive_count": np.asarray(labels.visible_positive_count, dtype=np.int64),
            "matched_positive_count": np.asarray(labels.matched_positive_count, dtype=np.int64),
            "sample_id": record["sample_id"],
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.load_numpy(index)
        # ``sample_id`` is deliberately omitted from tensor batches; a smoke
        # caller can use load_numpy() when it needs the audit string.
        return {key: torch.from_numpy(value) for key, value in item.items() if key != "sample_id"}
