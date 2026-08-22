"""Canonical unified-v4 depth, frame, sampling, and sparse-label adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

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


DEFAULT_ROOT = Path("/home/irsl/datasets/dlr/compiled/scdm_justin_right_vgn_train_10000_reconstruction_view_aligned_v4")
DEFAULT_DERIVED_ROOT = Path("/home/irsl/datasets/dlr/derived/hgc_derived")
DEFAULT_GRIPPER_CONFIG = Path("/home/irsl/datasets/dlr/grippers/justin_hand/justin_right_hand_simple.yaml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        derived_root: str | Path | None = DEFAULT_DERIVED_ROOT,
        gripper_config: str | Path = DEFAULT_GRIPPER_CONFIG,
        sample_ids: Iterable[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.point_count = int(point_count)
        self.negative_fraction = float(negative_fraction)
        self.derived_root = None if derived_root is None else Path(derived_root)
        if self.derived_root is not None:
            self._validate_derived_manifest()
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
        records = [
            record
            for record in _json_lines(self.root / "index" / "samples.jsonl")
            if scene_splits.get(record["scene_id"]) == split
        ]
        if sample_ids is not None:
            requested = list(sample_ids)
            by_id = {record["sample_id"]: record for record in records}
            missing = [sample_id for sample_id in requested if sample_id not in by_id]
            if missing:
                raise ValueError(f"canonical split {split!r} lacks requested sample IDs: {missing}")
            records = [by_id[sample_id] for sample_id in requested]
        self.records = records
        self.views = {record["view_id"]: record for record in _json_lines(self.root / "index" / "views.jsonl")}
        if not self.records:
            raise ValueError(f"canonical split {split!r} has no samples")

    def _validate_derived_manifest(self) -> None:
        assert self.derived_root is not None
        path = self.derived_root / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"HGC derived manifest not found: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "format": "hgc_negative_points_v1",
            "point_count": self.point_count,
            "positive_match_radius_m": 0.005,
            "positive_filter": "all_approach_points_no_thumb_visible_mask",
            "negative_fraction_of_remaining_points": self.negative_fraction,
            "completed_sample_count": 10000,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"{path}: {key}={manifest.get(key)!r}, expected {value!r}")
        source_root = Path(manifest.get("source_root", ""))
        if source_root.resolve() != self.root.resolve():
            raise ValueError(f"{path}: source_root does not match canonical dataset root")
        source_files = {
            "source_dataset_sha256": self.root / "dataset.yaml",
            "source_samples_sha256": self.root / "index" / "samples.jsonl",
            "source_splits_sha256": self.root / "index" / "splits.json",
        }
        for key, source_path in source_files.items():
            if manifest.get(key) != _sha256(source_path):
                raise ValueError(f"{path}: {key} does not match {source_path}")

    def __len__(self) -> int:
        return len(self.records)

    def _positive_payload(self, record: dict[str, Any]) -> dict[str, np.ndarray]:
        with np.load(self.root / record["grasp_path"], allow_pickle=False) as payload:
            required = ("palm_pose9d", "approach_point", "q_contact", "q_squeeze", "grasp_type_idx", "source_grasp_index")
            missing = [key for key in required if key not in payload]
            if missing:
                raise KeyError(f"{record['grasp_path']} missing {missing}")
            return {
                "palm_pose9d": np.asarray(payload["palm_pose9d"], dtype=np.float32),
                "approach_point": np.asarray(payload["approach_point"], dtype=np.float32),
                "q_contact": np.asarray(payload["q_contact"], dtype=np.float32),
                "q_squeeze": np.asarray(payload["q_squeeze"], dtype=np.float32),
                "template_index": np.asarray(payload["grasp_type_idx"], dtype=np.int64),
                "source_grasp_index": np.asarray(payload["source_grasp_index"], dtype=np.int64),
            }

    def _negative_point_indices(
        self,
        record: dict[str, Any],
        *,
        points: np.ndarray,
        sampled_crop_indices: np.ndarray,
    ) -> np.ndarray | None:
        if self.derived_root is None:
            return None
        path = self.derived_root / "neg_points" / f"{record['sample_id']}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"HGC negative-point sidecar not found: {path}")
        with np.load(path, allow_pickle=False) as payload:
            for key in ("sample_id", "scene_id", "view_id", "negative_point_indices", "sampled_crop_indices"):
                if key not in payload:
                    raise KeyError(f"{path} missing {key}")
            identities = {
                "sample_id": record["sample_id"],
                "scene_id": record["scene_id"],
                "view_id": record["view_id"],
            }
            for key, expected in identities.items():
                actual = str(np.asarray(payload[key]).item())
                if actual != expected:
                    raise ValueError(f"{path}: {key}={actual!r}, expected {expected!r}")
            cached_sampling = np.asarray(payload["sampled_crop_indices"], dtype=np.int64)
            if not np.array_equal(cached_sampling, sampled_crop_indices):
                raise ValueError(f"{path}: deterministic point sampling does not match the loader")
            negative_indices = np.asarray(payload["negative_point_indices"], dtype=np.int64)
            if negative_indices.ndim != 1 or np.any(
                (negative_indices < 0) | (negative_indices >= len(points))
            ):
                raise ValueError(f"{path}: negative_point_indices is not a valid point-row vector")
            if "negative_points" in payload:
                cached_points = np.asarray(payload["negative_points"], dtype=np.float32)
                if not np.array_equal(cached_points, points[negative_indices]):
                    raise ValueError(f"{path}: cached negative coordinates do not match their indices")
        return negative_indices

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
        negative_point_indices = self._negative_point_indices(
            record, points=point, sampled_crop_indices=point_indices
        )
        labels = make_sparse_template_labels(
            points=point,
            **self._positive_payload(record),
            num_templates=len(TEMPLATE_NAMES),
            negative_fraction=self.negative_fraction,
            key=record["sample_id"],
            negative_point_indices=negative_point_indices,
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
            "canonical_positive_count": np.asarray(labels.canonical_positive_count, dtype=np.int64),
            "matched_positive_count": np.asarray(labels.matched_positive_count, dtype=np.int64),
            "sample_id": record["sample_id"],
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.load_numpy(index)
        # ``sample_id`` is deliberately omitted from tensor batches; a smoke
        # caller can use load_numpy() when it needs the audit string.
        return {key: torch.from_numpy(value) for key, value in item.items() if key != "sample_id"}
