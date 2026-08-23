"""CPU contract tests for the paper-modified Justin HGC arm."""

from __future__ import annotations

import importlib.util
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from paper_modified_hgc.data import PaperModifiedCanonicalDataset, _grid_point_to_index
from paper_modified_hgc.encoder import ConvBlock3D, ThreeDFPNEncoder
from paper_modified_hgc.model import PaperModifiedHGC, PaperModifiedHGCHead
from paper_modified_hgc.pose import (
    ORIENTATION_BINS,
    bin_and_residual_to_mat,
    mat_to_bin_and_residual,
    pose9d_to_targets,
    targets_to_pose9d,
)
from paper_modified_hgc.runtime import decode_paper_candidates


TRAIN_PATH = Path(__file__).resolve().parents[1] / "tools" / "train_justin_hgc.py"
SPEC = importlib.util.spec_from_file_location("issue59_train_paper_test", TRAIN_PATH)
assert SPEC and SPEC.loader
train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train)


def _fixture_root(root: Path, *, canonical_count: int = 2) -> tuple[Path, Path]:
    (root / "index").mkdir(parents=True)
    (root / "reconstruction_full_seg_grid").mkdir()
    (root / "grasp").mkdir()
    (root / "view_meta").mkdir()
    (root / "dataset.yaml").write_text(
        """format: scdm_unified
format_version: 2
grid:
  shape: [64, 64, 64]
  axis_order: [z, y, x]
  edge_length: 0.5
hand:
  name: justin_right_hand_simple
  dof: 12
""",
        encoding="utf-8",
    )
    record = {
        "sample_id": "packed_000000_view000",
        "scene_id": "packed_000000",
        "view_id": "packed_000000_view000",
        "reconstruction_full_seg_grid_path": "reconstruction_full_seg_grid/sample.npz",
        "grasp_path": "grasp/sample.npz",
    }
    (root / "index" / "samples.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (root / "index" / "splits.json").write_text(json.dumps({"packed_000000": "train"}), encoding="utf-8")
    (root / "index" / "views.jsonl").write_text(
        json.dumps(
            {
                "view_id": "packed_000000_view000",
                "scene_id": "packed_000000",
                "reconstruction_full_seg_grid_path": "reconstruction_full_seg_grid/sample.npz",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    if canonical_count < 2 or canonical_count > 4:
        raise ValueError("fixture canonical_count must be in [2, 4]")
    approach = np.asarray(
        [[0.02 * row, 0.01 * row, 0.0] for row in range(canonical_count)],
        dtype=np.float32,
    )
    grid = np.zeros((64, 64, 64), dtype=np.uint8)
    grid[2, 3, 4] = 1
    grid[10, 11, 12] = 2
    all_positive_indices = _grid_point_to_index(approach)
    grid[all_positive_indices[:, 0], all_positive_indices[:, 1], all_positive_indices[:, 2]] = 2
    np.savez_compressed(root / "reconstruction_full_seg_grid" / "sample.npz", full_seg_grid=grid)
    pose = np.asarray(
        [
            [0.02 * row, 0.01 * row, 0.10 + 0.01 * row, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0]
            for row in range(canonical_count)
        ],
        dtype=np.float32,
    )
    q_contact = np.zeros((canonical_count, 12), dtype=np.float32)
    q_squeeze = np.ones((canonical_count, 12), dtype=np.float32)
    template = np.asarray([0, 2, 1, 0][:canonical_count], dtype=np.int64)
    np.savez_compressed(
        root / "grasp" / "sample.npz",
        palm_pose9d=pose,
        approach_point=approach,
        q_contact=q_contact,
        q_squeeze=q_squeeze,
        grasp_type_idx=template,
    )
    gripper = root / "gripper.yaml"
    gripper.write_text(
        """grasp_templates:
  finger2:
    q_open: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  finger3:
    q_open: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  finger4:
    q_open: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
""",
        encoding="utf-8",
    )
    return root, gripper


class PaperModifiedDataTest(unittest.TestCase):
    def test_all_fallback_sample_keeps_negative_quality_supervision(self) -> None:
        dataset = PaperModifiedCanonicalDataset(
            split="train", sample_ids=["dense_001811_view000"]
        )
        sample = dataset[0]
        self.assertEqual(int(sample["positive_feature_count"]), 0)
        self.assertEqual(int(sample["negative_feature_count"]), 100)
        self.assertEqual(float(sample["quality_target"].sum()), 0.0)

        count = len(sample["feature_indices"])
        batch = {key: value.unsqueeze(0) for key, value in sample.items()}
        prediction = (
            torch.zeros_like(batch["quality_target"], requires_grad=True),
            torch.zeros((1, count, ORIENTATION_BINS), requires_grad=True),
            torch.zeros((1, count, 4), requires_grad=True),
            torch.zeros((1, count, 12), requires_grad=True),
        )
        losses = PaperModifiedHGCHead(feature_dim=4).loss(prediction, batch)
        self.assertEqual(float(losses["cls_loss"].detach()), 0.0)
        self.assertEqual(float(losses["reg_loss"].detach()), 0.0)
        self.assertEqual(float(losses["contact_loss"].detach()), 0.0)
        self.assertTrue(torch.isfinite(losses["total_loss"]))
        losses["total_loss"].backward()
        self.assertIsNotNone(prediction[0].grad)

    def test_v4_full_occupancy_and_ground_are_two_dhw_channels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, gripper = _fixture_root(Path(temporary))
            dataset = PaperModifiedCanonicalDataset(
                root, split="train", num_grasps=2, gripper_config=gripper
            )
            item = dataset[0]
            self.assertEqual(tuple(item["input_grid"].shape), (2, 64, 64, 64))
            self.assertEqual(item["input_grid"].dtype, torch.float32)
            self.assertTrue(item["input_grid"][0, 10, 11, 12])
            self.assertTrue(item["input_grid"][1, 2, 3, 4])
            self.assertFalse(item["input_grid"][0, 2, 3, 4])
            self.assertGreater(int(item["negative_feature_count"]), 0)
            self.assertEqual(tuple(item["feature_indices"].shape), (3, 3))
            self.assertEqual(tuple(item["template_graspable"].shape), (3, 3))
            self.assertEqual(int(item["positive_feature_count"]), 2)
            self.assertEqual(int(item["negative_feature_count"]), 1)
            self.assertEqual(int(item["tilted_excluded_count"]), 0)
            positive = item["positive_feature_indices"].tolist()
            negative = item["negative_feature_indices"].tolist()
            self.assertTrue(set(map(tuple, positive)).isdisjoint(map(tuple, negative)))
            self.assertTrue(torch.all(item["template_graspable"][2] == 0))
            self.assertEqual(int((item["template_graspable"] == 1).sum()), 2)
            self.assertEqual(tuple(item["template_q_open"].shape), (3, 3, 12))
            self.assertNotIn("template_q_squeeze", item)
            repeated = dataset[0]
            for key in ("feature_indices", "template_graspable", "template_pose"):
                self.assertTrue(torch.equal(item[key], repeated[key]))

    def test_negative_voxels_exclude_unselected_canonical_positive_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, gripper = _fixture_root(Path(temporary), canonical_count=4)
            dataset = PaperModifiedCanonicalDataset(
                root, split="train", num_grasps=2, gripper_config=gripper
            )
            item = dataset[0]
            self.assertEqual(int(item["positive_feature_count"]), 2)
            self.assertEqual(int(item["all_positive_feature_count"]), 4)
            self.assertGreater(int(item["negative_feature_count"]), 0)
            all_positive = set(map(tuple, item["all_positive_feature_indices"].tolist()))
            selected_positive = set(map(tuple, item["positive_feature_indices"].tolist()))
            negative = set(map(tuple, item["negative_feature_indices"].tolist()))
            self.assertTrue(selected_positive.issubset(all_positive))
            self.assertTrue(negative.isdisjoint(all_positive))
            self.assertTrue(torch.all(item["template_graspable"][2:, :] == 0))

    def test_tilted_approach_ray_is_excluded_from_scalar_depth_supervision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, gripper = _fixture_root(Path(temporary), canonical_count=4)
            grasp_path = root / "grasp" / "sample.npz"
            with np.load(grasp_path, allow_pickle=False) as payload:
                arrays = {key: payload[key] for key in payload.files}
            arrays["approach_point"] = arrays["approach_point"].copy()
            # With a 0.1m palm-to-anchor depth this is the canonical 10-degree
            # fallback ray (cosine ~= 0.984808), not a direct palm-z target.
            arrays["approach_point"][0] += np.asarray(
                (0.1 * np.tan(np.deg2rad(10.0)), 0.0, 0.0), dtype=np.float32
            )
            # The source archive may still carry a legacy q_squeeze array;
            # paper-modified loading must ignore it completely.
            arrays["q_squeeze"] = np.asarray([123.0], dtype=np.float32)
            np.savez_compressed(grasp_path, **arrays)
            dataset = PaperModifiedCanonicalDataset(
                root, split="train", num_grasps=3, gripper_config=gripper
            )
            item = dataset[0]
            self.assertEqual(int(item["canonical_positive_count"]), 3)
            self.assertEqual(int(item["all_positive_feature_count"]), 3)
            self.assertEqual(int(item["tilted_excluded_count"]), 1)
            expected_indices = set(
                map(tuple, _grid_point_to_index(arrays["approach_point"][1:]).tolist())
            )
            self.assertEqual(
                set(map(tuple, item["all_positive_feature_indices"].tolist())),
                expected_indices,
            )


class PaperModifiedModelTest(unittest.TestCase):
    def test_tiny_fpn_forward_loss_backward_and_strict_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, gripper = _fixture_root(Path(temporary))
            dataset = PaperModifiedCanonicalDataset(
                root, split="train", num_grasps=2, gripper_config=gripper
            )
            batch = {key: value.unsqueeze(0) for key, value in dataset[0].items()}
            model = PaperModifiedHGC(
                encoder_channels=(4, 4, 4, 4),
                encoder_scales=(2, 2, 2, 4),
                encoder_out_channels=8,
            )
            prediction = model.forward_batch(batch)
            losses = model.head.loss(prediction, batch)
            self.assertTrue(torch.isfinite(losses["total_loss"]))
            counts = train._dense_graspability_counts(
                torch.zeros_like(prediction.quality_logits), batch["quality_target"]
            )
            self.assertGreater(counts[1], 0)  # zero logits predict quality-positive everywhere
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                checkpoint = Path(checkpoint_dir) / "last.pt"
                train.save_checkpoint(
                    checkpoint,
                    model=model,
                    optimizer=optimizer,
                    epoch=0,
                    global_step=1,
                    config={"arm": "paper_modified"},
                    metrics={"val/total_loss": float(losses["total_loss"].detach())},
                )
                restored = PaperModifiedHGC(
                    encoder_channels=(4, 4, 4, 4),
                    encoder_scales=(2, 2, 2, 4),
                    encoder_out_channels=8,
                )
                restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4, weight_decay=1e-6)
                state = train.load_checkpoint(
                    checkpoint,
                    model=restored,
                    optimizer=restored_optimizer,
                    device=torch.device("cpu"),
                )
                self.assertEqual(state["epoch"], 0)
                self.assertEqual(state["global_step"], 1)
                for expected, actual in zip(model.parameters(), restored.parameters()):
                    self.assertTrue(torch.equal(expected, actual))


class SharedTrainerDispatchTest(unittest.TestCase):
    def test_paper_config_selects_adamw_and_fpn_without_changing_upstream_defaults(self) -> None:
        paper_config = train.load_training_config(
            Path(__file__).resolve().parents[1] / "config" / "issue59_paper_modified.yaml"
        )
        self.assertEqual(paper_config["arm"], "paper_modified")
        self.assertEqual(paper_config["optimizer"], "AdamW")
        self.assertEqual(paper_config["weight_decay"], 1e-6)
        self.assertEqual(train.load_training_config()["arm"], "upstream_faithful")
        self.assertEqual(train.load_training_config()["optimizer"], "Adam")

    def test_paper_factory_plumbs_declared_grid_contract(self) -> None:
        config = train.load_training_config(
            Path(__file__).resolve().parents[1] / "config" / "issue59_paper_modified.yaml"
        )
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        class FakeDataset:
            def __init__(self, *args: object, **kwargs: object) -> None:
                calls.append((args, kwargs))

            def __len__(self) -> int:
                return 1

        with patch.object(train, "PaperModifiedCanonicalDataset", FakeDataset):
            train.build_datasets(
                config,
                root=Path("/tmp/v4"),
                derived_root=None,
                train_sample_ids=["sample"],
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1]["grid_size"], 64)
        self.assertEqual(calls[0][1]["grid_edge_m"], 0.5)

    def test_paper_factory_rejects_representation_encoder_and_split_seed_drift(self) -> None:
        config = train.load_training_config(
            Path(__file__).resolve().parents[1] / "config" / "issue59_paper_modified.yaml"
        )
        bad_representation = {**config, "data": {**config["data"], "representation": "partial"}}
        with self.assertRaisesRegex(ValueError, "representation"):
            train.build_datasets(bad_representation, root=Path("/tmp/v4"), derived_root=None)

        bad_encoder = {**config, "model": {**config["model"], "encoder": "unet"}}
        with self.assertRaisesRegex(ValueError, "encoder"):
            train.build_model(bad_encoder)

        bad_split = {**config, "data": {**config["data"], "split_seed": 1}}
        with self.assertRaisesRegex(ValueError, "split_seed"):
            train.build_model(bad_split)

    def test_paper_model_interface_has_no_unused_source_alias_parameters(self) -> None:
        parameters = inspect.signature(PaperModifiedHGC).parameters
        for name in ("channels", "scales", "out_channels"):
            self.assertNotIn(name, parameters)

    def test_provenance_keeps_missing_derived_root_as_json_null(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            train.write_provenance(
                run_dir,
                config_path=Path(__file__).resolve().parents[1] / "config" / "issue59_paper_modified.yaml",
                config={"arm": "paper_modified"},
                dataset_root=Path("/tmp/v4"),
                derived_root=None,
                device=torch.device("cpu"),
                command=["test"],
                train_count=1,
                val_count=1,
            )
            payload = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
            self.assertIsNone(payload["derived_root"])
            self.assertEqual(
                payload["paper_positive_geometry_filter"]["approach_axis_cosine_min"],
                0.9999,
            )


class ThreeDFPNArchitectureTest(unittest.TestCase):
    def test_reference_groupnorm_counts_are_preserved_for_production_channels(self) -> None:
        encoder = ThreeDFPNEncoder(channels=(32, 64, 128, 256), out_channels=128)
        self.assertEqual(encoder.down1.norm1.num_groups, 4)
        self.assertEqual(encoder.down2.norm1.num_groups, 8)
        self.assertEqual(encoder.down3.norm1.num_groups, 16)
        self.assertEqual(encoder.down4.norm1.num_groups, 32)

    def test_tiny_channels_and_align_corners_false_use_reference_pooling_path(self) -> None:
        block = ConvBlock3D(2, 4, scale=2, align_corners=False)
        self.assertEqual(block.norm1.num_groups, 1)
        self.assertTrue(block.use_pooling)
        self.assertEqual(block.down.stride, (1, 1, 1))
        output = block(torch.zeros((1, 2, 16, 16, 16)))
        self.assertEqual(tuple(output.shape[-3:]), (8, 8, 8))


class PaperModifiedHeadSemanticsTest(unittest.TestCase):
    def test_reference_head_shapes_and_q_contact_only(self) -> None:
        model = PaperModifiedHGC(
            encoder_channels=(4, 4, 4, 4),
            encoder_scales=(2, 2, 2, 2),
            encoder_out_channels=8,
        )
        self.assertEqual(model.head.grasp_bin_pred_head.layers[-1].out_features, ORIENTATION_BINS)
        self.assertEqual(model.head.grasp_reg_head.layers[-1].out_features, 4)
        self.assertEqual(model.head.q_contact_head.layers[-1].out_features, 12)
        self.assertFalse(hasattr(model.head, "q_squeeze_head"))
        self.assertFalse(hasattr(model.head, "q_finger_head"))
        input_grid = torch.zeros((1, 2, 8, 8, 8))
        feature_indices = torch.tensor([[[3, 3, 3], [5, 5, 5]]])
        prediction = model.forward_batch(
            {"input_grid": input_grid, "feature_indices": feature_indices}
        )
        self.assertEqual(tuple(prediction.quality_logits.shape), (1, 1, 8, 8, 8))
        self.assertEqual(tuple(prediction.orientation_logits.shape), (1, 2, ORIENTATION_BINS))
        self.assertEqual(tuple(prediction.residual.shape), (1, 2, 4))
        self.assertEqual(tuple(prediction.q_contact.shape), (1, 2, 12))

    def test_dense_focal_loss_changes_when_positive_voxel_logit_changes(self) -> None:
        model = PaperModifiedHGC(
            encoder_channels=(2, 2, 2, 2),
            encoder_scales=(2, 2, 2, 2),
            encoder_out_channels=4,
        )
        batch = {
            "quality_target": torch.zeros((1, 1, 4, 4, 4)),
            "positive_mask": torch.tensor([[True]]),
            "orientation_bin": torch.tensor([[0]]),
            "pose_target": torch.zeros((1, 1, 4)),
            "q_contact": torch.zeros((1, 1, 12)),
        }
        batch["quality_target"][0, 0, 1, 1, 1] = 1.0
        quality_low = torch.zeros((1, 1, 4, 4, 4), requires_grad=True)
        quality_high = quality_low.detach().clone()
        quality_high[0, 0, 1, 1, 1] = 4.0
        orientation = torch.zeros((1, 1, ORIENTATION_BINS))
        residual = torch.zeros((1, 1, 4))
        contact = torch.zeros((1, 1, 12))
        low = model.head.loss((quality_low, orientation, residual, contact), batch)
        high = model.head.loss((quality_high, orientation, residual, contact), batch)
        self.assertLess(high["seg_loss"], low["seg_loss"])
        self.assertTrue(torch.isfinite(high["total_loss"]))

    def test_pose_bin_residual_roundtrip_for_valid_anchor_geometry(self) -> None:
        # The paper codec assumes the approach point lies on the palm local-z
        # ray, as in the canonical Justin target construction.
        anchor = torch.tensor([[0.1, -0.05, 0.0]], dtype=torch.float32)
        source_bin = torch.tensor([400], dtype=torch.long)
        source_residual = torch.tensor([[0.25, 0.1, -0.08, 0.07]], dtype=torch.float32)
        palm = targets_to_pose9d(source_bin, source_residual, anchor)
        bins, residual = pose9d_to_targets(palm, anchor)
        restored = targets_to_pose9d(bins, residual, anchor)
        torch.testing.assert_close(restored, palm, atol=2e-5, rtol=0.0)
        rotation = bin_and_residual_to_mat(bins, residual[:, 1:])
        decoded_bins, decoded_residual = mat_to_bin_and_residual(rotation)
        torch.testing.assert_close(decoded_bins, bins)
        torch.testing.assert_close(decoded_residual, residual[:, 1:], atol=2e-5, rtol=0.0)

    def test_runtime_returns_contact_only_without_waypoint_mapper(self) -> None:
        count = 100
        points = torch.zeros((count, 3), dtype=torch.float32)
        points[:, 0] = torch.arange(count) * 0.001
        indices = torch.zeros((count, 3), dtype=torch.long)
        indices[:, 2] = torch.arange(count) % 64
        quality = torch.zeros((1, 1, 64, 64, 64))
        quality[0, 0, indices[:, 0], indices[:, 1], indices[:, 2]] = torch.arange(count, dtype=quality.dtype)
        orientation = torch.zeros((1, count, ORIENTATION_BINS))
        residual = torch.zeros((1, count, 4))
        residual[..., 0] = 0.05
        contact = torch.zeros((1, count, 12))
        candidates, diagnostics = decode_paper_candidates(
            points=points,
            feature_indices=indices,
            quality_logits=quality,
            orientation_logits=orientation,
            residual=residual,
            q_contact=contact,
            topk=count,
            num_samples=count,
            minimum_candidates=count,
        )
        self.assertEqual(candidates["q_contact"].shape, (count, 12))
        self.assertNotIn("q_open", candidates)
        self.assertNotIn("q_squeeze", candidates)
        self.assertTrue(np.all(candidates["quality"][:-1] >= candidates["quality"][1:]))
        self.assertEqual(diagnostics["post_backfill"], count)


if __name__ == "__main__":
    unittest.main()
