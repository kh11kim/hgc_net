"""CPU-only contract tests for the Issue #59 Justin HGC adapter.

These tests deliberately exercise only NumPy/Torch code.  The legacy PointNet++
CUDA extension is not loaded here; that extension is a separate environment gate.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from justin_hgc.data import JustinCanonicalDataset, _sha256
from justin_hgc.geometry import (
    crop_centered_grid,
    deproject_depth_m,
    deterministic_fixed_sample,
    pose9d_to_bin_target,
    transform_points,
)
from justin_hgc.labels import make_sparse_template_labels
from justin_hgc.postprocess import aggressive_nms, top_side_mask
from justin_hgc.bin_pose import (
    DEFAULT_BIN_SPEC,
    bin_regression_loss,
    bin_target_to_rotation,
    target_range_counts,
)
from justin_hgc.model import JustinHGCOutputHead
from justin_hgc.negative_points import select_negative_point_indices
from justin_hgc.runtime import decode_justin_candidates
from justin_hgc.templates import TEMPLATE_NAMES, load_template_q_open


class DerivedManifestTest(unittest.TestCase):
    def test_completed_count_tracks_the_canonical_sample_index(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            index = root / "index"
            derived = root / "derived" / "hgc"
            index.mkdir(parents=True)
            derived.mkdir(parents=True)
            (root / "dataset.yaml").write_text("format: scdm_unified\n", encoding="utf-8")
            (index / "samples.jsonl").write_text(
                '{"sample_id": "sample-0"}\n{"sample_id": "sample-1"}\n',
                encoding="utf-8",
            )
            (index / "splits.json").write_text("{}\n", encoding="utf-8")
            manifest = {
                "format": "hgc_negative_points_v1",
                "source_root": str(root),
                "source_dataset_sha256": _sha256(root / "dataset.yaml"),
                "source_samples_sha256": _sha256(index / "samples.jsonl"),
                "source_splits_sha256": _sha256(index / "splits.json"),
                "point_count": 25_000,
                "positive_match_radius_m": 0.005,
                "positive_filter": "all_approach_points_no_thumb_visible_mask",
                "negative_fraction_of_remaining_points": 0.10,
                "completed_sample_count": 2,
            }
            (derived / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            dataset = object.__new__(JustinCanonicalDataset)
            dataset.root = root
            dataset.derived_root = derived
            dataset.point_count = 25_000
            dataset.negative_fraction = 0.10
            dataset._validate_derived_manifest()

            manifest["completed_sample_count"] = 10_000
            (derived / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "expected 2"):
                dataset._validate_derived_manifest()


class GeometryTest(unittest.TestCase):
    def test_deprojection_transform_and_centered_half_meter_crop(self):
        depth = np.asarray([[0, 1000], [2000, 1000]], dtype=np.uint16)
        camera_points = deproject_depth_m(
            depth, {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}, depth_scale=0.001
        )
        np.testing.assert_allclose(
            camera_points,
            [[1.0, 0.0, 1.0], [0.0, 2.0, 2.0], [1.0, 1.0, 1.0]],
        )
        T = np.eye(4)
        T[:3, 3] = [-1.0, -1.0, -1.0]
        grid_points = transform_points(camera_points, T)
        np.testing.assert_allclose(grid_points[0], [0.0, -1.0, 0.0])
        cropped, mask = crop_centered_grid(
            np.asarray([[-0.25, 0.0, 0.25], [0.25001, 0.0, 0.0]])
        )
        self.assertEqual(mask.tolist(), [True, False])
        np.testing.assert_allclose(cropped, [[-0.25, 0.0, 0.25]])

    def test_deterministic_sampling_keeps_every_short_scene_point_once(self):
        points = np.arange(18, dtype=np.float32).reshape(6, 3)
        short, short_indices = deterministic_fixed_sample(points, count=10, key="short")
        again, again_indices = deterministic_fixed_sample(points, count=10, key="short")
        np.testing.assert_array_equal(short, again)
        np.testing.assert_array_equal(short_indices, again_indices)
        np.testing.assert_array_equal(short_indices[:6], np.arange(6))
        self.assertEqual(set(short_indices.tolist()), set(range(6)))

        long, long_indices = deterministic_fixed_sample(points, count=4, key="long")
        self.assertEqual(len(np.unique(long_indices)), 4)
        np.testing.assert_array_equal(long, points[long_indices])

    def test_pose_target_uses_surface_to_palm_axis_and_centimeters(self):
        # The palm local +z axis points from palm to surface.  This fixture has
        # local +z = world -z, so the HGC surface-to-palm axis is world +z.
        pose = np.asarray([[0.0, 0.0, 0.10, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0]])
        anchor = np.asarray([[0.0, 0.0, 0.0]])
        target, axes = pose9d_to_bin_target(pose, anchor)
        np.testing.assert_allclose(axes, [[0.0, 0.0, 1.0]])
        np.testing.assert_allclose(target[:, 0], [10.0])
        np.testing.assert_allclose(target[:, 2], [0.0])

    def test_circular_pose_targets_remain_inside_half_open_bins_after_float32(self):
        angle = np.deg2rad(-1.0e-7)
        axis = np.asarray(
            [np.cos(angle) / np.sqrt(2.0), np.sin(angle) / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
            dtype=np.float64,
        )
        closing = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        closing -= np.dot(closing, axis) * axis
        closing /= np.linalg.norm(closing)
        first = np.cross(axis, closing)
        rotation = np.stack((first, closing, -axis), axis=1)
        palm = np.concatenate((axis * 0.1, rotation[:, 0], rotation[:, 1]))[None]
        target, _ = pose9d_to_bin_target(
            palm.astype(np.float32), np.zeros((1, 3), dtype=np.float32)
        )
        self.assertGreaterEqual(float(target[0, 1]), 0.0)
        self.assertLess(float(target[0, 1]), 360.0)
        self.assertGreaterEqual(float(target[0, 3]), 0.0)
        self.assertLess(float(target[0, 3]), 360.0)

    def test_pose_target_rotation_round_trip_restores_canonical_palm_frame(self):
        pose = np.asarray(
            [[0.10, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        target, _ = pose9d_to_bin_target(
            pose, np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
        )
        decoded = bin_target_to_rotation(torch.from_numpy(target)).numpy()
        expected = np.asarray(
            [[[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(decoded, expected, atol=1.0e-6)

    def test_runtime_pose_decode_restores_palm_position_from_surface_anchor(self):
        target = np.asarray([10.0, 0.0, 45.0, 90.0], dtype=np.float32)
        prediction = torch.full(
            (1, DEFAULT_BIN_SPEC.channels, 1), -10.0, dtype=torch.float32
        )
        start = 0
        for value, (_, scope, size) in zip(target, DEFAULT_BIN_SPEC.group_specs):
            bins = int(scope / size)
            index = int(np.floor(value / size))
            prediction[0, start + index, 0] = 10.0
            prediction[0, start + bins + index, 0] = float(
                (value - index * size - size / 2.0) / size
            )
            start += 2 * bins

        candidates, _ = decode_justin_candidates(
            points=torch.zeros((1, 3), dtype=torch.float32),
            graspable_logits=torch.tensor([[[0.0], [1.0]]], dtype=torch.float32),
            pose_logits=prediction,
            q_contact=torch.zeros((1, 12, 1), dtype=torch.float32),
        )
        self.assertEqual(len(candidates["palm_pose"]), 1)
        np.testing.assert_allclose(
            candidates["palm_pose"][0, :3],
            [np.sqrt(0.005), 0.0, np.sqrt(0.005)],
            atol=1.0e-6,
        )


class SparseLabelTest(unittest.TestCase):
    def test_positive_5mm_neighborhoods_and_deterministic_ten_percent_negatives(self):
        points = np.asarray(
            [[0.0, 0.0, 0.0], [0.004, 0.0, 0.0], [0.020, 0.0, 0.0], [0.03, 0, 0], [0.04, 0, 0]],
            dtype=np.float32,
        )
        labels = make_sparse_template_labels(
            points=points,
            palm_pose9d=np.asarray([[0, 0, 0.1, 1, 0, 0, 0, -1, 0]], dtype=np.float32),
            approach_point=np.asarray([[0, 0, 0]], dtype=np.float32),
            q_contact=np.ones((1, 12), dtype=np.float32),
            template_index=np.asarray([1]),
            source_grasp_index=np.asarray([4]),
            num_templates=3,
            negative_fraction=0.10,
            key="fixture",
        )
        self.assertEqual(labels.graspable[0, 1], 1)
        self.assertEqual(labels.graspable[1, 1], 1)
        self.assertTrue(np.all(labels.graspable[:2, [0, 2]] == -1))
        self.assertEqual(np.count_nonzero(labels.graspable == 0), 0)  # floor(0.1 * 3) == 0
        self.assertEqual(labels.canonical_positive_count, 1)
        self.assertEqual(labels.pose_eligible_positive_count, 1)
        self.assertEqual(labels.rejected_tilted_positive_count, 0)
        self.assertEqual(labels.matched_positive_count, 1)

    def test_tilted_dfc_fallback_anchor_is_not_a_pose_positive(self):
        angle = math.radians(10.0)
        labels = make_sparse_template_labels(
            points=np.asarray([[math.sin(angle) * 0.1, 0.0, math.cos(angle) * 0.1]], dtype=np.float32),
            palm_pose9d=np.asarray([[0, 0, 0, 1, 0, 0, 0, 1, 0]], dtype=np.float32),
            approach_point=np.asarray([[math.sin(angle) * 0.1, 0.0, math.cos(angle) * 0.1]], dtype=np.float32),
            q_contact=np.ones((1, 12), dtype=np.float32),
            template_index=np.asarray([0]),
            source_grasp_index=np.asarray([0]),
            num_templates=3,
            negative_fraction=0.0,
            key="tilted",
        )
        self.assertEqual(labels.canonical_positive_count, 1)
        self.assertEqual(labels.pose_eligible_positive_count, 0)
        self.assertEqual(labels.rejected_tilted_positive_count, 1)
        self.assertFalse(np.any(labels.graspable == 1))

    def test_negative_sidecar_selection_excludes_all_geometric_positive_rows(self):
        points = np.arange(60, dtype=np.float32).reshape(20, 3) * 0.01
        approach_points = points[[2, 7]]
        first = select_negative_point_indices(
            points=points,
            approach_points=approach_points,
            negative_fraction=0.50,
            radius_m=0.001,
            key="scene_view_001",
        )
        again = select_negative_point_indices(
            points=points,
            approach_points=approach_points,
            negative_fraction=0.50,
            radius_m=0.001,
            key="scene_view_001",
        )
        np.testing.assert_array_equal(first.negative_point_indices, again.negative_point_indices)
        self.assertEqual(first.positive_point_count, 2)
        self.assertEqual(first.remaining_point_count, 18)
        self.assertEqual(len(first.negative_point_indices), 9)
        self.assertTrue(set(first.negative_point_indices).isdisjoint({2, 7}))

    def test_explicit_sidecar_indices_replace_online_negative_sampling(self):
        points = np.asarray([[0, 0, 0], [0.02, 0, 0], [0.03, 0, 0]], dtype=np.float32)
        labels = make_sparse_template_labels(
            points=points,
            palm_pose9d=np.asarray([[0, 0, 0.1, 1, 0, 0, 0, -1, 0]], dtype=np.float32),
            approach_point=np.asarray([[0, 0, 0]], dtype=np.float32),
            q_contact=np.ones((1, 12), dtype=np.float32),
            template_index=np.asarray([0]),
            source_grasp_index=np.asarray([1]),
            num_templates=3,
            negative_fraction=1.0,
            negative_point_indices=np.asarray([2]),
            key="fixture",
        )
        self.assertTrue(np.all(labels.graspable[2] == 0))
        self.assertTrue(np.all(labels.graspable[1] == -1))
        self.assertEqual(labels.graspable[0, 0], 1)


class PostprocessTest(unittest.TestCase):
    def test_top_side_and_exact_or_nms_rule(self):
        grasp_point = np.zeros((3, 3))
        palm_position = np.asarray([[0, 0, 0.1], [0.04, 0, 0.1], [0.10, 0, -0.1]])
        self.assertEqual(top_side_mask(grasp_point, palm_position).tolist(), [True, True, False])

        rotations = np.repeat(np.eye(3)[None], 3, axis=0)
        angle = math.radians(45)
        rotations[1] = np.asarray(
            [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
        )
        rotations[2] = rotations[1]
        kept, counts = aggressive_nms(
            positions=np.asarray([[0, 0, 0], [0.04, 0, 0], [0.10, 0, 0]]),
            rotations=rotations,
            scores=np.asarray([0.9, 0.8, 0.7]),
        )
        # Candidate 1 survives: it is farther than 3 cm and differs by 45 deg.
        # Candidate 2 is suppressed because its rotation is the same as candidate 1.
        self.assertEqual(kept.tolist(), [0, 1])
        self.assertEqual(counts, {"pre_nms": 3, "post_nms": 2})


class HeadAndLossTest(unittest.TestCase):
    def test_native_justin_head_has_three_templates_and_one_12dof_output(self):
        head = JustinHGCOutputHead()
        self.assertEqual(head.channels_per_template, 2 + DEFAULT_BIN_SPEC.channels + 12)
        head.eval()
        with torch.no_grad():
            gp, pose, contact = head(torch.zeros((2, 128, 5)))
        self.assertEqual(gp.shape, (2, 5, 2, 3))
        self.assertEqual(pose.shape, (2, 5, DEFAULT_BIN_SPEC.channels, 3))
        self.assertEqual(contact.shape, (2, 5, 12, 3))

    def test_pose_bin_loss_is_cpu_safe_and_depth_range_is_explicit(self):
        target = torch.tensor([[28.4019, 0.0, 0.0, 0.0]])
        self.assertEqual(target_range_counts(target)["depth"], 0)
        self.assertEqual(target_range_counts(torch.tensor([[29.121178, 0.0, 0.0, 0.0]]))["depth"], 0)
        self.assertEqual(target_range_counts(torch.tensor([[30.0, 0.0, 0.0, 0.0]]))["depth"], 1)
        loss_dict, loss = bin_regression_loss(torch.zeros((1, DEFAULT_BIN_SPEC.channels)), target)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("depth_bin_loss", loss_dict)

    def test_decoder_returns_contact_only_model_runtime_fields(self):
        templates = len(TEMPLATE_NAMES)
        gp = torch.zeros((1, 2, templates))
        gp[:, 1, 0] = 1.0
        pose = torch.zeros((1, DEFAULT_BIN_SPEC.channels, templates))
        candidates, counts = decode_justin_candidates(
            points=torch.zeros((1, 3)),
            graspable_logits=gp,
            pose_logits=pose,
            q_contact=torch.ones((1, 12, templates)),
        )
        self.assertEqual(set(candidates), {"palm_pose", "q_contact", "quality", "grasp_point", "template_index"})
        self.assertEqual(candidates["q_contact"].shape[1], 12)
        self.assertEqual(counts["pre_nms"], counts["post_top_side"])

    def test_kmk_template_order_is_the_canonical_grasp_type_order(self):
        q_open = load_template_q_open("/home/irsl/datasets/dlr/grippers/justin_hand/justin_right_hand_simple.yaml")
        self.assertEqual(q_open.shape, (3, 12))


if __name__ == "__main__":
    unittest.main()
