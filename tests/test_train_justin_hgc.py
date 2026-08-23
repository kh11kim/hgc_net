"""CPU seams for Issue #59 production-training provenance and resume."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch


TRAIN_PATH = Path(__file__).resolve().parents[1] / "tools" / "train_justin_hgc.py"
SPEC = importlib.util.spec_from_file_location("issue59_train", TRAIN_PATH)
assert SPEC and SPEC.loader
train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train)


class TrainingContractTest(unittest.TestCase):
    def test_graspability_f1_ignores_unsupervised_rows(self) -> None:
        logits = torch.tensor(
            [[[[0.0], [1.0]], [[0.0], [1.0]], [[1.0], [0.0]], [[0.0], [1.0]]]]
        )
        labels = torch.tensor([[[1], [0], [1], [-1]]])
        counts = train._graspability_counts(logits, labels)
        self.assertEqual(counts, (1, 1, 1))
        self.assertAlmostEqual(train._f1(*counts), 0.5)

    def test_default_contract_matches_upstream_training_policy(self) -> None:
        config = train.load_training_config()
        self.assertEqual(config["epochs"], 80)
        self.assertEqual(config["batch_size"], 32)
        self.assertEqual(config["workers"], 12)
        self.assertEqual(config["optimizer"], "Adam")
        self.assertEqual(config["learning_rate"], 1e-4)
        self.assertEqual(config["seed"], 0)
        self.assertEqual(config["checkpoint_every_epochs"], 1)
        self.assertEqual(config["output_contract"], "q_contact_only")
        self.assertEqual(config["pose_bins"]["depth_scope_cm"], 30)

    def test_fixed_geometry_contract_rejects_silently_ignored_drift(self) -> None:
        config = train.load_training_config()
        for section, changed in (
            ("pose_supervision", {"direct_axis_min_cosine": 0.9}),
            ("post_processing", {**config["post_processing"], "nms_distance_m": 0.04}),
            ("hand", {**config["hand"], "dof": 11}),
            ("hand", {**config["hand"], "unsupported": True}),
        ):
            drifted = {**config, section: changed}
            with self.assertRaisesRegex(ValueError, section):
                train._validate_common_config(drifted)

    def test_run_directory_never_overwrites_and_checkpoint_resume_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = train.create_timestamped_run_dir(root, "smoke")
            self.assertTrue(run_dir.is_dir())
            with self.assertRaises(FileExistsError):
                train.prepare_run_dir(run_dir, resume=False)

            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            generator = torch.Generator().manual_seed(17)
            _ = torch.randperm(10, generator=generator)
            expected_generator_state = generator.get_state().clone()
            train.save_checkpoint(
                run_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                epoch=3,
                global_step=11,
                config={"seed": 0, "data": {"root": "/tmp/v5"}},
                metrics={"val/total_loss": 1.25},
                best_val=0.75,
                loader_generator=generator,
            )
            restored_generator = torch.Generator().manual_seed(999)
            state = train.load_checkpoint(
                run_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                device=torch.device("cpu"),
                expected_config={"seed": 0, "data": {"root": "/tmp/v5"}, "epochs": 9},
                loader_generator=restored_generator,
            )
            self.assertEqual(state["epoch"], 3)
            self.assertEqual(state["global_step"], 11)
            self.assertEqual(state["metrics"]["val/total_loss"], 1.25)
            self.assertEqual(state["best_val"], 0.75)
            self.assertTrue(torch.equal(restored_generator.get_state(), expected_generator_state))
            with self.assertRaisesRegex(ValueError, "does not match"):
                train.load_checkpoint(
                    run_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    device=torch.device("cpu"),
                    expected_config={"seed": 1, "data": {"root": "/tmp/v5"}},
                )
            with self.assertRaisesRegex(ValueError, "does not match"):
                train.load_checkpoint(
                    run_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    device=torch.device("cpu"),
                    expected_config={"seed": 0, "data": {"root": "/tmp/other"}},
                )
            with self.assertRaisesRegex(ValueError, "does not match"):
                train.load_checkpoint(
                    run_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    device=torch.device("cpu"),
                    expected_config={
                        "seed": 0,
                        "data": {
                            "root": "/tmp/v5",
                            "derived_root": "/tmp/other-derived",
                        },
                    },
                )


if __name__ == "__main__":
    unittest.main()
