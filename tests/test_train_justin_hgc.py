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
    def test_default_contract_matches_upstream_training_policy(self) -> None:
        config = train.load_training_config()
        self.assertEqual(config["epochs"], 80)
        self.assertEqual(config["batch_size"], 1)
        self.assertEqual(config["optimizer"], "Adam")
        self.assertEqual(config["learning_rate"], 1e-4)
        self.assertEqual(config["seed"], 0)
        self.assertEqual(config["checkpoint_every_epochs"], 1)

    def test_run_directory_never_overwrites_and_checkpoint_resume_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = train.create_timestamped_run_dir(root, "smoke")
            self.assertTrue(run_dir.is_dir())
            with self.assertRaises(FileExistsError):
                train.prepare_run_dir(run_dir, resume=False)

            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            train.save_checkpoint(
                run_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                epoch=3,
                global_step=11,
                config={"seed": 0},
                metrics={"val/total_loss": 1.25},
            )
            state = train.load_checkpoint(run_dir / "last.pt", model=model, optimizer=optimizer, device=torch.device("cpu"))
            self.assertEqual(state["epoch"], 3)
            self.assertEqual(state["global_step"], 11)
            self.assertEqual(state["metrics"]["val/total_loss"], 1.25)


if __name__ == "__main__":
    unittest.main()
