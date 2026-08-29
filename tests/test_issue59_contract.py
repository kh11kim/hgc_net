"""Focused stdlib tests for the Issue #59 contract validator."""

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "issue59_contract_validator", REPO_ROOT / "tools" / "validate_issue59_contract.py"
)
validator = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(validator)
GENERATOR_SPEC = importlib.util.spec_from_file_location(
    "issue59_negative_generator", REPO_ROOT / "tools" / "generate_hgc_negative_points.py"
)
negative_generator = importlib.util.module_from_spec(GENERATOR_SPEC)
assert GENERATOR_SPEC and GENERATOR_SPEC.loader
GENERATOR_SPEC.loader.exec_module(negative_generator)


class ContractValidatorTest(unittest.TestCase):
    def test_current_floor_v7_roots_and_arm_contracts_stay_aligned(self):
        from justin_hgc.data import DEFAULT_DERIVED_ROOT, DEFAULT_ROOT
        from paper_modified_hgc.data import DEFAULT_ROOT as PAPER_DEFAULT_ROOT

        expected_root = Path(
            "/home/irsl/datasets/dlr/compiled/scdm_paper_floor_10000_v7"
        )
        upstream_config = yaml.safe_load(
            (REPO_ROOT / "config" / "issue59_justin_adapter.yaml").read_text()
        )
        paper_config = yaml.safe_load(
            (REPO_ROOT / "config" / "issue59_paper_modified.yaml").read_text()
        )
        upstream_contract = json.loads(
            (REPO_ROOT / "contract" / "issue59_upstream_canonical_v7.json").read_text()
        )
        paper_contract = json.loads(
            (
                REPO_ROOT
                / "contract"
                / "issue59_paper_modified_canonical_v7.json"
            ).read_text()
        )

        self.assertEqual(DEFAULT_ROOT, expected_root)
        self.assertEqual(DEFAULT_DERIVED_ROOT, expected_root / "derived" / "hgc")
        self.assertEqual(PAPER_DEFAULT_ROOT, expected_root)
        self.assertEqual(Path(upstream_config["data"]["root"]), expected_root)
        self.assertEqual(
            Path(upstream_config["data"]["derived_root"]),
            expected_root / "derived" / "hgc",
        )
        self.assertEqual(Path(paper_config["data"]["root"]), expected_root)
        self.assertEqual(
            Path(upstream_contract["canonical_unified_v7"]["default_root"]),
            expected_root,
        )
        self.assertEqual(
            paper_contract["source_contract"],
            "issue59_upstream_canonical_v7.json",
        )
        self.assertEqual(
            Path(paper_contract["canonical_unified_v7"]["default_root"]),
            expected_root,
        )

    def test_stage4_decisions_are_pinned(self):
        contract = json.loads((REPO_ROOT / "contract" / "issue59_upstream_canonical_v3.json").read_text())
        decisions = contract["stage4_decisions"]
        sampling = decisions["point_sampling_policy"]
        self.assertEqual(sampling["fixed_input_count"], 25000)
        self.assertEqual(decisions["point_count_context"]["below_fixed_input_count"], 13)
        self.assertEqual(decisions["post_processing"]["status"], "Settled user decision.")

    def test_live_canonical_metadata_contract(self):
        result = validator.validate(check_upstream=False)
        self.assertEqual(
            result["canonical_dataset"],
            [
                "dataset.yaml",
                "index/splits.json",
                "index/scenes.jsonl",
                "index/views.jsonl",
                "index/samples.jsonl",
            ],
        )

    def test_current_contract_disables_thumb_visibility_filter(self):
        contract = json.loads((REPO_ROOT / "contract" / "issue59_upstream_canonical_v7.json").read_text())
        supervision = contract["stage4_decisions"]["sparse_point_supervision"]
        self.assertEqual(
            supervision["positive_filter"],
            "all_direct_axis_approach_points_no_thumb_visible_mask",
        )

    def test_default_derived_root_tracks_effective_dataset_root(self):
        root = Path("/tmp/custom-canonical-root")
        self.assertEqual(
            negative_generator.resolve_output_root(root, None),
            root / "derived" / "hgc",
        )

    def test_live_upstream_contract(self):
        result = validator.validate(check_dataset=False)
        self.assertIn("model.py", result["upstream"])
        self.assertIn("model/model_027.pth", result["upstream"])

    def test_floor_v7_matched_pose_depth_fits_current_bin_scope(self):
        from justin_hgc.bin_pose import DEFAULT_BIN_SPEC
        from justin_hgc.data import JustinCanonicalDataset

        dataset = JustinCanonicalDataset(
            split="train", sample_ids=["dense_000137_view000"]
        )
        sample = dataset[0]
        positive = sample["template_graspable"] > 0
        depths = sample["template_pose"][..., 0][positive]
        self.assertTrue(torch.any(depths >= 29.0))
        self.assertLess(float(depths.max()), DEFAULT_BIN_SPEC.depth_scope_cm)

    def test_descendant_commit_passes_and_source_mutation_fails(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.py"
            checkpoint = root / "checkpoint.bin"
            source.write_text("original\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            self._git(root, "init")
            self._git(root, "config", "user.email", "issue59-test@example.invalid")
            self._git(root, "config", "user.name", "Issue 59 test")
            self._git(root, "add", "source.py", "checkpoint.bin")
            self._git(root, "commit", "-m", "pinned upstream base")
            base_commit = self._git(root, "rev-parse", "HEAD")
            self._git(root, "remote", "add", "origin", "https://example.invalid/official.git")
            manifest = {
                "immutable_upstream": {
                    "base_commit": base_commit,
                    "official_repository": "https://example.invalid/official.git",
                    "allowed_official_remote_names": ["upstream"],
                    "source_sha256": {"architecture": {"source.py": validator.sha256(source)}},
                    "bundled_checkpoint": {"path": "checkpoint.bin", "sha256": validator.sha256(checkpoint)},
                },
                "development_remote": {
                    "required_remotes": {
                        "origin": "https://example.invalid/fork.git",
                        "upstream": "https://example.invalid/official.git",
                    }
                },
            }
            (root / "implementation_note.md").write_text("descendant change\n", encoding="utf-8")
            self._git(root, "add", "implementation_note.md")
            self._git(root, "commit", "-m", "implementation descendant")
            self._git(root, "remote", "rename", "origin", "upstream")
            self._git(root, "remote", "add", "origin", "https://example.invalid/fork.git")
            self.assertEqual(validator.validate_upstream(root, manifest), ["source.py", "checkpoint.bin"])

            source.write_text("mutated\n", encoding="utf-8")
            with self.assertRaisesRegex(validator.ContractError, "upstream architecture hash"):
                validator.validate_upstream(root, manifest)

    @staticmethod
    def _git(repo: Path, *args: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

if __name__ == "__main__":
    unittest.main()
