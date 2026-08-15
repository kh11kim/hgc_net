#!/usr/bin/env python3
"""Validate Issue #59's immutable upstream and canonical unified-v3 contract.

This uses only the standard library and reads dataset metadata only: dataset.yaml
and index/splits.json.  It deliberately does not open NPZ/HDF5 arrays or perform
any model, CUDA, or training work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "contract" / "issue59_upstream_canonical_v3.json"


class ContractError(RuntimeError):
    """The local checkout or selected canonical dataset violates the pinned contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(actual: Any, expected: Any, description: str) -> None:
    if actual != expected:
        raise ContractError(f"{description}: expected {expected!r}, found {actual!r}")


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ContractError(f"cannot inspect git checkout at {repo}: {error}") from error


def _git_success(repo: Path, *args: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError as error:
        raise ContractError(f"cannot inspect git checkout at {repo}: {error}") from error
    return result.returncode == 0


def validate_upstream(repo: Path, contract: dict[str, Any]) -> list[str]:
    upstream = contract["immutable_upstream"]
    base_commit = upstream["base_commit"]
    _git(repo, "cat-file", "-e", f"{base_commit}^{{commit}}")
    if not _git_success(repo, "merge-base", "--is-ancestor", base_commit, "HEAD"):
        raise ContractError(f"pinned upstream base {base_commit} is not an ancestor of HEAD")
    remote_names = set(_git(repo, "remote").splitlines())
    allowed_remotes = upstream["allowed_official_remote_names"]
    matching_remotes = [
        remote_name
        for remote_name in allowed_remotes
        if remote_name in remote_names
        and _git(repo, "remote", "get-url", remote_name) == upstream["official_repository"]
    ]
    if not matching_remotes:
        raise ContractError(
            "official upstream URL is not configured on an allowed remote "
            f"{allowed_remotes!r}; available remotes are {sorted(remote_names)!r}"
        )
    development_remote = contract.get("development_remote")
    if development_remote is not None:
        for remote_name, expected_url in development_remote["required_remotes"].items():
            if remote_name not in remote_names:
                raise ContractError(f"required development remote is missing: {remote_name}")
            _require(
                _git(repo, "remote", "get-url", remote_name),
                expected_url,
                f"development remote URL for {remote_name}",
            )
    checked: list[str] = []
    for group, files in upstream["source_sha256"].items():
        for relative_path, expected_hash in files.items():
            path = repo / relative_path
            if not path.is_file():
                raise ContractError(f"upstream {group} source missing: {path}")
            _require(sha256(path), expected_hash, f"upstream {group} hash for {relative_path}")
            checked.append(relative_path)
    checkpoint = upstream["bundled_checkpoint"]
    checkpoint_path = repo / checkpoint["path"]
    if not checkpoint_path.is_file():
        raise ContractError(f"bundled checkpoint missing: {checkpoint_path}")
    _require(sha256(checkpoint_path), checkpoint["sha256"], "bundled checkpoint hash")
    checked.append(checkpoint["path"])
    return checked


def _yaml_scalar(text: str, key: str) -> str:
    # dataset.yaml is immutable by its SHA256. This intentionally tiny parser only
    # extracts scalar metadata, keeping the validator stdlib-only.
    import re

    match = re.search(rf"(?m)^\s*{re.escape(key)}:\s*([^#\n]+?)\s*$", text)
    if not match:
        raise ContractError(f"dataset.yaml is missing scalar {key!r}")
    return match.group(1).strip().strip('"\'')


def validate_dataset(dataset_root: Path, contract: dict[str, Any]) -> list[str]:
    dataset = contract["canonical_unified_v3"]
    hashes = dataset["file_sha256"]
    checked: list[str] = []
    for relative_path, expected_hash in hashes.items():
        path = dataset_root / relative_path
        if not path.is_file():
            raise ContractError(f"canonical metadata missing: {path}")
        _require(sha256(path), expected_hash, f"canonical metadata hash for {relative_path}")
        checked.append(relative_path)
    for relative_path, expected_line_count in dataset["metadata_line_counts"].items():
        with (dataset_root / relative_path).open(encoding="utf-8") as source:
            actual_line_count = sum(1 for _ in source)
        _require(actual_line_count, expected_line_count, f"canonical metadata line count for {relative_path}")

    # The SHA256 pins the complete YAML manifest; these explicit checks make the
    # intended contract visible in a failure report without opening sample arrays.
    manifest_text = (dataset_root / "dataset.yaml").read_text(encoding="utf-8")
    expected = dataset["dataset_contract"]
    _require(_yaml_scalar(manifest_text, "format"), expected["format"], "dataset format")
    _require(int(_yaml_scalar(manifest_text, "format_version")), expected["format_version"], "dataset format_version")
    _require(int(_yaml_scalar(manifest_text, "seed")), expected["split_seed"], "split seed")
    hand = expected["hand"]
    _require(_yaml_scalar(manifest_text, "name"), hand["name"], "hand name")
    _require(_yaml_scalar(manifest_text, "palm_frame"), hand["palm_frame"], "hand palm frame")
    _require(_yaml_scalar(manifest_text, "base_link"), hand["base_link"], "hand base link")
    _require(int(_yaml_scalar(manifest_text, "dof")), hand["dof"], "hand dof")
    for joint_name in hand["joint_order"]:
        if f"  - {joint_name}\n" not in manifest_text:
            raise ContractError(f"dataset joint order is missing {joint_name!r}")
    for source in expected["sources"]:
        if f"- {source}\n" not in manifest_text:
            raise ContractError(f"dataset sources are missing {source!r}")
    for stat_name, stat_value in expected["stats"].items():
        _require(int(_yaml_scalar(manifest_text, stat_name)), stat_value, f"dataset stats.{stat_name}")

    with (dataset_root / "index" / "splits.json").open(encoding="utf-8") as source:
        splits = json.load(source)
    if not isinstance(splits, dict):
        raise ContractError("index/splits.json must be a scene-to-split object")
    _require(dict(sorted(Counter(splits.values()).items())), dict(sorted(dataset["split_counts"].items())), "split counts")
    _require(len(splits), expected["stats"]["scenes"], "split scene count")
    return checked


def validate(
    manifest_path: Path = DEFAULT_MANIFEST,
    dataset_root: Path | None = None,
    *,
    check_upstream: bool = True,
    check_dataset: bool = True,
) -> dict[str, list[str]]:
    contract = json.loads(manifest_path.read_text(encoding="utf-8"))
    if dataset_root is None:
        dataset_root = Path(contract["canonical_unified_v3"]["default_root"])
    result: dict[str, list[str]] = {}
    if check_upstream:
        result["upstream"] = validate_upstream(manifest_path.parent.parent, contract)
    if check_dataset:
        result["canonical_dataset"] = validate_dataset(dataset_root, contract)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--upstream-only", action="store_true")
    parser.add_argument("--dataset-only", action="store_true")
    args = parser.parse_args()
    if args.upstream_only and args.dataset_only:
        parser.error("--upstream-only and --dataset-only cannot be used together")
    try:
        result = validate(
            args.manifest,
            args.dataset_root,
            check_upstream=not args.dataset_only,
            check_dataset=not args.upstream_only,
        )
    except (ContractError, json.JSONDecodeError, OSError, ValueError) as error:
        print(f"CONTRACT INVALID: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "ok", "checked": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
