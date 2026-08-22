#!/usr/bin/env python3
"""Generate scene-matched HGC negative-point sidecars from canonical v4."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from justin_hgc.geometry import (
    GRID_EDGE_M,
    POINT_COUNT,
    crop_centered_grid,
    deproject_depth_m,
    deterministic_fixed_sample,
    transform_points,
)
from justin_hgc.negative_points import NEGATIVE_SEED_SALT, select_negative_point_indices


DEFAULT_ROOT = Path(
    "/home/irsl/datasets/dlr/compiled/"
    "scdm_justin_right_vgn_train_10000_reconstruction_view_aligned_v4"
)


def _json_lines(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _all_approach_points(root: Path, record: dict[str, Any]) -> np.ndarray:
    with np.load(root / record["grasp_path"], allow_pickle=False) as payload:
        if "approach_point" not in payload:
            raise KeyError(f"{record['grasp_path']} lacks approach_point")
        approach_points = np.asarray(payload["approach_point"], dtype=np.float32)
    if approach_points.ndim != 2 or approach_points.shape[1] != 3:
        raise ValueError(
            f"{record['sample_id']}: approach_point must be Nx3, got {approach_points.shape}"
        )
    return approach_points


def _fixed_points(
    root: Path,
    record: dict[str, Any],
    view: dict[str, Any],
    *,
    point_count: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    with (root / view["view_meta_path"]).open(encoding="utf-8") as source:
        meta = json.load(source)
    transform_key = "T_reconstruction_grid_camera"
    if transform_key not in meta:
        raise KeyError(f"{view['view_meta_path']} lacks {transform_key}")
    depth = np.asarray(Image.open(root / view["depth_path"]))
    camera_points = deproject_depth_m(depth, meta["camera_intrinsics"], depth_scale=0.001)
    grid_points = transform_points(camera_points, np.asarray(meta[transform_key]))
    cropped, _ = crop_centered_grid(grid_points, edge_length_m=GRID_EDGE_M)
    points, sample_indices = deterministic_fixed_sample(
        cropped, count=point_count, key=record["sample_id"]
    )
    return points, sample_indices, len(cropped)


def _write_manifest(
    *,
    output_root: Path,
    source_root: Path,
    point_count: int,
    negative_fraction: float,
    radius_m: float,
    completed_count: int,
) -> None:
    manifest_path = source_root / "dataset.yaml"
    samples_path = source_root / "index" / "samples.jsonl"
    splits_path = source_root / "index" / "splits.json"
    manifest = {
        "format": "hgc_negative_points_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root.resolve()),
        "source_dataset_sha256": _sha256(manifest_path),
        "source_samples_sha256": _sha256(samples_path),
        "source_splits_sha256": _sha256(splits_path),
        "sample_key": "sample_id",
        "point_count": point_count,
        "point_sampling": "deterministic_fixed_sample_v1",
        "positive_match_radius_m": radius_m,
        "positive_filter": "all_approach_points_no_thumb_visible_mask",
        "negative_fraction_of_remaining_points": negative_fraction,
        "negative_seed_salt": NEGATIVE_SEED_SALT,
        "completed_sample_count": completed_count,
    }
    temporary = output_root / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_root / "manifest.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Separate derived root; files are written below OUTPUT_ROOT/neg_points.",
    )
    parser.add_argument("--point-count", type=int, default=POINT_COUNT)
    parser.add_argument("--negative-fraction", type=float, default=0.10)
    parser.add_argument("--positive-radius-m", type=float, default=0.005)
    parser.add_argument("--limit", type=int, help="Process only the first N samples for a smoke run.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output_root = args.output_root.resolve()
    if output_root == root or root in output_root.parents:
        raise ValueError("output-root must be outside the immutable canonical dataset root")
    records = _json_lines(root / "index" / "samples.jsonl")
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("limit must be non-negative")
        records = records[: args.limit]
    views = {record["view_id"]: record for record in _json_lines(root / "index" / "views.jsonl")}
    negative_dir = output_root / "neg_points"
    negative_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for position, record in enumerate(records, start=1):
        sample_id = record["sample_id"]
        destination = negative_dir / f"{sample_id}.npz"
        if destination.exists() and not args.overwrite:
            skipped += 1
            continue
        points, sample_indices, visible_crop_count = _fixed_points(
            root, record, views[record["view_id"]], point_count=args.point_count
        )
        selection = select_negative_point_indices(
            points=points,
            approach_points=_all_approach_points(root, record),
            negative_fraction=args.negative_fraction,
            radius_m=args.positive_radius_m,
            key=sample_id,
        )
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as sink:
            np.savez_compressed(
                sink,
                sample_id=np.asarray(sample_id),
                scene_id=np.asarray(record["scene_id"]),
                view_id=np.asarray(record["view_id"]),
                negative_point_indices=selection.negative_point_indices,
                negative_points=points[selection.negative_point_indices],
                sampled_crop_indices=sample_indices,
                visible_crop_count=np.asarray(visible_crop_count, dtype=np.int64),
                positive_point_count=np.asarray(selection.positive_point_count, dtype=np.int64),
                remaining_point_count=np.asarray(selection.remaining_point_count, dtype=np.int64),
            )
        temporary.replace(destination)
        written += 1
        print(
            json.dumps(
                {
                    "position": position,
                    "sample_id": sample_id,
                    "negative_point_count": len(selection.negative_point_indices),
                    "positive_point_count": selection.positive_point_count,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    _write_manifest(
        output_root=output_root,
        source_root=root,
        point_count=args.point_count,
        negative_fraction=args.negative_fraction,
        radius_m=args.positive_radius_m,
        completed_count=len(list(negative_dir.glob("*.npz"))),
    )
    print(json.dumps({"written": written, "skipped": skipped, "output_root": str(output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
