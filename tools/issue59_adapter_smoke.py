#!/usr/bin/env python3
"""CPU smoke for one canonical-v3 view; no PointNet++ CUDA code is loaded."""

from __future__ import annotations

import argparse
import json

import numpy as np

from justin_hgc.data import DEFAULT_ROOT, JustinCanonicalDataset
from justin_hgc.bin_pose import target_range_counts
from justin_hgc.geometry import POINT_COUNT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    dataset = JustinCanonicalDataset(args.root, split=args.split, point_count=POINT_COUNT)
    item = dataset.load_numpy(args.index)
    if not np.isfinite(item["point"]).all():
        raise RuntimeError("adapter produced non-finite point coordinates")
    if int(item["matched_positive_count"]) == 0:
        raise RuntimeError("selected smoke view has no visible positive approach-point match")
    labels = item["template_graspable"]
    range_counts = {}
    for template in range(labels.shape[1]):
        positive = labels[:, template] == 1
        range_counts[str(template)] = target_range_counts(item["template_pose"][:, template][positive])
    if any(value for counts in range_counts.values() for value in counts.values()):
        raise RuntimeError(f"smoke found pose targets outside configured range: {range_counts}")
    result = {
        "sample_id": item["sample_id"],
        "point_shape": list(item["point"].shape),
        "visible_crop_count": int(item["visible_crop_count"]),
        "visible_positive_count": int(item["visible_positive_count"]),
        "matched_positive_count": int(item["matched_positive_count"]),
        "positive_point_labels": int((labels == 1).sum()),
        "negative_point_labels": int((labels == 0).sum()),
        "ignored_point_labels": int((labels == -1).sum()),
        "positive_pose_out_of_range": range_counts,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
