"""Decode native Justin HGC tensors to the common palm/q runtime contract."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .bin_pose import DEFAULT_BIN_SPEC, BinPoseSpec, bin_target_to_rotation, decode_pose_bins
from .postprocess import aggressive_nms, top_side_mask


def _farthest_point_backfill(
    *,
    positions: np.ndarray,
    quality: np.ndarray,
    kept: np.ndarray,
    minimum_candidates: int,
) -> np.ndarray:
    """Append suppressed candidates by deterministic palm-position FPS.

    ``positions`` are already in the canonical runtime frame and in metres.  The
    NMS indices remain in their original order; only candidates not in that set
    are considered for backfill.  Lexicographic selection makes ties explicit:
    greater distance, then greater quality, then lower original index.
    """
    positions = np.asarray(positions, dtype=np.float32)
    quality = np.asarray(quality, dtype=np.float32)
    selected = np.asarray(kept, dtype=np.int64).copy()
    if len(selected) >= minimum_candidates or len(positions) < minimum_candidates:
        return selected

    selected_set = {int(index) for index in selected}
    remaining = np.asarray(
        [index for index in range(len(positions)) if index not in selected_set],
        dtype=np.int64,
    )
    if len(selected):
        minimum_distances = np.linalg.norm(
            positions[remaining, None, :] - positions[selected][None, :, :],
            axis=2,
        ).min(axis=1)
    else:
        # Aggressive NMS keeps one candidate for every non-empty pool.  This
        # fallback keeps the helper total for defensive callers that supply
        # an empty NMS result by applying the remaining tie-break rules.
        minimum_distances = np.full(len(remaining), np.inf, dtype=np.float32)
    while len(selected) < minimum_candidates and len(remaining):
        order = np.lexsort(
            (remaining, -quality[remaining], -minimum_distances)
        )
        best_position = int(order[0])
        best_index = int(remaining[best_position])
        selected = np.append(selected, best_index)
        remaining = np.delete(remaining, best_position)
        minimum_distances = np.delete(minimum_distances, best_position)
        if len(remaining):
            distance_to_best = np.linalg.norm(
                positions[remaining] - positions[best_index], axis=1
            )
            minimum_distances = np.minimum(minimum_distances, distance_to_best)
    return selected


@torch.no_grad()
def decode_justin_candidates(
    *,
    points: torch.Tensor,
    graspable_logits: torch.Tensor,
    pose_logits: torch.Tensor,
    q_contact: torch.Tensor,
    bin_spec: BinPoseSpec = DEFAULT_BIN_SPEC,
    candidate_pool_size: int | None = None,
    minimum_candidates: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Produce palm pose, q_contact and quality candidates.

    Execution q_open/q_squeeze waypoints are derived from q_contact by the
    grasp-simulator's Justin kinematics mapper, not by this model runtime.
    Counts include every candidate before filtering and after top-side plus the
    exact upstream aggressive NMS so the server can validate post-processing.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be Nx3")
    templates = graspable_logits.shape[-1]
    if graspable_logits.shape != (len(points), 2, templates):
        raise ValueError("graspable_logits must be Nx2xT")
    if pose_logits.shape != (len(points), bin_spec.channels, templates):
        raise ValueError("pose_logits must be NxCxT")
    if q_contact.shape != (len(points), 12, templates):
        raise ValueError("Justin q_contact must be Nx12xT")
    probability = F.softmax(graspable_logits, dim=1)[:, 1, :]
    selected_pairs: torch.Tensor | None = None
    if candidate_pool_size is not None:
        pool_size = int(candidate_pool_size)
        if pool_size < 1:
            raise ValueError("candidate_pool_size must be positive")
        pool_size = min(pool_size, int(probability.numel()))
        flat_probability = probability.reshape(-1)
        try:
            ranked_pairs = torch.argsort(
                flat_probability, descending=True, stable=True
            )[:pool_size]
        except TypeError:  # pragma: no cover - older torch compatibility
            ranked_pairs = torch.topk(
                flat_probability, k=pool_size, sorted=True
            ).indices
        selected_pairs = torch.zeros_like(flat_probability, dtype=torch.bool)
        selected_pairs[ranked_pairs] = True
        selected_pairs = selected_pairs.reshape_as(probability)
    all_items: list[dict[str, torch.Tensor]] = []
    for template in range(templates):
        template_probability = probability[:, template]
        if selected_pairs is None:
            selected = torch.argmax(
                graspable_logits[:, :, template], dim=1
            ).bool()
        else:
            selected = selected_pairs[:, template]
        if not torch.any(selected):
            continue
        target = decode_pose_bins(pose_logits[:, :, template][selected], bin_spec)
        rotation = bin_target_to_rotation(target)
        anchor = points[selected]
        # Canonical palm local +z points from palm back to the surface anchor.
        # Move from that anchor in -Rz to recover the palm translation.
        palm_position = anchor - rotation[:, :, 2] * (target[:, 0:1] / 100.0)
        all_items.append(
            {
                "palm_position": palm_position,
                "rotation": rotation,
                "q_contact": q_contact[:, :, template][selected],
                "quality": template_probability[selected],
                "grasp_point": anchor,
                "template_index": torch.full((int(selected.sum()),), template, device=points.device, dtype=torch.long),
            }
        )
    if not all_items:
        empty = np.empty((0,), dtype=np.float32)
        return {
            "palm_pose": np.empty((0, 9), dtype=np.float32),
            "q_contact": np.empty((0, 12), dtype=np.float32),
            "quality": empty,
            "grasp_point": np.empty((0, 3), dtype=np.float32),
            "template_index": np.empty((0,), dtype=np.int64),
        }, {"pre_top_side": 0, "post_top_side": 0, "pre_nms": 0, "post_nms": 0}
    joined = {key: torch.cat([item[key] for item in all_items], dim=0) for key in all_items[0]}
    numpy = {key: value.detach().cpu().numpy() for key, value in joined.items()}
    top_mask = top_side_mask(numpy["grasp_point"], numpy["palm_position"])
    pre_top_side = len(top_mask)
    for key in numpy:
        numpy[key] = numpy[key][top_mask]
    kept, nms_counts = aggressive_nms(
        positions=numpy["palm_position"], rotations=numpy["rotation"], scores=numpy["quality"]
    )
    nms_kept = len(kept)
    if minimum_candidates is not None:
        minimum = int(minimum_candidates)
        if minimum < 1:
            raise ValueError("minimum_candidates must be positive")
        if nms_kept < minimum <= len(numpy["quality"]):
            kept = _farthest_point_backfill(
                positions=numpy["palm_position"],
                quality=numpy["quality"],
                kept=kept,
                minimum_candidates=minimum,
            )
    for key in numpy:
        numpy[key] = numpy[key][kept]
    rotation = numpy.pop("rotation")
    palm_pose = np.concatenate(
        (numpy.pop("palm_position"), rotation[:, :, :2].transpose(0, 2, 1).reshape(len(rotation), 6)), axis=1
    ).astype(np.float32)
    return {
        "palm_pose": palm_pose,
        "q_contact": numpy["q_contact"].astype(np.float32),
        "quality": numpy["quality"].astype(np.float32),
        "grasp_point": numpy["grasp_point"].astype(np.float32),
        "template_index": numpy["template_index"].astype(np.int64),
    }, {
        "pre_top_side": pre_top_side,
        "post_top_side": int(top_mask.sum()),
        **nms_counts,
        "nms_backfill": int(len(kept) - nms_kept),
        "post_backfill": int(len(kept)),
    }
