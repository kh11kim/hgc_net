"""Sampling and pose reconstruction for paper-modified HGC inference."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from justin_hgc.postprocess import aggressive_nms, top_side_mask
from justin_hgc.runtime import _farthest_point_backfill

from .pose import bin_and_residual_to_mat


def _stable_descending(values: torch.Tensor) -> torch.Tensor:
    """Stable descending order for deterministic top-k tie handling."""

    try:
        return torch.argsort(values, descending=True, stable=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        # CPU numpy's stable sort is deterministic and this path is only used
        # by old environments that do not expose argsort(stable=...).
        order = np.argsort(-values.detach().cpu().numpy(), kind="stable")
        return torch.from_numpy(order).to(device=values.device)


def farthest_point_indices(points: torch.Tensor, count: int) -> torch.Tensor:
    """Reference FPS with deterministic first point and no duplicate rows."""

    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must be Nx3, got {tuple(points.shape)}")
    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive")
    if len(points) == 0:
        raise ValueError("cannot FPS an empty point set")
    count = min(count, len(points))
    distance_points = points.to(dtype=torch.float32)
    selected = torch.empty((count,), dtype=torch.long, device=points.device)
    selected[0] = 0
    closest = torch.full(
        (len(points),), float("inf"), dtype=distance_points.dtype, device=points.device
    )
    selected_mask = torch.zeros((len(points),), dtype=torch.bool, device=points.device)
    selected_mask[0] = True
    for index in range(1, count):
        delta = distance_points - distance_points[selected[index - 1]]
        closest = torch.minimum(closest, (delta * delta).sum(dim=-1))
        # Mask selected rows explicitly.  A geometric zero-distance tie is
        # valid for duplicate voxel centres, but FPS must still return unique
        # row indices so downstream backfill/NMS never duplicates candidates.
        masked = closest.masked_fill(selected_mask, float("-inf"))
        selected[index] = torch.argmax(masked)
        selected_mask[selected[index]] = True
    return selected


def _unpack_prediction(output: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(output, Mapping):
        names = {
            "quality_logits": ("quality_logits", "grasp_qual_grid", "quality"),
            "orientation_logits": ("orientation_logits", "bin_logits", "grasp_bin_logits"),
            "residual": ("residual", "reg_pred", "regression"),
            "q_contact": ("q_contact", "contact"),
        }
        values = []
        for name, aliases in names.items():
            found = next((output[key] for key in aliases if key in output), None)
            if found is None:
                raise RuntimeError(f"paper-modified model output is missing {name}")
            values.append(found)
        return tuple(values)  # type: ignore[return-value]
    if hasattr(output, "quality_logits"):
        return (
            output.quality_logits,
            output.orientation_logits,
            output.residual,
            output.q_contact,
        )
    if isinstance(output, (tuple, list)) and len(output) == 4:
        return tuple(output)  # type: ignore[return-value]
    raise RuntimeError("paper-modified model must return quality/orientation/residual/q_contact outputs")


@torch.no_grad()
def decode_paper_candidates(
    *,
    points: torch.Tensor,
    feature_indices: torch.Tensor | None = None,
    quality_logits: torch.Tensor,
    orientation_logits: torch.Tensor,
    residual: torch.Tensor,
    q_contact: torch.Tensor,
    template_indices: torch.Tensor | None = None,
    topk: int = 100,
    num_samples: int = 100,
    minimum_candidates: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Select quality-volume peaks, FPS them, and reconstruct Justin poses.

    ``points`` are the object-occupancy voxel centres corresponding to the
    selected-row predictions.  The quality head remains dense; restricting
    its values to these points is the runtime equivalent of the reference
    quality-volume top-k before pose decoding.
    """

    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("points must be Nx3")
    if quality_logits.ndim == 5:
        if quality_logits.shape[0] != 1 or quality_logits.shape[1] != 1:
            raise ValueError("quality_logits must have shape [1,1,D,H,W]")
        quality_volume = quality_logits[0, 0]
    elif quality_logits.ndim == 3:
        quality_volume = quality_logits
    else:
        raise ValueError("quality_logits must be [1,1,D,H,W] or [D,H,W]")
    if orientation_logits.ndim == 3:
        orientation_logits = orientation_logits[0]
    if residual.ndim == 3:
        residual = residual[0]
    if q_contact.ndim == 3:
        q_contact = q_contact[0]
    if orientation_logits.ndim != 2 or orientation_logits.shape[0] != len(points):
        raise ValueError("orientation_logits must be [N,864]")
    if orientation_logits.shape[1] != 864:
        raise ValueError("paper-modified orientation logits must have 864 channels")
    if residual.shape != (len(points), 4):
        raise ValueError("paper-modified residual must be [N,4]")
    if q_contact.shape != (len(points), 12):
        raise ValueError("paper-modified q_contact must be [N,12]")
    if len(points) == 0:
        raise ValueError("paper-modified runtime received no object voxel centres")
    if quality_volume.ndim != 3:
        raise ValueError("quality volume must be D,H,W")
    quality = torch.sigmoid(quality_volume)
    # Points are DHW indices converted to xyz by the server.  Reconstructing
    # the indices here from points is intentionally avoided; callers pass
    # values in the same row order as the sparse predictions.  The runtime
    # helper therefore also accepts a [N,3] integer index view via attributes
    # below when available, and otherwise derives nearest grid centres.
    index_like = feature_indices if feature_indices is not None else getattr(points, "feature_indices", None)
    if index_like is None:
        # The server's voxel-center transform is affine and exact at this
        # resolution.  This fallback keeps the public helper convenient for
        # tests that construct points directly from that transform.
        grid_size = quality.shape[0]
        edge = 0.5
        norm_scale = (edge - edge / grid_size) / 2.0
        xyz = np.rint(
            (points.detach().cpu().numpy() / norm_scale + 1.0) * 0.5 * (grid_size - 1)
        ).astype(np.int64)
        indices = torch.from_numpy(xyz[:, [2, 1, 0]]).to(device=points.device)
    else:  # pragma: no cover - tensor attributes are not normally used
        indices = torch.as_tensor(index_like, device=points.device, dtype=torch.long)
    if indices.shape != (len(points), 3):
        raise ValueError("feature indices must be Nx3")
    point_quality = quality[indices[:, 0], indices[:, 1], indices[:, 2]]
    topk = min(int(topk), len(points))
    num_samples = min(int(num_samples), topk)
    if topk <= 0 or num_samples <= 0:
        raise ValueError("topk and num_samples must be positive")
    top_order = _stable_descending(point_quality)[:topk]
    fps_local = farthest_point_indices(points[top_order], num_samples)
    selected = top_order[fps_local]

    selected_bin = torch.argmax(orientation_logits[selected], dim=-1)
    selected_residual = residual[selected]
    rotation = bin_and_residual_to_mat(selected_bin, selected_residual[:, 1:])
    anchor = points[selected]
    palm_position = anchor - rotation[:, :, 2] * selected_residual[:, :1]
    contact = q_contact[selected]
    if template_indices is None:
        selected_templates = torch.full(
            (len(selected),), -1, dtype=torch.long, device=selected.device
        )
    else:
        selected_templates = torch.as_tensor(
            template_indices, device=selected.device, dtype=torch.long
        )[selected]
        if selected_templates.shape != (len(selected),):
            raise ValueError("template_indices must align with the sparse feature rows")
    # q_open/q_squeeze are execution waypoints derived by the grasp-sim
    # JustinContactWaypointMapper from q_contact.  The model runtime returns
    # only the learned contact posture, matching the shared server contract.
    quality_selected = point_quality[selected]
    candidate = {
        "palm_position": palm_position,
        "rotation": rotation,
        "q_contact": contact,
        "quality": quality_selected,
        "grasp_point": anchor,
        "template_index": selected_templates,
    }
    numpy = {key: value.detach().cpu().numpy() for key, value in candidate.items()}
    top_mask = top_side_mask(numpy["grasp_point"], numpy["palm_position"])
    pre_top_side = len(top_mask)
    for key in numpy:
        numpy[key] = numpy[key][top_mask]
    if len(numpy["quality"]):
        kept, counts = aggressive_nms(
            positions=numpy["palm_position"],
            rotations=numpy["rotation"],
            scores=numpy["quality"],
        )
    else:
        kept = np.empty((0,), dtype=np.int64)
        counts = {"pre_nms": 0, "post_nms": 0}
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
    # NMS emits descending scores; FPS backfill appends spatially diverse
    # rows, so restore the public quality-descending contract after backfill.
    if len(numpy["quality"]):
        score_order = np.argsort(-numpy["quality"], kind="stable")
        for key in numpy:
            numpy[key] = numpy[key][score_order]
    rotation_numpy = numpy.pop("rotation")
    palm_pose = np.concatenate(
        (
            numpy.pop("palm_position"),
            rotation_numpy[:, :, :2].transpose(0, 2, 1).reshape(len(rotation_numpy), 6),
        ),
        axis=1,
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
        **counts,
        "nms_backfill": int(len(kept) - nms_kept),
        "post_backfill": int(len(kept)),
        "quality_topk": int(topk),
        "quality_fps": int(num_samples),
    }


__all__ = ["decode_paper_candidates", "farthest_point_indices"]
