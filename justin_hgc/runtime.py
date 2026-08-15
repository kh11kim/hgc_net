"""Decode native Justin HGC tensors to the common palm/q runtime contract."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .bin_pose import DEFAULT_BIN_SPEC, BinPoseSpec, bin_target_to_rotation, decode_pose_bins
from .postprocess import aggressive_nms, top_side_mask


@torch.no_grad()
def decode_justin_candidates(
    *,
    points: torch.Tensor,
    graspable_logits: torch.Tensor,
    pose_logits: torch.Tensor,
    q_contact: torch.Tensor,
    q_squeeze: torch.Tensor,
    template_q_open: torch.Tensor,
    bin_spec: BinPoseSpec = DEFAULT_BIN_SPEC,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Produce ``palm_pose/q_open/q_contact/q_squeeze/quality`` candidates.

    The q_open rows are selected from static KMK template data, never predicted.
    Counts include every candidate before filtering and after top-side plus the
    exact upstream aggressive NMS so evaluation logs can expose both reductions.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be Nx3")
    templates = graspable_logits.shape[-1]
    if graspable_logits.shape != (len(points), 2, templates):
        raise ValueError("graspable_logits must be Nx2xT")
    if pose_logits.shape != (len(points), bin_spec.channels, templates):
        raise ValueError("pose_logits must be NxCxT")
    if q_contact.shape != (len(points), 12, templates) or q_squeeze.shape != q_contact.shape:
        raise ValueError("Justin q_contact/q_squeeze must be Nx12xT")
    if template_q_open.shape != (templates, 12):
        raise ValueError("template_q_open must be Tx12")
    all_items: list[dict[str, torch.Tensor]] = []
    for template in range(templates):
        probability = F.softmax(graspable_logits[:, :, template], dim=1)[:, 1]
        selected = torch.argmax(graspable_logits[:, :, template], dim=1).bool()
        if not torch.any(selected):
            continue
        target = decode_pose_bins(pose_logits[:, :, template][selected], bin_spec)
        rotation = bin_target_to_rotation(target)
        anchor = points[selected]
        palm_position = anchor + rotation[:, :, 2] * (target[:, 0:1] / 100.0)
        all_items.append(
            {
                "palm_position": palm_position,
                "rotation": rotation,
                "q_contact": q_contact[:, :, template][selected],
                "q_squeeze": q_squeeze[:, :, template][selected],
                "q_open": template_q_open[template].expand(int(selected.sum()), -1),
                "quality": probability[selected],
                "grasp_point": anchor,
                "template_index": torch.full((int(selected.sum()),), template, device=points.device, dtype=torch.long),
            }
        )
    if not all_items:
        empty = np.empty((0,), dtype=np.float32)
        return {
            "palm_pose": np.empty((0, 9), dtype=np.float32),
            "q_open": np.empty((0, 12), dtype=np.float32),
            "q_contact": np.empty((0, 12), dtype=np.float32),
            "q_squeeze": np.empty((0, 12), dtype=np.float32),
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
    for key in numpy:
        numpy[key] = numpy[key][kept]
    rotation = numpy.pop("rotation")
    palm_pose = np.concatenate(
        (numpy.pop("palm_position"), rotation[:, :, :2].transpose(0, 2, 1).reshape(len(rotation), 6)), axis=1
    ).astype(np.float32)
    return {
        "palm_pose": palm_pose,
        "q_open": numpy["q_open"].astype(np.float32),
        "q_contact": numpy["q_contact"].astype(np.float32),
        "q_squeeze": numpy["q_squeeze"].astype(np.float32),
        "quality": numpy["quality"].astype(np.float32),
        "grasp_point": numpy["grasp_point"].astype(np.float32),
        "template_index": numpy["template_index"].astype(np.int64),
    }, {"pre_top_side": pre_top_side, "post_top_side": int(top_mask.sum()), **nms_counts}
