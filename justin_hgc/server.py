"""Checkpoint-driven HGC inference service over pickle-free NPZ/ZMQ."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from .checkpoint import (
    LEGACY_UPSTREAM_ARM,
    PAPER_MODIFIED_ARM,
    CheckpointMetadata,
    load_checkpoint,
)
from .geometry import (
    GRID_EDGE_M,
    crop_centered_grid,
    deproject_depth_m,
    deterministic_seed_sample,
    transform_points,
)
from .protocol import (
    PROTOCOL_VERSION,
    decode_message,
    error_response,
    encode_message,
    scalar,
    success_response,
)
from .runtime import decode_justin_candidates
from .templates import TEMPLATE_NAMES


GRID_SHAPE = (64, 64, 64)
POINT_COUNT = 25_000
CANDIDATE_COUNT = 100
OUTPUT_FIELDS = ("T_grid_palm", "q_contact", "quality")


class HGCGenerationError(RuntimeError):
    """Raised when an input cannot produce the required 100 valid candidates."""


def dilate_object_mask_6(mask: torch.Tensor) -> torch.Tensor:
    """Add the six face-adjacent voxels used by approach-point supervision."""

    if mask.ndim != 3:
        raise ValueError(f"object mask must be 3D, got {tuple(mask.shape)}")
    result = mask.bool().clone()
    for axis in range(3):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(1, None)
        upper[axis] = slice(None, -1)
        result[tuple(lower)] |= mask[tuple(upper)].bool()
        result[tuple(upper)] |= mask[tuple(lower)].bool()
    return result


class HGCModel(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


def _grid_index_to_point(indices: np.ndarray) -> np.ndarray:
    """Map DHW/ZYX indices to canonical 0.5 m align-corners voxel centres."""

    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError(f"indices must be Nx3, got {indices.shape}")
    voxel = GRID_EDGE_M / GRID_SHAPE[0]
    norm_scale = (GRID_EDGE_M - voxel) / 2.0
    xyz_index = indices[:, [2, 1, 0]].astype(np.float32, copy=False)
    normalized = (xyz_index / float(GRID_SHAPE[0] - 1)) * 2.0 - 1.0
    return (normalized * norm_scale).astype(np.float32, copy=False)


def _validate_intrinsics(intrinsics: np.ndarray) -> None:
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape (3,3), got {intrinsics.shape}")
    if not np.isfinite(intrinsics).all():
        raise ValueError("intrinsics must be finite")
    if float(intrinsics[0, 0]) <= 0.0 or float(intrinsics[1, 1]) <= 0.0:
        raise ValueError("intrinsics fx and fy must be positive")


def preprocess_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    T_grid_camera: np.ndarray,
    *,
    seed: int,
    point_count: int = POINT_COUNT,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Deproject, transform, crop and seeded-sample one depth observation.

    Depth values are metres, matching ``grasp_sim.ReconstructionObservation``.
    Direct callers get the same no-return sanitization as ``grasp_sim``;
    protocol handlers validate finite wire values before calling this helper.
    """

    depth = np.asarray(depth)
    intrinsics = np.asarray(intrinsics)
    T_grid_camera = np.asarray(T_grid_camera)
    if depth.ndim != 2:
        raise ValueError(f"depth must have shape HxW, got {depth.shape}")
    _validate_intrinsics(intrinsics)
    if T_grid_camera.shape != (4, 4):
        raise ValueError(
            f"T_grid_camera must have shape (4,4), got {T_grid_camera.shape}"
        )
    if not np.isfinite(T_grid_camera).all():
        raise ValueError("T_grid_camera must be finite")
    # MuJoCo no-return pixels are NaN.  Zero is the protocol-safe unobserved
    # convention and is ignored by deprojection.
    depth = np.nan_to_num(depth.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    camera_points = deproject_depth_m(
        depth,
        {
            "fx": float(intrinsics[0, 0]),
            "fy": float(intrinsics[1, 1]),
            "cx": float(intrinsics[0, 2]),
            "cy": float(intrinsics[1, 2]),
        },
        depth_scale=1.0,
    )
    grid_points = transform_points(camera_points, T_grid_camera)
    cropped, _ = crop_centered_grid(grid_points, edge_length_m=GRID_EDGE_M)
    if len(cropped) == 0:
        raise HGCGenerationError(
            "depth observation has no valid points in the centred 0.5 m crop"
        )
    sampled, indices = deterministic_seed_sample(
        cropped, count=int(point_count), seed=int(seed)
    )
    return sampled, indices, {
        "input_valid_count": int(len(camera_points)),
        "cropped_count": int(len(cropped)),
        "sampled_count": int(len(sampled)),
    }


def _as_output_tensors(output: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(output, Mapping):
        names = ("graspable_logits", "pose_logits", "q_contact")
        aliases = {
            "graspable_logits": ("graspable_logits", "gp", "graspable"),
            "pose_logits": ("pose_logits", "pose"),
            "q_contact": ("q_contact", "contact"),
        }
        values = []
        for name in names:
            value = next((output[key] for key in aliases[name] if key in output), None)
            if value is None:
                raise RuntimeError(f"HGC model output is missing {name}")
            values.append(value)
    elif isinstance(output, (tuple, list)) and len(output) == 3:
        values = list(output)
    else:
        raise RuntimeError("HGC model must return three prediction tensors")
    result = tuple(value if isinstance(value, torch.Tensor) else torch.as_tensor(value) for value in values)
    if any(value.ndim != 4 for value in result):
        raise RuntimeError("HGC prediction tensors must have shape [B,N,C,T]")
    if any(value.shape[0] != 1 for value in result):
        raise RuntimeError("HGC service accepts one observation per request")
    if any(not bool(torch.isfinite(value).all()) for value in result):
        raise RuntimeError("HGC model returned non-finite prediction values")
    return tuple(value[0] for value in result)


def _pose9d_to_transform(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.ndim != 2 or pose.shape[1] != 9:
        raise RuntimeError(f"decoder returned invalid palm_pose shape {pose.shape}")
    first = pose[:, 3:6]
    second = pose[:, 6:9]
    first = first / np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1.0e-12)
    second = second - np.sum(first * second, axis=1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1.0e-12)
    third = np.cross(first, second)
    transform = np.repeat(np.eye(4, dtype=np.float32)[None], len(pose), axis=0)
    transform[:, :3, :3] = np.stack((first, second, third), axis=2)
    transform[:, :3, 3] = pose[:, :3]
    return transform


def _validate_transforms(transforms: np.ndarray, *, name: str) -> None:
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape [N,4,4], got {transforms.shape}")
    if not np.isfinite(transforms).all():
        raise ValueError(f"{name} must be finite")
    if not np.allclose(transforms[:, 3, :], np.asarray((0, 0, 0, 1), dtype=np.float32), atol=1e-4, rtol=0.0):
        raise ValueError(f"{name} has invalid homogeneous bottom row")
    rotations = transforms[:, :3, :3]
    if not np.allclose(rotations @ np.swapaxes(rotations, 1, 2), np.eye(3), atol=2e-3, rtol=0.0):
        raise ValueError(f"{name} has invalid rotation matrices")
    if np.any(np.linalg.det(rotations) <= 0.0):
        raise ValueError(f"{name} has non-positive rotation determinant")


class HGCService:
    """Serve one strict-loaded HGC model and exactly 100 ranked candidates."""

    def __init__(
        self,
        model: HGCModel,
        *,
        device: str | torch.device = "cpu",
        arm: str = LEGACY_UPSTREAM_ARM,
        metadata: CheckpointMetadata | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_sha256: str = "unknown",
        checkpoint_epoch: int | None = None,
        checkpoint_global_step: int | None = None,
        source_commit: str = "unknown",
    ) -> None:
        if metadata is not None and arm in {LEGACY_UPSTREAM_ARM, "upstream", "upstream_faithful"}:
            arm = metadata.arm
        arm = {"upstream": LEGACY_UPSTREAM_ARM, "upstream_faithful": LEGACY_UPSTREAM_ARM}.get(arm, arm)
        if arm not in {LEGACY_UPSTREAM_ARM, PAPER_MODIFIED_ARM}:
            raise ValueError(f"unsupported HGC arm {arm!r}")
        self.device = torch.device(device)
        self.arm = arm
        self.model = model
        if isinstance(model, torch.nn.Module):
            model.to(self.device).eval()
        self.metadata = metadata
        self.checkpoint_path = str(metadata.path if metadata is not None else (Path(checkpoint_path).resolve() if checkpoint_path else "<in-process>"))
        self.checkpoint_sha256 = metadata.sha256 if metadata is not None else str(checkpoint_sha256)
        self.checkpoint_epoch = metadata.epoch if metadata is not None else checkpoint_epoch
        self.checkpoint_global_step = metadata.global_step if metadata is not None else checkpoint_global_step
        self.source_commit = metadata.source_commit if metadata is not None else str(source_commit or "unknown")
        if self.source_commit == "":
            self.source_commit = "unknown"

    @property
    def input_representation(self) -> str:
        return "gt_full_occupancy_ground" if self.arm == PAPER_MODIFIED_ARM else "depth_point_cloud"

    def handle(self, payload: bytes) -> bytes:
        try:
            request = decode_message(payload)
            version_array = request.get("protocol_version")
            if version_array is None or version_array.shape != () or version_array.dtype != np.dtype(np.int16):
                raise TypeError("protocol_version must be scalar int16")
            version = int(version_array.item())
            if version != PROTOCOL_VERSION:
                raise ValueError(f"unsupported protocol_version: {version}")
            op_array = request.get("op")
            if op_array is None or op_array.shape != () or op_array.dtype.kind not in {"U", "S"}:
                raise TypeError("op must be a scalar string")
            operation = str(op_array.item())
            if operation == "describe":
                return self._describe()
            if operation != "infer":
                raise ValueError(f"unsupported op: {operation!r}")
            return self._infer(request)
        except Exception as error:
            return error_response(error)

    def _describe(self) -> bytes:
        epoch = -1 if self.checkpoint_epoch is None else int(self.checkpoint_epoch)
        global_step = -1 if self.checkpoint_global_step is None else int(self.checkpoint_global_step)
        return success_response(
            service=np.asarray("hgc"),
            arm=np.asarray(self.arm),
            checkpoint_path=np.asarray(self.checkpoint_path),
            checkpoint_sha256=np.asarray(self.checkpoint_sha256),
            checkpoint_epoch=np.asarray(epoch, dtype=np.int64),
            checkpoint_global_step=np.asarray(global_step, dtype=np.int64),
            epoch=np.asarray(epoch, dtype=np.int64),
            global_step=np.asarray(global_step, dtype=np.int64),
            source_commit=np.asarray(self.source_commit),
            trusted_pickle=np.asarray(True),
            input_representation=np.asarray(self.input_representation),
            candidate_count=np.asarray(CANDIDATE_COUNT, dtype=np.int16),
            quality_order=np.asarray("descending"),
            template_order=np.asarray(TEMPLATE_NAMES),
            output_fields=np.asarray(OUTPUT_FIELDS),
        )

    def _required_array(
        self, request: Mapping[str, np.ndarray], name: str, dtype: np.dtype[Any]
    ) -> np.ndarray:
        if name not in request:
            raise KeyError(f"request is missing {name}")
        value = request[name]
        expected = np.dtype(dtype)
        # Check wire dtype before any coercion.  This preserves protocol errors
        # instead of silently accepting a caller's wrong representation.
        if value.dtype != expected:
            raise TypeError(f"{name} must be {expected.name}, got {value.dtype}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
        return value

    @staticmethod
    def _seed(request: Mapping[str, np.ndarray]) -> int:
        if "seed" not in request:
            raise KeyError("request is missing seed")
        value = request["seed"]
        if value.shape != () or value.dtype.kind not in {"i", "u"}:
            raise TypeError("seed must be a scalar integer")
        seed = int(value.item())
        if seed < 0:
            raise ValueError("seed must be non-negative")
        return seed

    @staticmethod
    def _candidate_count(request: Mapping[str, np.ndarray]) -> None:
        if "candidate_count" not in request:
            raise KeyError("request is missing candidate_count")
        value = request["candidate_count"]
        if value.shape != () or value.dtype.kind not in {"i", "u"}:
            raise TypeError("candidate_count must be a scalar integer")
        if int(value.item()) != CANDIDATE_COUNT:
            raise ValueError("HGC service supports exactly candidate_count=100")

    @torch.inference_mode()
    def _infer(self, request: dict[str, np.ndarray]) -> bytes:
        seed = self._seed(request)
        self._candidate_count(request)
        if self.arm == PAPER_MODIFIED_ARM:
            if any(name in request for name in ("depth", "intrinsics", "T_grid_camera", "target_mask")):
                raise ValueError(
                    "paper_modified HGC accepts only the GT input_grid representation"
                )
            model_input, diagnostics = self._prepare_paper(request)
            feature_grid, quality_logits, ranked_indices = self._run_paper_grid(model_input)
            from paper_modified_hgc.runtime import _unpack_prediction, decode_paper_candidates

            pool_size = CANDIDATE_COUNT
            pool_expansions = 0
            while True:
                points, feature_indices, model_output = self._run_paper_selected(
                    feature_grid, quality_logits, ranked_indices[:pool_size]
                )
                _, orientation_logits, residual, q_contact = _unpack_prediction(model_output)
                candidates, counts = decode_paper_candidates(
                    points=torch.from_numpy(np.ascontiguousarray(points)).to(self.device),
                    feature_indices=torch.from_numpy(np.ascontiguousarray(feature_indices)).to(self.device),
                    quality_logits=quality_logits,
                    orientation_logits=orientation_logits,
                    residual=residual,
                    q_contact=q_contact,
                    topk=len(points),
                    num_samples=len(points),
                    minimum_candidates=CANDIDATE_COUNT,
                )
                if len(candidates["quality"]) >= CANDIDATE_COUNT:
                    break
                if pool_size >= len(ranked_indices):
                    raise HGCGenerationError(
                        "paper_modified HGC exhausted the dense quality volume "
                        f"but produced only {len(candidates['quality'])} valid candidates"
                    )
                pool_size = min(pool_size * 2, len(ranked_indices))
                pool_expansions += 1
            diagnostics["candidate_pool_count"] = int(pool_size)
            diagnostics["candidate_pool_expansions"] = int(pool_expansions)
        else:
            if any(name in request for name in ("input_grid", "target_mask")):
                raise ValueError(
                    "upstream_faithful HGC accepts only depth/intrinsics/T_grid_camera"
                )
            depth = self._required_array(request, "depth", np.dtype(np.float32))
            intrinsics = self._required_array(request, "intrinsics", np.dtype(np.float32))
            transform = self._required_array(request, "T_grid_camera", np.dtype(np.float32))
            if depth.ndim != 2:
                raise ValueError(f"depth must have shape HxW, got {depth.shape}")
            _validate_intrinsics(intrinsics)
            if transform.shape != (4, 4):
                raise ValueError(f"T_grid_camera must have shape (4,4), got {transform.shape}")
            points, _, diagnostics = preprocess_depth(
                depth, intrinsics, transform, seed=seed, point_count=POINT_COUNT
            )
            model_output = self._run_point_model(points)
            feature_indices = None
            graspable, pose, q_contact = _as_output_tensors(model_output)
            if graspable.shape[0] != len(points) or pose.shape[0] != len(points):
                raise RuntimeError("HGC prediction point count does not match the input feature count")
            candidates, counts = decode_justin_candidates(
                points=torch.from_numpy(np.ascontiguousarray(points)).to(self.device),
                graspable_logits=graspable.to(self.device),
                pose_logits=pose.to(self.device),
                q_contact=q_contact.to(self.device),
                minimum_candidates=CANDIDATE_COUNT,
            )
        candidate_count = len(candidates["quality"])
        if candidate_count < CANDIDATE_COUNT:
            raise HGCGenerationError(
                f"HGC post-filter generation produced {candidate_count} candidates; exactly 100 are required"
            )
        quality = np.asarray(candidates["quality"], dtype=np.float32)
        if not np.isfinite(quality).all():
            raise RuntimeError("HGC decoder returned non-finite quality values")
        order = np.argsort(-quality, kind="stable")[:CANDIDATE_COUNT]
        T_grid_palm = _pose9d_to_transform(np.asarray(candidates["palm_pose"], dtype=np.float32)[order])
        q_contact = np.asarray(candidates["q_contact"], dtype=np.float32)[order]
        quality = quality[order]
        _validate_transforms(T_grid_palm, name="T_grid_palm")
        for name, value in (("q_contact", q_contact),):
            if value.shape != (CANDIDATE_COUNT, 12):
                raise RuntimeError(
                    f"HGC decoder returned invalid {name} shape {value.shape}; "
                    f"expected ({CANDIDATE_COUNT}, 12)"
                )
            if not np.isfinite(value).all():
                raise RuntimeError(f"HGC decoder returned non-finite {name}")
        if quality.shape != (CANDIDATE_COUNT,):
            raise RuntimeError(
                f"HGC decoder returned invalid quality shape {quality.shape}; "
                f"expected ({CANDIDATE_COUNT},)"
            )
        if not np.all(quality[:-1] >= quality[1:]):
            raise RuntimeError("HGC result is not sorted by descending quality")
        diagnostics = {**diagnostics, **counts, "post_filter_count": int(candidate_count), "returned_count": CANDIDATE_COUNT}
        if feature_indices is not None:
            diagnostics["feature_count"] = int(len(feature_indices))
        return success_response(
            T_grid_palm=np.ascontiguousarray(T_grid_palm, dtype=np.float32),
            q_contact=np.ascontiguousarray(q_contact, dtype=np.float32),
            quality=np.ascontiguousarray(quality, dtype=np.float32),
            **{name: np.asarray(value, dtype=np.int64) for name, value in diagnostics.items()},
        )

    def _run_point_model(self, points: np.ndarray) -> Any:
        point_tensor = torch.from_numpy(np.ascontiguousarray(points[None])).to(self.device)
        normalized = point_tensor - point_tensor.mean(dim=1, keepdim=True)
        if hasattr(self.model, "forward_batch"):
            return self.model.forward_batch({
                "point": point_tensor,
                "norm_point": normalized,
            })
        return self.model(point_tensor, normalized.transpose(1, 2).contiguous())

    def _prepare_paper(self, request: Mapping[str, np.ndarray]) -> tuple[torch.Tensor, dict[str, int]]:
        grid = self._required_array(request, "input_grid", np.dtype(np.float32))
        if grid.shape != (2, *GRID_SHAPE):
            raise ValueError(f"input_grid must have shape (2,64,64,64), got {grid.shape}")
        if not np.all((grid == 0.0) | (grid == 1.0)):
            raise ValueError(
                "paper_modified input_grid must contain only binary 0/1 "
                "object/ground occupancy"
            )
        object_count = int(np.count_nonzero(grid[0] > 0.5))
        if object_count == 0:
            raise HGCGenerationError("paper_modified input_grid has no object-occupancy voxel centres")
        model_input = torch.from_numpy(np.ascontiguousarray(grid[None])).to(self.device)
        return model_input, {"object_voxel_count": object_count}

    def _run_paper_grid(
        self, model_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode once and rank quality only at object-occupancy voxels."""
        if not (
            hasattr(self.model, "forward_grid")
            and hasattr(self.model, "_select")
            and hasattr(self.model, "head")
            and hasattr(self.model.head, "forward_selected")
        ):
            raise RuntimeError(
                "paper_modified model must expose forward_grid, _select, and head.forward_selected"
            )
        feature_grid, quality_logits = self.model.forward_grid(model_input)
        if quality_logits.shape != (1, 1, *GRID_SHAPE):
            raise RuntimeError(
                f"paper_modified quality logits must have shape (1,1,64,64,64), got {tuple(quality_logits.shape)}"
            )
        flat = quality_logits[0, 0].reshape(-1)
        object_mask = dilate_object_mask_6(model_input[0, 0] > 0.5).reshape(-1)
        valid_indices = torch.nonzero(object_mask, as_tuple=False).squeeze(1)
        if len(valid_indices) < CANDIDATE_COUNT:
            raise HGCGenerationError(
                "paper_modified HGC needs at least 100 object-occupancy voxels "
                f"but received {len(valid_indices)}"
            )
        valid_quality = flat[valid_indices]
        try:
            local_order = torch.argsort(valid_quality, descending=True, stable=True)
        except TypeError:  # pragma: no cover - older torch compatibility
            local_order = torch.topk(
                valid_quality, k=int(valid_quality.numel()), sorted=True
            ).indices
        order = valid_indices[local_order]
        return feature_grid, quality_logits, order

    def _run_paper_selected(
        self,
        feature_grid: torch.Tensor,
        quality_logits: torch.Tensor,
        order: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        """Evaluate pose/contact for one progressively enlarged quality pool."""
        if order.ndim != 1 or len(order) < CANDIDATE_COUNT:
            raise RuntimeError("paper_modified quality pool must contain at least 100 voxels")
        z = order // (GRID_SHAPE[1] * GRID_SHAPE[2])
        remainder = order % (GRID_SHAPE[1] * GRID_SHAPE[2])
        y = remainder // GRID_SHAPE[2]
        x = remainder % GRID_SHAPE[2]
        indices = torch.stack((z, y, x), dim=-1)
        selected = self.model._select(feature_grid, indices[None])
        orientation, residual, q_contact = self.model.head.forward_selected(selected)
        indices_numpy = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        points = _grid_index_to_point(indices_numpy)
        from paper_modified_hgc.model import PaperModifiedPrediction

        return points, indices_numpy, PaperModifiedPrediction(
            quality_logits, orientation, residual, q_contact
        )
def serve_once(service: HGCService, socket: Any) -> None:
    socket.send(service.handle(socket.recv()))


def serve(service: HGCService, bind: str) -> None:
    import zmq

    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.linger = 0
    socket.bind(str(bind))
    print(
        f"hgc_server bind={bind} device={service.device} arm={service.arm} "
        f"candidate_count={CANDIDATE_COUNT}",
        flush=True,
    )
    try:
        while True:
            serve_once(service, socket)
    except KeyboardInterrupt:
        pass
    finally:
        socket.close(linger=0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve one strict-loaded HGC checkpoint over ZMQ.")
    parser.add_argument("checkpoint", nargs="?", type=Path)
    parser.add_argument("--checkpoint", dest="checkpoint_option", type=Path)
    parser.add_argument("--bind", default="tcp://*:5559")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.checkpoint is None:
        args.checkpoint = args.checkpoint_option
    elif args.checkpoint_option is not None and args.checkpoint.resolve() != args.checkpoint_option.resolve():
        parser.error("checkpoint path was supplied twice with different values")
    if args.checkpoint is None:
        parser.error("a checkpoint path is required")
    delattr(args, "checkpoint_option")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    loaded = load_checkpoint(args.checkpoint, device=args.device)
    serve(
        HGCService(
            loaded.model,
            device=args.device,
            metadata=loaded.metadata,
            arm=loaded.metadata.arm,
        ),
        args.bind,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CANDIDATE_COUNT",
    "GRID_SHAPE",
    "HGCGenerationError",
    "HGCService",
    "POINT_COUNT",
    "main",
    "parse_args",
    "preprocess_depth",
    "serve",
    "serve_once",
]
