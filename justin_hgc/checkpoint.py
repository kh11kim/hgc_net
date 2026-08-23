"""Checkpoint metadata, arm dispatch, and strict model loading for HGC.

Issue #59 training checkpoints contain optimizer and RNG state.  Those fields
are Python objects, so loading an existing v1 checkpoint with
``weights_only=True`` is intentionally unsupported.  This module makes the
trust boundary explicit: the checkpoint must be a locally trusted artifact and
is read once with ``weights_only=False``; the model weights themselves are
then loaded with ``strict=True``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_FORMAT = "issue59_justin_hgc_checkpoint_v1"
LEGACY_UPSTREAM_ARM = "upstream_faithful"
PAPER_MODIFIED_ARM = "paper_modified"
TRUSTED_PICKLE_LOAD = "torch.load(weights_only=False) is required for trusted v1 checkpoints containing optimizer/RNG objects"


class CheckpointLoadError(RuntimeError):
    """Raised when a checkpoint cannot be safely dispatched and strict-loaded."""


@dataclass(frozen=True)
class CheckpointMetadata:
    path: Path
    sha256: str
    format: str
    arm: str
    config: Mapping[str, Any]
    epoch: int | None
    global_step: int | None
    source_commit: str
    trusted_pickle: bool = True

    @property
    def input_representation(self) -> str:
        if self.arm == PAPER_MODIFIED_ARM:
            return "gt_full_occupancy_ground"
        return "depth_point_cloud"


@dataclass(frozen=True)
class LoadedHGCCheckpoint:
    metadata: CheckpointMetadata
    model: torch.nn.Module


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer_metadata(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise CheckpointLoadError(f"checkpoint {name} must be an integer or null")
    return int(value)


def detect_checkpoint_arm(payload: Mapping[str, Any]) -> str:
    """Resolve the arm without guessing across ambiguous checkpoint formats.

    A missing ``config.arm`` is accepted only for the known Issue #59 v1
    format and only when there is no non-empty model architecture declaration.
    This is the documented legacy upstream checkpoint shape.  Paper-modified
    checkpoints must explicitly declare ``arm: paper_modified``.
    """

    checkpoint_format = payload.get("format")
    if checkpoint_format != CHECKPOINT_FORMAT:
        if checkpoint_format is None:
            raise CheckpointLoadError("checkpoint format is missing; refusing arm inference")
        raise CheckpointLoadError(
            f"unsupported checkpoint format {checkpoint_format!r}; expected {CHECKPOINT_FORMAT!r}"
        )
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise CheckpointLoadError("checkpoint config metadata is missing or not a mapping")
    if "arm" not in config:
        model_config = config.get("model")
        architecture = config.get("architecture")
        if (isinstance(model_config, Mapping) and len(model_config)) or architecture not in (None, ""):
            raise CheckpointLoadError(
                "checkpoint config.arm is missing but architecture metadata is present; "
                "refusing ambiguous arm inference"
            )
        return LEGACY_UPSTREAM_ARM
    raw_arm = config["arm"]
    if not isinstance(raw_arm, str):
        raise CheckpointLoadError("checkpoint config.arm must be a string")
    arm = {"upstream": LEGACY_UPSTREAM_ARM, "upstream_faithful": LEGACY_UPSTREAM_ARM}.get(
        raw_arm, raw_arm
    )
    if arm not in {LEGACY_UPSTREAM_ARM, PAPER_MODIFIED_ARM}:
        raise CheckpointLoadError(f"unsupported HGC checkpoint arm {raw_arm!r}")
    if arm == PAPER_MODIFIED_ARM:
        model_config = config.get("model")
        if not isinstance(model_config, Mapping) or not model_config:
            raise CheckpointLoadError(
                "paper_modified checkpoint must declare a non-empty config.model"
            )
    return arm


def checkpoint_metadata(
    path: str | Path, payload: Mapping[str, Any]
) -> CheckpointMetadata:
    path = Path(path).resolve()
    arm = detect_checkpoint_arm(payload)
    config = payload["config"]
    assert isinstance(config, Mapping)
    source_commit = config.get("source_commit", config.get("git_commit", "unknown"))
    if not isinstance(source_commit, str) or not source_commit:
        source_commit = "unknown"
    return CheckpointMetadata(
        path=path,
        sha256=sha256_file(path),
        format=CHECKPOINT_FORMAT,
        arm=arm,
        config=config,
        epoch=_integer_metadata(payload.get("epoch"), "epoch"),
        global_step=_integer_metadata(payload.get("global_step"), "global_step"),
        source_commit=source_commit,
    )


def load_checkpoint_payload(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> Mapping[str, Any]:
    """Load a trusted v1 payload with the required explicit pickle policy."""

    path = Path(path)
    if not path.is_file():
        raise CheckpointLoadError(f"checkpoint does not exist: {path}")
    try:
        # Do not replace this with weights_only=True: v1 stores optimizer/RNG
        # objects and the server's trust decision must remain visible.
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as error:
        raise CheckpointLoadError(
            f"failed trusted checkpoint load for {path}: {TRUSTED_PICKLE_LOAD}; {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise CheckpointLoadError("checkpoint payload must be a mapping")
    return payload


def _bin_spec_from_config(config: Mapping[str, Any]):
    from .bin_pose import BinPoseSpec

    values = config.get("pose_bins")
    if not isinstance(values, Mapping):
        return None
    aliases = {
        "depth_scope_cm": "depth_scope_cm",
        "depth_bin_cm": "depth_bin_cm",
        "azimuth_scope_deg": "azimuth_scope_deg",
        "azimuth_bin_deg": "azimuth_bin_deg",
        "elevation_scope_deg": "elevation_scope_deg",
        "elevation_bin_deg": "elevation_bin_deg",
        "grasp_angle_scope_deg": "grasp_angle_scope_deg",
        "grasp_angle_bin_deg": "grasp_angle_bin_deg",
    }
    kwargs = {target: values[source] for source, target in aliases.items() if source in values}
    return BinPoseSpec(**kwargs)


def build_model_from_metadata(metadata: CheckpointMetadata) -> torch.nn.Module:
    """Construct the model solely from checkpoint arm/config metadata."""

    config = metadata.config
    bin_spec = _bin_spec_from_config(config)
    if metadata.arm == LEGACY_UPSTREAM_ARM:
        from .model import JustinPointNet2

        kwargs: dict[str, Any] = {}
        model_config = config.get("model")
        if isinstance(model_config, Mapping):
            if "num_templates" in model_config:
                kwargs["num_templates"] = int(model_config["num_templates"])
            if "bin_spec" in model_config:
                raise CheckpointLoadError(
                    "model.bin_spec must be represented by scalar pose_bins metadata"
                )
        if bin_spec is not None:
            kwargs["bin_spec"] = bin_spec
        return JustinPointNet2(**kwargs)

    from paper_modified_hgc.model import PaperModifiedHGC

    model_config = config.get("model")
    if not isinstance(model_config, Mapping) or not model_config:
        raise CheckpointLoadError("paper_modified checkpoint is missing config.model")
    encoder = model_config.get("encoder", "3d_fpn")
    if encoder != "3d_fpn":
        raise CheckpointLoadError(f"unsupported paper_modified encoder {encoder!r}")
    kwargs = {
        "in_channels": int(model_config.get("in_channels", 2)),
        "encoder_channels": tuple(
            model_config.get("channels", model_config.get("encoder_channels", (32, 64, 128, 256)))
        ),
        "encoder_scales": tuple(
            model_config.get("scales", model_config.get("encoder_scales", (2, 2, 2, 4)))
        ),
        "encoder_out_channels": int(
            model_config.get("out_channels", model_config.get("encoder_out_channels", 128))
        ),
        "align_corners": bool(model_config.get("align_corners", True)),
    }
    return PaperModifiedHGC(**kwargs)


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    model_factory: Callable[[CheckpointMetadata], torch.nn.Module] | None = None,
) -> LoadedHGCCheckpoint:
    """Load one checkpoint and strict-load its model weights."""

    device = torch.device(device)
    payload = load_checkpoint_payload(path, map_location=device)
    metadata = checkpoint_metadata(path, payload)
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise CheckpointLoadError("checkpoint model state is missing or not a mapping")
    try:
        model = model_factory(metadata) if model_factory is not None else build_model_from_metadata(metadata)
    except CheckpointLoadError:
        raise
    except Exception as error:
        raise CheckpointLoadError(
            f"failed to construct {metadata.arm} model from checkpoint metadata: {error}"
        ) from error
    try:
        model.load_state_dict(state, strict=True)
    except Exception as error:
        raise CheckpointLoadError(
            f"strict model load failed for {metadata.path}: {error}"
        ) from error
    model.to(device).eval()
    return LoadedHGCCheckpoint(metadata=metadata, model=model)


# Descriptive aliases keep the loader seam discoverable to callers that name
# the service explicitly while retaining the short trainer-compatible name.
load_hgc_checkpoint = load_checkpoint
detect_arm = detect_checkpoint_arm


__all__ = [
    "CHECKPOINT_FORMAT",
    "CheckpointLoadError",
    "CheckpointMetadata",
    "LEGACY_UPSTREAM_ARM",
    "LoadedHGCCheckpoint",
    "PAPER_MODIFIED_ARM",
    "TRUSTED_PICKLE_LOAD",
    "build_model_from_metadata",
    "checkpoint_metadata",
    "detect_checkpoint_arm",
    "detect_arm",
    "load_checkpoint",
    "load_hgc_checkpoint",
    "load_checkpoint_payload",
    "sha256_file",
]
