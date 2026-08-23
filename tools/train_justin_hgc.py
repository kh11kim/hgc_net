#!/usr/bin/env python3
"""Reproducible production trainer for Issue #59 native Justin-right HGC."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import wandb
import yaml
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from justin_hgc.data import DEFAULT_DERIVED_ROOT, DEFAULT_ROOT, JustinCanonicalDataset
from justin_hgc.bin_pose import BinPoseSpec
from justin_hgc.model import JustinPointNet2
from paper_modified_hgc.data import PaperModifiedCanonicalDataset, collate_paper_modified
from paper_modified_hgc.model import PaperModifiedHGC


DEFAULT_CONFIG = REPO_ROOT / "config" / "issue59_justin_adapter.yaml"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs" / "issue59"
UPSTREAM_TRAIN_DEFAULTS = {
    "epochs": 80,
    "batch_size": 32,
    "workers": 12,
    "optimizer": "Adam",
    "learning_rate": 1e-4,
    "seed": 0,
    "checkpoint_every_epochs": 1,
    "validate_every_epochs": 1,
    "progress_every_batches": 100,
}
PAPER_MODIFIED_TRAIN_DEFAULTS = {
    "epochs": 80,
    "batch_size": 1,
    "workers": 1,
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "weight_decay": 1e-6,
    "seed": 0,
    "checkpoint_every_epochs": 1,
    "validate_every_epochs": 1,
    "progress_every_batches": 100,
}


def _validate_common_config(config: dict[str, Any]) -> None:
    """Reject declared fixed contracts that the implementation cannot vary."""

    if config.get("output_contract") != "q_contact_only":
        raise ValueError("output_contract must be 'q_contact_only'")
    fixed_sections = {
        "pose_supervision": {"direct_axis_min_cosine": 0.9999},
        "hand": {
            "name": "justin_right_hand_simple",
            "dof": 12,
            "template_names": ["finger2", "finger3", "finger4"],
            "gripper_config": "/home/irsl/datasets/dlr/grippers/justin_hand/justin_right_hand_simple.yaml",
        },
        "post_processing": {
            "top_side_world_up": [0, 0, 1],
            "nms_distance_m": 0.03,
            "nms_angle_deg": 30,
        },
    }
    for section, expected in fixed_sections.items():
        declared = config.get(section)
        if declared is not None:
            if not isinstance(declared, dict):
                raise ValueError(f"{section} must be a mapping")
            unsupported = set(declared) - set(expected)
            if unsupported:
                raise ValueError(
                    f"{section} has unsupported settings {sorted(unsupported)!r}"
                )
            for name, value in expected.items():
                if declared.get(name) != value:
                    raise ValueError(
                        f"{section}.{name} must equal the fixed supported value {value!r}"
                    )
    pose_bins = dict(config.get("pose_bins", {}))
    if float(pose_bins.get("depth_base_cm", 0.0)) != 0.0:
        raise ValueError("pose_bins.depth_base_cm must be 0 for Justin direct-depth targets")


def _validate_paper_config(config: dict[str, Any]) -> None:
    """Validate declared paper-arm representation, encoder, and split seed."""

    if config.get("arm") != "paper_modified":
        return
    data_config = dict(config.get("data", {}))
    model_config = dict(config.get("model", {}))
    representation = data_config.get("representation", "gt_full_occupancy_ground")
    if representation != "gt_full_occupancy_ground":
        raise ValueError(
            "paper_modified data.representation must be 'gt_full_occupancy_ground', "
            f"got {representation!r}"
        )
    encoder = model_config.get("encoder", "3d_fpn")
    if encoder != "3d_fpn":
        raise ValueError(
            "paper_modified model.encoder must be '3d_fpn', "
            f"got {encoder!r}"
        )
    split_seed = data_config.get("split_seed")
    if split_seed is not None and int(split_seed) != int(config["seed"]):
        raise ValueError(
            "paper_modified data.split_seed must equal training seed "
            f"({config['seed']}), got {split_seed}"
        )


def load_training_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read either arm's config while preserving upstream defaults."""
    with path.open(encoding="utf-8") as source:
        payload = yaml.safe_load(source) or {}
    arm = str(payload.get("arm", "upstream_faithful"))
    arm = {
        "upstream": "upstream_faithful",
        "paper_modified_hgc": "paper_modified",
    }.get(arm, arm)
    if arm not in {"upstream_faithful", "paper_modified"}:
        raise ValueError(f"{path}: unsupported HGC arm {arm!r}")
    defaults = PAPER_MODIFIED_TRAIN_DEFAULTS if arm == "paper_modified" else UPSTREAM_TRAIN_DEFAULTS
    training = dict(defaults)
    training.update(payload.get("training", {}))
    missing = set(defaults) - set(training)
    if missing:
        raise ValueError(f"{path}: missing training settings {sorted(missing)}")
    if training["optimizer"] not in {"Adam", "AdamW"}:
        raise ValueError(f"{path}: optimizer must be Adam or AdamW")
    if arm == "upstream_faithful" and training["optimizer"] != "Adam":
        raise ValueError("upstream_faithful preserves upstream Adam optimizer")
    training["arm"] = arm
    training["data"] = dict(payload.get("data", {}))
    training["model"] = dict(payload.get("model", {}))
    for section in (
        "pose_bins",
        "pose_supervision",
        "hand",
        "post_processing",
        "output_contract",
    ):
        if section in payload:
            value = payload[section]
            training[section] = dict(value) if isinstance(value, dict) else value
    _validate_common_config(training)
    _validate_paper_config(training)
    return training


def prepare_run_dir(run_dir: Path, *, resume: bool) -> None:
    if run_dir.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite existing run directory: {run_dir}")
        return
    if resume:
        raise FileNotFoundError(f"resume run directory does not exist: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)


def create_timestamped_run_dir(run_root: Path, label: str) -> Path:
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / f"{timestamp}_{label}"
    prepare_run_dir(run_dir, resume=False)
    return run_dir


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    metrics: dict[str, float],
    best_val: float = float("inf"),
    loader_generator: torch.Generator | None = None,
) -> None:
    """Persist model, optimizer, counters, and deterministic RNG state together."""
    model_device = next(model.parameters()).device
    _atomic_torch_save(
        {
            "format": "issue59_justin_hgc_checkpoint_v1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "config": config,
            "metrics": metrics,
            "best_val": float(best_val),
            "loader_generator_state": (
                loader_generator.get_state() if loader_generator is not None else None
            ),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if model_device.type == "cuda" else None,
            },
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    expected_config: dict[str, Any] | None = None,
    loader_generator: torch.Generator | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != "issue59_justin_hgc_checkpoint_v1":
        raise ValueError(f"{path}: unsupported checkpoint format")
    if expected_config is not None:
        saved_config = payload.get("config")
        if not isinstance(saved_config, dict):
            raise ValueError(f"{path}: checkpoint config is missing")
        if _resume_contract(saved_config) != _resume_contract(expected_config):
            raise ValueError(f"{path}: checkpoint config does not match the active run contract")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    rng = payload.get("rng", {})
    if rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if device.type == "cuda" and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
    generator_state = payload.get("loader_generator_state")
    if loader_generator is not None and generator_state is not None:
        loader_generator.set_state(generator_state.cpu())
    return payload


def _resume_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Return settings that must remain identical across a resumed run."""

    excluded = {
        "epochs",
        "checkpoint_every_epochs",
        "validate_every_epochs",
        "progress_every_batches",
    }
    return {key: value for key, value in config.items() if key not in excluded}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def _finite_metrics(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in losses.items()}


def _mean_metrics(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot average zero batches")
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _graspability_counts(
    graspable_logits: torch.Tensor, labels: torch.Tensor
) -> tuple[int, int, int]:
    prediction = torch.argmax(graspable_logits, dim=2)
    supervised = labels >= 0
    positive = labels == 1
    predicted_positive = prediction == 1
    true_positive = int((supervised & positive & predicted_positive).sum().item())
    false_positive = int((supervised & ~positive & predicted_positive).sum().item())
    false_negative = int((supervised & positive & ~predicted_positive).sum().item())
    return true_positive, false_positive, false_negative


def _dense_graspability_counts(
    quality_logits: torch.Tensor, quality_target: torch.Tensor
) -> tuple[int, int, int]:
    """Compute dense quality counts for the paper-modified voxel arm."""

    if quality_logits.shape != quality_target.shape:
        raise ValueError("dense quality logits and target shapes must match")
    predicted_positive = quality_logits >= 0.0
    positive = quality_target > 0.5
    true_positive = int((positive & predicted_positive).sum().item())
    false_positive = int((~positive & predicted_positive).sum().item())
    false_negative = int((positive & ~predicted_positive).sum().item())
    return true_positive, false_positive, false_negative


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2.0 * true_positive / denominator


def run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    limit_batches: int | None,
    progress_label: str,
    progress_every_batches: int,
) -> tuple[dict[str, float], int]:
    training = optimizer is not None
    model.train(training)
    rows: list[dict[str, float]] = []
    true_positive = 0
    false_positive = 0
    false_negative = 0
    tilted_excluded_count = 0
    steps = 0
    started = time.monotonic()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if limit_batches is not None and batch_index >= limit_batches:
                break
            batch = _to_device(batch, device)
            if getattr(model, "paper_modified", False):
                prediction = model.forward_batch(batch)  # type: ignore[attr-defined]
                counts = _dense_graspability_counts(
                    prediction.quality_logits, batch["quality_target"]
                )
                losses = model.head.loss(prediction, batch)
            else:
                if hasattr(model, "forward_batch"):
                    prediction = model.forward_batch(batch)  # type: ignore[attr-defined]
                else:
                    prediction = model(
                        batch["point"], batch["norm_point"].transpose(1, 2).contiguous()
                    )
                counts = _graspability_counts(prediction[0], batch["template_graspable"])
                losses = model.head.loss(prediction, batch)
            excluded = batch.get("tilted_excluded_count")
            if excluded is not None:
                tilted_excluded_count += int(excluded.sum().item())
            batch_true_positive, batch_false_positive, batch_false_negative = counts
            true_positive += batch_true_positive
            false_positive += batch_false_positive
            false_negative += batch_false_negative
            loss = losses["total_loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite total loss at batch {batch_index}: {loss.item()}")
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            rows.append(_finite_metrics(losses))
            steps += 1
            if steps % progress_every_batches == 0 or (
                limit_batches is not None and steps == limit_batches
            ):
                elapsed = time.monotonic() - started
                print(
                    json.dumps(
                        {
                            "progress": progress_label,
                            "batch": steps,
                            "available_batches": len(loader),
                            "elapsed_seconds": elapsed,
                            "seconds_per_batch": elapsed / steps,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    metrics = _mean_metrics(rows)
    metrics["graspability_f1"] = _f1(true_positive, false_positive, false_negative)
    # Both arms record every excluded 10-degree fallback row.  Keep the count
    # in epoch metrics so provenance can audit the geometry mask without
    # rereading the immutable source archives.
    metrics["tilted_excluded_count"] = float(tilted_excluded_count)
    return metrics, steps


def _git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def write_provenance(
    run_dir: Path,
    *,
    config_path: Path,
    config: dict[str, Any],
    dataset_root: Path,
    derived_root: Path | None,
    device: torch.device,
    command: list[str],
    train_count: int,
    val_count: int,
) -> None:
    provenance: dict[str, Any] = {
        "format": "issue59_justin_hgc_run_v1",
        "started_utc": dt.datetime.now(dt.UTC).isoformat(),
        "command": command,
        "git_commit": _git_text("rev-parse", "HEAD"),
        "git_status_porcelain": _git_text("status", "--porcelain"),
        "host": socket.gethostname(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dataset_root": str(dataset_root),
        "derived_root": str(derived_root) if derived_root is not None else None,
        "train_views": train_count,
        "val_views": val_count,
        "training_config": config,
    }
    if config.get("arm") == "paper_modified":
        provenance["paper_positive_geometry_filter"] = {
            "approach_axis_cosine_min": 0.9999,
            "excluded_count_metric": [
                "train/tilted_excluded_count",
                "val/tilted_excluded_count",
            ],
            "source_immutable": True,
        }
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    # None means "use the selected arm's config".  For the default upstream
    # config this resolves to the historical DEFAULT_ROOT/DEFAULT_DERIVED_ROOT
    # values, preserving the existing CLI behavior.
    parser.add_argument("--root", type=Path)
    parser.add_argument("--derived-root", type=Path)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-name", default="justin_hgc")
    parser.add_argument("--resume", type=Path, help="path to an existing last.pt checkpoint")
    parser.add_argument("--epochs", type=int, help="total epoch count, not additional epochs")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint-every-epochs", type=int)
    parser.add_argument("--validate-every-epochs", type=int)
    parser.add_argument("--limit-train-batches", type=int, help="smoke-only cap; omit for production")
    parser.add_argument("--limit-val-batches", type=int, help="smoke-only cap; omit for production")
    parser.add_argument(
        "--train-sample-id",
        action="append",
        help="smoke/regression only: restrict train data to an explicit canonical sample ID (repeatable)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-project", default="hgc-net-issue59")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def build_datasets(
    config: dict[str, Any],
    *,
    root: Path,
    derived_root: Path | None,
    train_sample_ids: Iterable[str] | None = None,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Construct both splits for the selected arm behind one trainer seam."""

    _validate_common_config(config)
    arm = config["arm"]
    data_config = dict(config.get("data", {}))
    if arm == "paper_modified":
        _validate_paper_config(config)
        kwargs = {
            "num_grasps": int(data_config.get("num_grasps", 100)),
            "grid_size": int(data_config.get("grid_size", 64)),
            "grid_edge_m": float(data_config.get("grid_edge_m", 0.5)),
            "seed": int(config["seed"]),
            "gripper_config": data_config.get("gripper_config"),
        }
        if data_config.get("negative_voxel_fraction") is not None:
            kwargs["negative_voxel_count"] = None
            kwargs["negative_voxel_fraction"] = float(data_config["negative_voxel_fraction"])
        else:
            kwargs["negative_voxel_count"] = int(data_config.get("negative_voxel_count", 100))
        if kwargs["gripper_config"] is None:
            kwargs.pop("gripper_config")
        train_data = PaperModifiedCanonicalDataset(
            root, split="train", sample_ids=train_sample_ids, **kwargs
        )
        val_data = PaperModifiedCanonicalDataset(root, split="val", **kwargs)
        return train_data, val_data
    train_data = JustinCanonicalDataset(
        root,
        derived_root=derived_root,
        split="train",
        point_count=int(data_config.get("point_count", 25_000)),
        sample_ids=train_sample_ids,
    )
    val_data = JustinCanonicalDataset(
        root,
        derived_root=derived_root,
        split="val",
        point_count=int(data_config.get("point_count", 25_000)),
    )
    return train_data, val_data


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    """Instantiate the arm selected by config."""

    _validate_common_config(config)

    if config["arm"] == "paper_modified":
        _validate_paper_config(config)
        model_config = dict(config.get("model", {}))
        return PaperModifiedHGC(
            in_channels=int(model_config.get("in_channels", 2)),
            encoder_channels=tuple(model_config.get("channels", (32, 64, 128, 256))),
            encoder_scales=tuple(model_config.get("scales", (2, 2, 2, 4))),
            encoder_out_channels=int(model_config.get("out_channels", 128)),
            align_corners=bool(model_config.get("align_corners", True)),
        )
    pose_bins = dict(config.get("pose_bins", {}))
    bin_spec = BinPoseSpec(
        **{
            name: float(value)
            for name, value in pose_bins.items()
            if name
            in {
                "depth_scope_cm",
                "depth_bin_cm",
                "azimuth_scope_deg",
                "azimuth_bin_deg",
                "elevation_scope_deg",
                "elevation_bin_deg",
                "grasp_angle_scope_deg",
                "grasp_angle_bin_deg",
            }
        }
    )
    return JustinPointNet2(bin_spec=bin_spec)


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    """Build the arm-specific optimizer without changing upstream Adam."""

    kwargs = {"lr": float(config["learning_rate"]), "betas": (0.9, 0.999), "eps": 1e-8}
    if config["optimizer"] == "AdamW":
        kwargs["weight_decay"] = float(config.get("weight_decay", 1e-6))
        return torch.optim.AdamW(model.parameters(), **kwargs)
    return torch.optim.Adam(model.parameters(), **kwargs)


def main() -> int:
    args = _parse_args()
    config = load_training_config(args.config)
    for key, value in {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "checkpoint_every_epochs": args.checkpoint_every_epochs,
        "validate_every_epochs": args.validate_every_epochs,
    }.items():
        if value is not None:
            config[key] = value
    _validate_paper_config(config)
    if config["epochs"] <= 0 or config["batch_size"] <= 0 or config["workers"] < 0:
        raise ValueError("epochs and batch size must be positive; workers must be non-negative")
    data_config = dict(config.get("data", {}))
    dataset_root = Path(
        args.root
        or data_config.get(
            "root",
            DEFAULT_ROOT,
        )
    )
    if args.derived_root is not None:
        derived_root = args.derived_root
    elif config["arm"] == "paper_modified":
        derived_root = None
    else:
        derived_root = Path(data_config.get("derived_root", DEFAULT_DERIVED_ROOT))
    config["data"] = {
        **data_config,
        "root": str(dataset_root.resolve()),
        "derived_root": str(derived_root.resolve()) if derived_root is not None else None,
    }
    if args.resume is not None:
        run_dir = args.resume.resolve().parent
        prepare_run_dir(run_dir, resume=True)
    else:
        run_dir = create_timestamped_run_dir(args.run_root, args.run_name)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(int(config["seed"]))
    train_data, val_data = build_datasets(
        config,
        root=dataset_root,
        derived_root=derived_root,
        train_sample_ids=args.train_sample_id,
    )
    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        train_data,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["workers"]),
        pin_memory=device.type == "cuda",
        worker_init_fn=_worker_init,
        generator=generator,
        collate_fn=collate_paper_modified if config["arm"] == "paper_modified" else None,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["workers"]),
        pin_memory=device.type == "cuda",
        worker_init_fn=_worker_init,
        collate_fn=collate_paper_modified if config["arm"] == "paper_modified" else None,
    )
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    if args.resume is None:
        write_provenance(
            run_dir,
            config_path=args.config,
            config=config,
            dataset_root=dataset_root,
            derived_root=derived_root,
            device=device,
            command=sys.argv,
            train_count=len(train_data),
            val_count=len(val_data),
        )
    else:
        state = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            device=device,
            expected_config=config,
            loader_generator=generator,
        )
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best_val = float(state.get("best_val", best_val))
    metrics_path = run_dir / "metrics.jsonl"
    wandb_run = wandb.init(
        project=args.wandb_project,
        name=run_dir.name,
        id=run_dir.name,
        resume="allow",
        mode=args.wandb_mode,
        config={
            **config,
            "dataset_root": str(dataset_root),
            "derived_root": str(derived_root) if derived_root is not None else None,
            "git_commit": _git_text("rev-parse", "HEAD"),
        },
    )
    for epoch in range(start_epoch, int(config["epochs"])):
        started = time.monotonic()
        train_metrics, train_steps = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            limit_batches=args.limit_train_batches,
            progress_label=f"train/epoch-{epoch}",
            progress_every_batches=int(config["progress_every_batches"]),
        )
        global_step += train_steps
        metrics = {f"train/{key}": value for key, value in train_metrics.items()}
        if epoch % int(config["validate_every_epochs"]) == 0:
            val_metrics, _ = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                limit_batches=args.limit_val_batches,
                progress_label=f"val/epoch-{epoch}",
                progress_every_batches=int(config["progress_every_batches"]),
            )
            metrics.update({f"val/{key}": value for key, value in val_metrics.items()})
        metrics.update({"epoch": epoch, "global_step": global_step, "epoch_seconds": time.monotonic() - started})
        wandb_metrics = {
            "train/total_loss": metrics["train/total_loss"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        for key in ("val/total_loss", "val/graspability_f1"):
            if key in metrics:
                wandb_metrics[key] = metrics[key]
        wandb_run.log(wandb_metrics, step=epoch)
        with metrics_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(metrics, sort_keys=True) + "\n")
        print(json.dumps(metrics, sort_keys=True), flush=True)
        checkpoint_metrics = {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
        val_loss = metrics.get("val/total_loss")
        is_best = val_loss is not None and float(val_loss) < best_val
        if is_best:
            best_val = float(val_loss)
        if epoch % int(config["checkpoint_every_epochs"]) == 0:
            save_checkpoint(run_dir / f"epoch-{epoch:03d}.pt", model=model, optimizer=optimizer, epoch=epoch, global_step=global_step, config=config, metrics=checkpoint_metrics, best_val=best_val, loader_generator=generator)
        save_checkpoint(run_dir / "last.pt", model=model, optimizer=optimizer, epoch=epoch, global_step=global_step, config=config, metrics=checkpoint_metrics, best_val=best_val, loader_generator=generator)
        if is_best:
            save_checkpoint(run_dir / "best.pt", model=model, optimizer=optimizer, epoch=epoch, global_step=global_step, config=config, metrics=checkpoint_metrics, best_val=best_val, loader_generator=generator)
            wandb_run.summary["best/val_total_loss"] = best_val
            wandb_run.summary["best/epoch"] = epoch
    wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
