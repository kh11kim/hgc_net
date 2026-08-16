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
import yaml
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from justin_hgc.data import DEFAULT_ROOT, JustinCanonicalDataset
from justin_hgc.model import JustinPointNet2


DEFAULT_CONFIG = REPO_ROOT / "config" / "issue59_justin_adapter.yaml"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs" / "issue59"
UPSTREAM_TRAIN_DEFAULTS = {
    "epochs": 80,
    "batch_size": 1,
    "workers": 1,
    "optimizer": "Adam",
    "learning_rate": 1e-4,
    "seed": 0,
    "checkpoint_every_epochs": 1,
    "validate_every_epochs": 1,
}


def load_training_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read the Justin config and validate the upstream-derived train defaults."""
    with path.open(encoding="utf-8") as source:
        payload = yaml.safe_load(source)
    training = dict(UPSTREAM_TRAIN_DEFAULTS)
    training.update(payload.get("training", {}))
    missing = set(UPSTREAM_TRAIN_DEFAULTS) - set(training)
    if missing:
        raise ValueError(f"{path}: missing training settings {sorted(missing)}")
    if training["optimizer"] != "Adam":
        raise ValueError("Issue #59 preserves upstream Adam optimizer")
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
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != "issue59_justin_hgc_checkpoint_v1":
        raise ValueError(f"{path}: unsupported checkpoint format")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    rng = payload.get("rng", {})
    if rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if device.type == "cuda" and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
    return payload


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


def run_epoch(
    *,
    model: JustinPointNet2,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    limit_batches: int | None,
) -> tuple[dict[str, float], int]:
    training = optimizer is not None
    model.train(training)
    rows: list[dict[str, float]] = []
    steps = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if limit_batches is not None and batch_index >= limit_batches:
                break
            batch = _to_device(batch, device)
            prediction = model(batch["point"], batch["norm_point"].transpose(1, 2).contiguous())
            losses = model.head.loss(prediction, batch)
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
    return _mean_metrics(rows), steps


def _git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def write_provenance(
    run_dir: Path,
    *,
    config_path: Path,
    config: dict[str, Any],
    dataset_root: Path,
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
        "train_views": train_count,
        "val_views": val_count,
        "training_config": config,
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
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
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


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
    if config["epochs"] <= 0 or config["batch_size"] <= 0 or config["workers"] < 0:
        raise ValueError("epochs and batch size must be positive; workers must be non-negative")
    if args.resume is not None:
        run_dir = args.resume.resolve().parent
        prepare_run_dir(run_dir, resume=True)
    else:
        run_dir = create_timestamped_run_dir(args.run_root, args.run_name)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(int(config["seed"]))
    train_data = JustinCanonicalDataset(args.root, split="train")
    val_data = JustinCanonicalDataset(args.root, split="val")
    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        train_data,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["workers"]),
        pin_memory=device.type == "cuda",
        worker_init_fn=_worker_init,
        generator=generator,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["workers"]),
        pin_memory=device.type == "cuda",
        worker_init_fn=_worker_init,
    )
    model = JustinPointNet2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), betas=(0.9, 0.999), eps=1e-8)
    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    if args.resume is None:
        write_provenance(
            run_dir,
            config_path=args.config,
            config=config,
            dataset_root=args.root,
            device=device,
            command=sys.argv,
            train_count=len(train_data),
            val_count=len(val_data),
        )
    else:
        state = load_checkpoint(args.resume, model=model, optimizer=optimizer, device=device)
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best_val = float(state.get("metrics", {}).get("val/total_loss", best_val))
    metrics_path = run_dir / "metrics.jsonl"
    for epoch in range(start_epoch, int(config["epochs"])):
        started = time.monotonic()
        train_metrics, train_steps = run_epoch(
            model=model, loader=train_loader, optimizer=optimizer, device=device, limit_batches=args.limit_train_batches
        )
        global_step += train_steps
        metrics = {f"train/{key}": value for key, value in train_metrics.items()}
        if epoch % int(config["validate_every_epochs"]) == 0:
            val_metrics, _ = run_epoch(
                model=model, loader=val_loader, optimizer=None, device=device, limit_batches=args.limit_val_batches
            )
            metrics.update({f"val/{key}": value for key, value in val_metrics.items()})
        metrics.update({"epoch": epoch, "global_step": global_step, "epoch_seconds": time.monotonic() - started})
        with metrics_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(metrics, sort_keys=True) + "\n")
        print(json.dumps(metrics, sort_keys=True), flush=True)
        checkpoint_metrics = {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
        if epoch % int(config["checkpoint_every_epochs"]) == 0:
            save_checkpoint(run_dir / f"epoch-{epoch:03d}.pt", model=model, optimizer=optimizer, epoch=epoch, global_step=global_step, config=config, metrics=checkpoint_metrics)
        save_checkpoint(run_dir / "last.pt", model=model, optimizer=optimizer, epoch=epoch, global_step=global_step, config=config, metrics=checkpoint_metrics)
        val_loss = metrics.get("val/total_loss")
        if val_loss is not None and float(val_loss) < best_val:
            best_val = float(val_loss)
            save_checkpoint(run_dir / "best.pt", model=model, optimizer=optimizer, epoch=epoch, global_step=global_step, config=config, metrics=checkpoint_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
