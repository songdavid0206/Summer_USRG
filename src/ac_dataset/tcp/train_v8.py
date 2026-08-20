"""V8 single-session convergence training with TensorBoard scalars."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from .config import lap_phase_metadata
from .profile_v5 import sha256_file
from .train_v5 import _tensor_stats, compute_losses


def _seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extra_control_metrics(pred, batch):
    return {
        "steer_mae": (pred["controls"][:, 0].float() - batch["controls"][:, 0]).abs().mean(),
        "throttle_mae": (pred["controls"][:, 1].float() - batch["controls"][:, 1]).abs().mean(),
        "brake_mae": (pred["controls"][:, 2].float() - batch["controls"][:, 2]).abs().mean(),
    }


def _mean(sums: dict[str, float], batches: int) -> dict[str, float]:
    return {name: value / max(1, batches) for name, value in sums.items()}


def _accumulate(sums: dict[str, float], metrics: dict) -> None:
    for name, value in metrics.items():
        sums[name] = sums.get(name, 0.0) + float(value.detach())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V8 single-session 5/2 convergence run")
    parser.add_argument("--prepared", type=Path, default=Path("training/tcp_lagunaseca_v8"))
    parser.add_argument("--output", type=Path,
                        default=Path("training/tcp_lagunaseca_v8/checkpoints"))
    parser.add_argument("--tensorboard-dir", type=Path,
                        default=Path("training/tcp_lagunaseca_v8/tensorboard"))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imagenet-pretrained", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr-patience", type=int, default=25)
    parser.add_argument("--lr-factor", type=float, default=.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="0 disables; 1000 epochs is then a maximum, best.pt remains safest")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="save an immutable epoch snapshot at this interval; 0 disables")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError(
            "V8 requires TensorBoard. Install with: python -m pip install tensorboard") from error
    from .data_v5 import TCPDatasetV5
    from .model_v5 import build_model_v5

    _seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schema = json.loads((args.prepared / "schema.json").read_text(encoding="utf-8"))
    experiment = schema.get("experiment", {})
    if int(experiment.get("version", 0)) != 8:
        raise RuntimeError("prepared directory is not the audited V8 experiment")
    split = json.loads((args.prepared / "split.json").read_text(encoding="utf-8"))
    if split.get("train_laps") != [1, 2, 3, 4, 5] or split.get("val_laps") != [6, 7]:
        raise RuntimeError("V8 split must be train laps 1-5 and val laps 6-7")
    for name, expected in schema["artifacts"].items():
        if sha256_file(args.prepared / name) != expected:
            raise RuntimeError(f"V8 artifact hash mismatch: {name}")

    train_set = TCPDatasetV5(args.prepared, "train", augment=True)
    val_set = TCPDatasetV5(args.prepared, "val", augment=False)
    if len(train_set) != 13918 or len(val_set) != 5503:
        raise RuntimeError(f"unexpected V8 sample counts: {len(train_set)}/{len(val_set)}")
    worker_options = ({"persistent_workers": True, "prefetch_factor": 2}
                      if args.workers else {})
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda", **worker_options)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda", **worker_options)

    normalization = json.loads(
        (args.prepared / "normalization.json").read_text(encoding="utf-8"))
    stats = _tensor_stats(normalization, device)
    model = build_model_v5(args.imagenet_pretrained and args.resume is None).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience,
        min_lr=args.min_lr)
    amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=32768.0)
    args.output.mkdir(parents=True, exist_ok=True)
    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.tensorboard_dir))
    history_path = args.output / "history.json"
    history: list[dict] = []
    best, stale, start_epoch = float("inf"), 0, 1

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("training_recipe", {}).get("version") != 8:
            raise RuntimeError("--resume checkpoint is not V8")
        if checkpoint.get("artifacts") != schema["artifacts"]:
            raise RuntimeError("V8 resume/prepared artifact mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if amp and checkpoint.get("grad_scaler") is not None:
            scaler.load_state_dict(checkpoint["grad_scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        passed = [row for row in history if "val" in row]
        if passed:
            best = min(float(row["val"]["total"]) for row in passed)
            best_epoch = max(int(row["epoch"]) for row in passed
                             if float(row["val"]["total"]) == best)
            stale = int(checkpoint["epoch"]) - best_epoch

    print(f"[v8] train_laps=1-5 val_laps=6-7 standing_lap=train_only ", end="", flush=True)
    print(f"train={len(train_set)} val={len(val_set)} batches={len(train_loader)}+"
          f"{len(val_loader)} params={parameter_count:,} device={device} amp={amp}", flush=True)
    print(f"[tensorboard] logdir={args.tensorboard_dir.resolve()}", flush=True)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            train_sums: dict[str, float] = {}
            train_batches = overflow = 0
            started = time.perf_counter()
            limit = min(len(train_loader), args.max_train_batches) if args.max_train_batches else len(train_loader)
            marks = {max(1, round(limit * fraction)) for fraction in (.25, .5, .75, 1.0)}
            for number, batch in enumerate(train_loader, 1):
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                    pred = model(batch["image"], batch["ego"], batch["racing_line"],
                                 batch["command"], batch["target_speed"])
                    loss, metrics = compute_losses(pred, batch, stats, planner_only=False)
                    metrics.update(_extra_control_metrics(pred, batch))
                before = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                overflow += int(scaler.get_scale() < before)
                _accumulate(train_sums, metrics)
                train_batches += 1
                if number in marks:
                    print(f"[epoch {epoch}/{args.epochs}] train {number}/{limit} "
                          f"loss={train_sums['total']/train_batches:.5f} "
                          f"elapsed={time.perf_counter()-started:.0f}s", flush=True)
                if args.max_train_batches and number >= args.max_train_batches:
                    break

            model.eval()
            val_sums: dict[str, float] = {}
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {key: value.to(device, non_blocking=True)
                             for key, value in batch.items()}
                    pred = model(batch["image"], batch["ego"], batch["racing_line"],
                                 batch["command"], batch["target_speed"])
                    _, metrics = compute_losses(pred, batch, stats, planner_only=False)
                    metrics.update(_extra_control_metrics(pred, batch))
                    _accumulate(val_sums, metrics)
                    val_batches += 1
                    if args.max_train_batches and val_batches >= args.max_train_batches:
                        break
            train_metrics, val_metrics = _mean(train_sums, train_batches), _mean(val_sums, val_batches)
            scheduler.step(val_metrics["total"])
            lr = float(optimizer.param_groups[0]["lr"])
            improved = val_metrics["total"] < best
            stale = 0 if improved else stale + 1
            if improved:
                best = val_metrics["total"]
            row = {
                "epoch": epoch, "lr": lr, "train": train_metrics, "val": val_metrics,
                "best_val": best, "is_best": improved, "amp_overflow_steps": overflow,
                "elapsed_s": time.perf_counter() - started,
            }
            history.append(row)
            for name, value in train_metrics.items():
                writer.add_scalar(f"loss/train_{name}", value, epoch)
            for name, value in val_metrics.items():
                writer.add_scalar(f"loss/val_{name}", value, epoch)
            writer.add_scalar("optimization/learning_rate", lr, epoch)
            writer.add_scalar("optimization/amp_overflow_steps", overflow, epoch)
            writer.add_scalar("optimization/best_val_total", best, epoch)
            writer.flush()

            phase = lap_phase_metadata()
            phase["enabled"] = True
            state = {
                "epoch": epoch, "stage": "v8_joint_supervised", "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "grad_scaler": scaler.state_dict() if amp else None,
                "model_schema": schema["model_schema"], "artifacts": schema["artifacts"],
                "normalization": normalization, "lap_phase": phase,
                "training_recipe": {
                    "version": 8, "requires_full_causal_gate": False,
                    "dataset": "single normal session; lap1-5 train; lap6-7 val",
                    "sampler": "uniform shuffled observations",
                    "causal_regularization": False,
                },
                "args": vars(args), "parameter_count": parameter_count,
                "val_loss": val_metrics["total"],
            }
            torch.save(state, args.output / "last.pt")
            if improved:
                torch.save(state, args.output / "best.pt")
            if args.checkpoint_every and epoch % args.checkpoint_every == 0:
                torch.save(state, args.output / f"epoch_{epoch:04d}.pt")
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(json.dumps({
                "epoch": epoch, "lr": lr, "train_total": train_metrics["total"],
                "val_total": val_metrics["total"], "val_control": val_metrics["control"],
                "val_waypoint_mae_m": val_metrics["waypoint_mae_m"],
                "val_future_speed_mae_mps": val_metrics["future_speed_mae_mps"],
                "best_val": best, "is_best": improved, "amp_overflow_steps": overflow,
            }), flush=True)
            if args.early_stopping_patience and stale >= args.early_stopping_patience:
                print(f"[early-stop] no validation improvement for {stale} epochs", flush=True)
                break
    finally:
        writer.close()


if __name__ == "__main__":
    main()
