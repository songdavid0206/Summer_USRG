"""Train the Laguna Seca solo TCP-AC baseline."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from .config import (
    FLYING_LAP_COMMAND_INDEX, SPEED_SCALE_MPS, STANDING_LAP_COMMAND_INDEX,
    lap_phase_metadata, target_speed_input_width,
)


LOSS_NAMES = (
    "total", "control", "steer", "throttle", "brake", "waypoint",
    "future_control", "speed",
)


def _duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def _progress(epoch: int, epochs: int, phase: str, batch: int, phase_batches: int,
              completed_before: int, epoch_batches: int, started: float,
              average_loss: float) -> None:
    elapsed = time.perf_counter() - started
    rate = elapsed / max(1, batch)
    phase_eta = rate * max(0, phase_batches - batch)
    phase_percent = 100.0 * batch / max(1, phase_batches)
    epoch_percent = 100.0 * (completed_before + batch) / max(1, epoch_batches)
    print(
        f"[epoch {epoch}/{epochs}] {phase} {batch}/{phase_batches} "
        f"phase={phase_percent:5.1f}% epoch={epoch_percent:5.1f}% "
        f"loss={average_loss:.5f} elapsed={_duration(elapsed)} "
        f"phase_eta={_duration(phase_eta)}",
        flush=True,
    )


def _should_print_progress(batch: int, phase_batches: int, mode: str,
                           every: int, completed_before: int = 0,
                           epoch_batches: int | None = None) -> bool:
    if mode == "none":
        return False
    if mode == "every":
        return batch == phase_batches or (every > 0 and batch % every == 0)
    # Exactly four milestones over the complete train+validation epoch.
    total = phase_batches if epoch_batches is None else epoch_batches
    milestones = {math.ceil(total * fraction / 4) for fraction in range(1, 5)}
    return completed_before + batch in milestones


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _weighted_l1(prediction, target, weight):
    import torch
    error = (prediction - target).abs() * weight
    return error.sum() / torch.clamp(weight.expand_as(error).sum(), min=1.0)


def losses(pred, batch, speed_weight: float = .05):
    import torch.nn.functional as F
    steer = F.l1_loss(pred["controls"][:, 0], batch["controls"][:, 0])
    pedal_weight = batch.get("pedal_loss_weight", pred["controls"][:, 0].new_ones(
        pred["controls"].shape[0]))
    throttle = _weighted_l1(
        pred["controls"][:, 1], batch["controls"][:, 1], pedal_weight)
    brake = _weighted_l1(
        pred["controls"][:, 2], batch["controls"][:, 2], pedal_weight)
    control = steer + throttle + brake
    future_mask = batch.get(
        "future_mask", pred["waypoints"].new_ones(pred["waypoints"].shape[:2]))
    waypoint = _weighted_l1(
        pred["waypoints"], batch["waypoints"], future_mask.unsqueeze(-1))
    future_steer = _weighted_l1(
        pred["future_controls"][:, :, 0], batch["future_controls"][:, :, 0], future_mask)
    future_pedal_weight = batch.get(
        "future_pedal_loss_weight", future_mask.new_ones(future_mask.shape)) * future_mask
    future_throttle = _weighted_l1(
        pred["future_controls"][:, :, 1], batch["future_controls"][:, :, 1],
        future_pedal_weight)
    future_brake = _weighted_l1(
        pred["future_controls"][:, :, 2], batch["future_controls"][:, :, 2],
        future_pedal_weight)
    future_control = future_steer + future_throttle + future_brake
    speed = F.l1_loss(pred["speed"], batch["speed"] / SPEED_SCALE_MPS)
    total = control + waypoint + 0.5 * future_control + speed_weight * speed
    return total, {
        "control": control, "steer": steer, "throttle": throttle,
        "brake": brake, "waypoint": waypoint,
        "future_control": future_control, "speed": speed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TCP-AC on Laguna Seca solo runs")
    parser.add_argument("--prepared", type=Path, default=Path("training/tcp_lagunaseca"))
    parser.add_argument("--output", type=Path, default=Path("training/tcp_lagunaseca/checkpoints"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imagenet-pretrained", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                        help="CUDA mixed precision with dynamic loss scaling")
    parser.add_argument("--balanced-sampler", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--speed-loss-weight", type=float,
                        help="default: 0 for v4 target-speed input, otherwise 0.05")
    parser.add_argument("--early-stopping-patience", type=int, default=8,
                        help="stop after this many epochs without val improvement; 0 disables")
    parser.add_argument("--max-train-batches", type=int, default=0,
                        help="nonzero for a quick smoke/overfit run")
    parser.add_argument("--progress-mode", choices=("quarters", "every", "none"),
                        default="quarters",
                        help="quarters prints only 25/50/75/100 percent per phase")
    parser.add_argument("--progress-every", type=int, default=100,
                        help="batch interval used only with --progress-mode every")
    parser.add_argument("--resume", type=Path,
                        help="resume model and optimizer from a saved checkpoint")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from .data import TCPDataset
    from .model import build_model

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = TCPDataset(args.prepared, "train", augment=True)
    val_set = TCPDataset(args.prepared, "val", augment=False)
    prepared_schema = json.loads((args.prepared / "schema.json").read_text(encoding="utf-8"))
    model_schema = prepared_schema["model_schema"]
    normalization = prepared_schema["normalization"]
    target_width = target_speed_input_width(model_schema)
    speed_weight = (0.0 if target_width else .05) if args.speed_loss_weight is None else args.speed_loss_weight
    sampler = None
    if args.balanced_sampler and target_width:
        target_shares = {
            "normal_straight": .25, "normal_corner": .25, "normal_brake": .20,
            "synthetic_recovery_mild": .09, "synthetic_recovery_severe": .025,
            "synthetic_recovery_extreme": .005,
            "dagger_mild": .14, "dagger_severe": .035, "dagger_extreme": .005,
        }
        counts = {name: train_set.sample_buckets.count(name)
                  for name in set(train_set.sample_buckets)}
        weights = [target_shares.get(name, 0.0) / max(1, counts.get(name, 0))
                   for name in train_set.sample_buckets]
        if not any(weights):
            raise RuntimeError("balanced sampler found no recognized v4 buckets")
        sampler = WeightedRandomSampler(weights, num_samples=len(train_set), replacement=True)
        print(f"[sampler] target_shares={target_shares} observed_counts={counts}", flush=True)
    worker_options = ({"persistent_workers": True, "prefetch_factor": 2}
                      if args.workers > 0 else {})
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=sampler is None, sampler=sampler,
                              num_workers=args.workers, pin_memory=device.type == "cuda",
                              **worker_options)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda",
                            **worker_options)
    if train_set.offsets != val_set.offsets:
        raise RuntimeError("train/validation planning horizons differ")
    model = build_model(
        args.imagenet_pretrained and args.resume is None,
        pred_len=len(train_set.offsets),
        target_speed_width=target_width,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled = bool(args.amp and device.type == "cuda")
    # TITAN RTX/this model overflows once at PyTorch's 65536 default. Starting
    # at 32768 avoids throwing away the first optimizer update.
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled, init_scale=32768.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    args.output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    start_epoch = 1
    epochs_without_improvement = 0

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if amp_enabled and checkpoint.get("grad_scaler") is not None:
            scaler.load_state_dict(checkpoint["grad_scaler"])
        completed_epoch = int(checkpoint["epoch"])
        start_epoch = completed_epoch + 1

        history_path = args.output / "history.json"
        if not history_path.exists():
            raise FileNotFoundError(
                f"resume history does not exist: {history_path}"
            )
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not history or int(history[-1]["epoch"]) != completed_epoch:
            raise RuntimeError(
                "resume checkpoint epoch does not match the output history"
            )
        best = min(float(row["val"]["total"]) for row in history)
        best_epoch = max(int(row["epoch"]) for row in history
                         if float(row["val"]["total"]) == best)
        epochs_without_improvement = completed_epoch - best_epoch

        # Older checkpoints did not store scheduler state. StepLR only depends on
        # the completed epoch count and the optimizer's saved learning rate.
        scheduler.last_epoch = completed_epoch
        scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
        print(
            f"[resume] checkpoint={args.resume} completed_epoch={completed_epoch} "
            f"next_epoch={start_epoch} best_val={best:.8f}",
            flush=True,
        )

    train_phase_batches = (min(len(train_loader), args.max_train_batches)
                           if args.max_train_batches else len(train_loader))
    val_phase_batches = (min(len(val_loader), args.max_train_batches)
                         if args.max_train_batches else len(val_loader))
    epoch_batches = train_phase_batches + val_phase_batches
    print(
        f"[dataset] train_samples={len(train_set)} val_samples={len(val_set)} "
        f"batch_size={args.batch_size} train_batches={train_phase_batches} "
        f"val_batches={val_phase_batches} batches_per_epoch={epoch_batches}",
        flush=True,
    )
    print(f"[precision] amp={'enabled' if amp_enabled else 'disabled'}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        amp_overflow_steps = 0
        train_sums = {name: 0.0 for name in LOSS_NAMES}
        train_batches = 0
        phase_started = time.perf_counter()
        for batch_number, batch in enumerate(train_loader, 1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                pred = model(
                    batch["image"], batch["ego"], batch["racing_line"], batch["command"],
                    batch["target_speed"] if target_width else None)
                loss, components = losses(pred, batch, speed_weight)
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            amp_overflow_steps += int(scaler.get_scale() < scale_before)
            train_sums["total"] += float(loss.detach())
            for name, value in components.items():
                train_sums[name] += float(value.detach())
            train_batches += 1
            if _should_print_progress(
                    batch_number, train_phase_batches,
                    args.progress_mode, args.progress_every,
                    completed_before=0, epoch_batches=epoch_batches):
                _progress(
                    epoch, args.epochs, "train", batch_number, train_phase_batches,
                    0, epoch_batches, phase_started,
                    train_sums["total"] / train_batches,
                )
            if args.max_train_batches and batch_number >= args.max_train_batches:
                break

        model.eval()
        val_sums = {name: 0.0 for name in LOSS_NAMES}
        val_batches = 0
        val_phase_sums = {
            phase: {name: 0.0 for name in LOSS_NAMES}
            for phase in ("standing", "flying")
        }
        val_phase_samples = {"standing": 0, "flying": 0}
        val_source_sums = {
            source: {name: 0.0 for name in LOSS_NAMES}
            for source in ("normal", "synthetic_recovery", "dagger")
        }
        val_source_samples = {source: 0 for source in val_source_sums}
        phase_started = time.perf_counter()
        with torch.no_grad():
            for batch_number, batch in enumerate(val_loader, 1):
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                    pred = model(
                        batch["image"], batch["ego"], batch["racing_line"], batch["command"],
                        batch["target_speed"] if target_width else None)
                    loss, components = losses(pred, batch, speed_weight)
                val_sums["total"] += float(loss)
                for name, value in components.items():
                    val_sums[name] += float(value)
                val_batches += 1
                for phase, command_index in (
                    ("standing", STANDING_LAP_COMMAND_INDEX),
                    ("flying", FLYING_LAP_COMMAND_INDEX),
                ):
                    mask = batch["command"][:, command_index] > 0.5
                    count = int(mask.sum())
                    if not count:
                        continue
                    phase_pred = {key: value[mask] for key, value in pred.items()}
                    phase_batch = {key: value[mask] for key, value in batch.items()}
                    phase_loss, phase_components = losses(phase_pred, phase_batch, speed_weight)
                    val_phase_sums[phase]["total"] += float(phase_loss) * count
                    for name, value in phase_components.items():
                        val_phase_sums[phase][name] += float(value) * count
                    val_phase_samples[phase] += count
                for source, source_id in (("normal", 0), ("synthetic_recovery", 1),
                                          ("dagger", 2)):
                    mask = batch["source_id"] == source_id
                    count = int(mask.sum())
                    if not count:
                        continue
                    source_pred = {key: value[mask] for key, value in pred.items()}
                    source_batch = {key: value[mask] for key, value in batch.items()}
                    source_loss, source_components = losses(
                        source_pred, source_batch, speed_weight)
                    val_source_sums[source]["total"] += float(source_loss) * count
                    for name, value in source_components.items():
                        val_source_sums[source][name] += float(value) * count
                    val_source_samples[source] += count
                if _should_print_progress(
                        batch_number, val_phase_batches,
                        args.progress_mode, args.progress_every,
                        completed_before=train_phase_batches,
                        epoch_batches=epoch_batches):
                    _progress(
                        epoch, args.epochs, "val", batch_number, val_phase_batches,
                        train_phase_batches, epoch_batches, phase_started,
                        val_sums["total"] / val_batches,
                    )
                if args.max_train_batches and batch_number >= args.max_train_batches:
                    break
        scheduler.step()
        train_metrics = {
            name: value / max(1, train_batches) for name, value in train_sums.items()
        }
        val_metrics = {
            name: value / max(1, val_batches) for name, value in val_sums.items()
        }
        val_by_phase = {
            phase: {
                name: value / max(1, val_phase_samples[phase])
                for name, value in sums.items()
            }
            for phase, sums in val_phase_sums.items()
        }
        val_by_source = {
            source: {name: value / max(1, val_source_samples[source])
                     for name, value in sums.items()}
            for source, sums in val_source_sums.items()
        }
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
            "val_by_phase": val_by_phase,
            "val_by_source": val_by_source,
            "amp": {"enabled": amp_enabled, "overflow_steps": amp_overflow_steps,
                    "scale": scaler.get_scale() if amp_enabled else None},
        }
        history.append(row)
        console_row = {
            "epoch": epoch, "lr": row["lr"],
            "train_total": train_metrics["total"],
            "val_total": val_metrics["total"],
            "val_control": val_metrics["control"],
            "val_waypoint": val_metrics["waypoint"],
            "val_future_control": val_metrics["future_control"],
            "val_by_source_total": {
                source: metrics["total"] for source, metrics in val_by_source.items()
                if val_source_samples[source] > 0
            },
            "amp_overflow_steps": amp_overflow_steps,
        }
        print(json.dumps(console_row))
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(),
                 "grad_scaler": scaler.state_dict() if amp_enabled else None,
                 "amp_enabled": amp_enabled,
                 "val_loss": val_metrics["total"], "args": vars(args),
                 "normalization": normalization,
                 "model_schema": model_schema,
                 "speed_loss_weight": speed_weight,
                 "lap_phase": lap_phase_metadata()}
        torch.save(state, args.output / "last.pt")
        if val_metrics["total"] < best:
            best = val_metrics["total"]
            epochs_without_improvement = 0
            torch.save(state, args.output / "best.pt")
        else:
            epochs_without_improvement += 1
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if (args.early_stopping_patience > 0 and
                epochs_without_improvement >= args.early_stopping_patience):
            print(f"[early-stop] no validation improvement for "
                  f"{epochs_without_improvement} epochs", flush=True)
            break


if __name__ == "__main__":
    main()
