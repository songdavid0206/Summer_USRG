"""Train the Laguna Seca solo TCP-AC baseline."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from .config import (
    FLYING_LAP_COMMAND_INDEX, SPEED_SCALE_MPS, STANDING_LAP_COMMAND_INDEX,
    lap_phase_metadata, normalization_metadata,
)


LOSS_NAMES = ("total", "action", "waypoint", "future_action", "speed")


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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def losses(pred, batch):
    import torch.nn.functional as F
    action = F.l1_loss(pred["action"], batch["action"])
    waypoint = F.l1_loss(pred["waypoints"], batch["waypoints"])
    future_action = F.l1_loss(pred["future_actions"], batch["future_actions"])
    speed = F.l1_loss(pred["speed"], batch["speed"] / SPEED_SCALE_MPS)
    total = action + waypoint + 0.5 * future_action + 0.05 * speed
    return total, {"action": action, "waypoint": waypoint, "future_action": future_action, "speed": speed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TCP-AC on Laguna Seca solo runs")
    parser.add_argument("--prepared", type=Path, default=Path("training/tcp_lagunaseca"))
    parser.add_argument("--output", type=Path, default=Path("training/tcp_lagunaseca/checkpoints"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imagenet-pretrained", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0,
                        help="nonzero for a quick smoke/overfit run")
    parser.add_argument("--progress-every", type=int, default=100,
                        help="print in-epoch progress every N batches (0 disables)")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from .data import TCPDataset
    from .model import build_model

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = TCPDataset(args.prepared, "train", augment=True)
    val_set = TCPDataset(args.prepared, "val", augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
    model = build_model(args.imagenet_pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-7)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    args.output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []

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

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sums = {name: 0.0 for name in LOSS_NAMES}
        train_batches = 0
        phase_started = time.perf_counter()
        for batch_number, batch in enumerate(train_loader, 1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch["image"], batch["speed"], batch["target_point"], batch["command"])
            loss, components = losses(pred, batch)
            loss.backward()
            optimizer.step()
            train_sums["total"] += float(loss.detach())
            for name, value in components.items():
                train_sums[name] += float(value.detach())
            train_batches += 1
            if (args.progress_every and
                    (batch_number == 1 or batch_number % args.progress_every == 0
                     or batch_number == train_phase_batches)):
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
        phase_started = time.perf_counter()
        with torch.no_grad():
            for batch_number, batch in enumerate(val_loader, 1):
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                pred = model(batch["image"], batch["speed"], batch["target_point"], batch["command"])
                loss, components = losses(pred, batch)
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
                    phase_loss, phase_components = losses(phase_pred, phase_batch)
                    val_phase_sums[phase]["total"] += float(phase_loss) * count
                    for name, value in phase_components.items():
                        val_phase_sums[phase][name] += float(value) * count
                    val_phase_samples[phase] += count
                if (args.progress_every and
                        (batch_number == 1 or batch_number % args.progress_every == 0
                         or batch_number == val_phase_batches)):
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
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
            "val_by_phase": val_by_phase,
        }
        history.append(row)
        print(json.dumps(row))
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "val_loss": val_metrics["total"], "args": vars(args),
                 "normalization": normalization_metadata(),
                 "lap_phase": lap_phase_metadata()}
        torch.save(state, args.output / "last.pt")
        if val_metrics["total"] < best:
            best = val_metrics["total"]
            torch.save(state, args.output / "best.pt")
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
