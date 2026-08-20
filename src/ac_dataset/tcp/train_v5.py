"""Staged v5 training: planner warm-up, causal gate, then predicted-plan joint IL."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from .config import PLANNING_HORIZONS_SECONDS, lap_phase_metadata
from .profile_v5 import sha256_file


TARGET_SHARES = {
    "normal_straight": .10, "normal_full_throttle": .10,
    "normal_corner": .12, "normal_pre_lift": .12,
    "normal_pre_brake": .12, "normal_braking": .09,
    "normal_brake_release_turnin": .05,
    "synthetic_recovery_mild": .12, "synthetic_recovery_severe": .025,
    "synthetic_recovery_extreme": .005,
    "dagger_mild": .123, "dagger_severe": .025, "dagger_extreme": .002,
}


def _seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor_stats(normalization: dict, device):
    import torch
    result = {}
    for name in ("waypoints_local_m", "future_speed_mps", "target_speed_features_mps"):
        result[name] = (
            torch.as_tensor(normalization[name]["mean"], device=device, dtype=torch.float32),
            torch.as_tensor(normalization[name]["std"], device=device, dtype=torch.float32),
        )
    return result


def _masked_mean(value, mask):
    expanded = mask
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(value)
    return (value * expanded).sum() / expanded.sum().clamp_min(1.0)


def compute_losses(pred, batch, stats, planner_only: bool):
    import torch

    wp_mean, wp_std = stats["waypoints_local_m"]
    speed_mean, speed_std = stats["future_speed_mps"]
    pred_wp_m = pred["waypoints"].float() * wp_std + wp_mean
    pred_speed_mps = pred["future_speeds"].float().squeeze(-1) * speed_std + speed_mean
    target_wp_m = batch["waypoints_m"].float()
    target_speed_mps = batch["future_speeds_mps"].float().squeeze(-1)
    mask = batch["future_mask"].float()

    # Explicit physical tolerances replace opaque scale division.
    wp_error = (pred_wp_m - target_wp_m).abs() / pred_wp_m.new_tensor([2.0, 5.0])
    waypoint = _masked_mean(wp_error, mask)
    future_speed = _masked_mean((pred_speed_mps - target_speed_mps).abs() / 5.0, mask)
    horizons = pred_wp_m.new_tensor(PLANNING_HORIZONS_SECONDS)
    origin = pred_wp_m.new_zeros((pred_wp_m.shape[0], 1, 2))
    segments = pred_wp_m - torch.cat((origin, pred_wp_m[:, :-1]), dim=1)
    dt = torch.cat((horizons[:1], horizons[1:] - horizons[:-1]))
    implied_speed = torch.linalg.vector_norm(segments, dim=-1) / dt
    # A waypoint segment encodes distance travelled over an interval, so its
    # physically corresponding speed is the interval average, not the endpoint
    # speed. Use trapezoidal integration with current ego speed for segment 0.
    current_speed_mps = batch["ego_raw"][:, :1].float().abs()
    previous_speed_mps = torch.cat((current_speed_mps, pred_speed_mps[:, :-1]), dim=1)
    segment_average_speed = .5 * (previous_speed_mps + pred_speed_mps)
    consistency = _masked_mean(
        (implied_speed - segment_average_speed).abs() / 5.0, mask)

    if planner_only:
        control = pred_wp_m.new_zeros(())
        future_control = pred_wp_m.new_zeros(())
        total = waypoint + future_speed + .25 * consistency
    else:
        current_error = (pred["controls"].float() - batch["controls"].float()).abs()
        current_weight = torch.stack((
            torch.ones_like(batch["pedal_loss_weight"]),
            batch["pedal_loss_weight"], batch["pedal_loss_weight"]), dim=1)
        control = (current_error * current_weight).sum() / current_weight.sum().clamp_min(1.0)
        future_error = (pred["future_controls"].float() -
                        batch["future_controls"].float()).abs()
        pedal = batch["future_pedal_loss_weight"].float()
        future_weight = torch.stack((torch.ones_like(pedal), pedal, pedal), dim=2)
        future_control_mask = batch.get("future_control_mask", mask).float()
        future_weight = future_weight * future_control_mask.unsqueeze(-1)
        future_control = (future_error * future_weight).sum() / future_weight.sum().clamp_min(1.0)
        total = control + waypoint + future_speed + .5 * future_control + .25 * consistency
    return total, {
        "total": total, "control": control, "waypoint": waypoint,
        "future_speed": future_speed, "future_control": future_control,
        "consistency": consistency,
        "waypoint_mae_m": _masked_mean((pred_wp_m - target_wp_m).abs(), mask),
        "future_speed_mae_mps": _masked_mean(
            (pred_speed_mps - target_speed_mps).abs(), mask),
        "residual_abs_max": pred["visual_residual"].float().abs().max(),
    }


def causal_probe(model, batch, stats, drop_mps: float = 10.0) -> dict[str, float]:
    """Same observations, only the target profile is lowered counterfactually."""
    import torch
    model.eval()
    target_mean, target_std = stats["target_speed_features_mps"]
    speed_mean, speed_std = stats["future_speed_mps"]
    original = batch["target_speed"].float()
    physical = original * target_std + target_mean
    lowered_physical = physical - drop_mps
    lowered = (lowered_physical - target_mean) / target_std
    with torch.no_grad():
        baseline = model(batch["image"], batch["ego"], batch["racing_line"],
                         batch["command"], original)
        changed = model(batch["image"], batch["ego"], batch["racing_line"],
                        batch["command"], lowered)
    baseline_speed = baseline["future_speeds"].float().squeeze(-1) * speed_std + speed_mean
    changed_speed = changed["future_speeds"].float().squeeze(-1) * speed_std + speed_mean
    return {
        "target_drop_mps": float(drop_mps),
        "future_speed_delta_mps": float((changed_speed - baseline_speed).mean()),
        "future_speed_monotonic_fraction": float(
            ((changed_speed - baseline_speed).mean(dim=1) < 0).float().mean()),
        "plan_throttle_delta": float(
            (changed["plan_controls"][:, 1] - baseline["plan_controls"][:, 1]).mean()),
        "plan_brake_delta": float(
            (changed["plan_controls"][:, 2] - baseline["plan_controls"][:, 2]).mean()),
        "final_throttle_delta": float(
            (changed["controls"][:, 1] - baseline["controls"][:, 1]).mean()),
        "final_brake_delta": float(
            (changed["controls"][:, 2] - baseline["controls"][:, 2]).mean()),
        "residual_change_abs_max": float(
            (changed["visual_residual"] - baseline["visual_residual"]).abs().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train frozen TCP-AC v5")
    parser.add_argument("--prepared", type=Path, default=Path("training/tcp_lagunaseca_v5"))
    parser.add_argument("--output", type=Path,
                        default=Path("training/tcp_lagunaseca_v5/checkpoints"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--planner-warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imagenet-pretrained", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--balanced-sampler", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--causal-probe-min-drop-mps", type=float, default=.50,
                        help="minimum mean future-speed response to a 10m/s target reduction")
    parser.add_argument("--require-causal-probe", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--resume", type=Path,
                        help="resume a v5 model/optimizer/scheduler from last.pt")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from .data_v5 import TCPDatasetV5
    from .model_v5 import build_model_v5

    _seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = TCPDatasetV5(args.prepared, "train", augment=True)
    val_set = TCPDatasetV5(args.prepared, "val", augment=False)
    schema = json.loads((args.prepared / "schema.json").read_text(encoding="utf-8"))
    normalization = json.loads((args.prepared / "normalization.json").read_text(encoding="utf-8"))
    stats = _tensor_stats(normalization, device)
    sampler = None
    if args.balanced_sampler:
        counts = {name: train_set.sample_buckets.count(name)
                  for name in set(train_set.sample_buckets)}
        available_total = sum(share for name, share in TARGET_SHARES.items()
                              if counts.get(name, 0))
        weights = [TARGET_SHARES.get(name, 0.0) / available_total /
                   max(1, counts.get(name, 0)) for name in train_set.sample_buckets]
        if not any(weights):
            raise RuntimeError("v5 sampler found no recognized buckets")
        sampler = WeightedRandomSampler(weights, len(train_set), replacement=True)
        print(f"[sampler] target_shares={TARGET_SHARES} observed_counts={counts}", flush=True)
    worker_args = ({"persistent_workers": True, "prefetch_factor": 2}
                   if args.workers else {})
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
        num_workers=args.workers, pin_memory=device.type == "cuda", **worker_args)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda", **worker_args)
    model = build_model_v5(args.imagenet_pretrained and args.resume is None).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=32768.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=.5)
    args.output.mkdir(parents=True, exist_ok=True)
    history, best, stale, start_epoch = [], float("inf"), 0, 1
    probe_batch = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("model_schema") != schema["model_schema"]:
            raise RuntimeError("resume checkpoint/prepared v5 schema mismatch")
        if checkpoint.get("artifacts") != schema["artifacts"]:
            raise RuntimeError("resume checkpoint/prepared artifact mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if amp and checkpoint.get("grad_scaler") is not None:
            scaler.load_state_dict(checkpoint["grad_scaler"])
        completed_epoch = int(checkpoint["epoch"])
        start_epoch = completed_epoch + 1
        history_path = args.output / "history.json"
        if not history_path.exists():
            raise FileNotFoundError(f"resume history does not exist: {history_path}")
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not history or int(history[-1]["epoch"]) != completed_epoch:
            raise RuntimeError("resume checkpoint epoch does not match output history")
        joint_rows = [row for row in history if row.get("stage") == "joint_predicted_plan"]
        if joint_rows:
            best = min(float(row["val"]["total"]) for row in joint_rows)
            best_epoch = max(int(row["epoch"]) for row in joint_rows
                             if float(row["val"]["total"]) == best)
            stale = completed_epoch - best_epoch
        resume_probe = checkpoint.get("causal_probe", {})
        if (args.require_causal_probe and completed_epoch >= args.planner_warmup_epochs and
                (float(resume_probe.get("future_speed_delta_mps", float("inf"))) >
                 -abs(args.causal_probe_min_drop_mps) or
                 float(resume_probe.get("future_speed_monotonic_fraction", 0.0)) < .75)):
            raise RuntimeError("resume checkpoint did not pass the planner causal gate")
        print(f"[resume] checkpoint={args.resume} completed_epoch={completed_epoch} "
              f"next_epoch={start_epoch} best_joint_val={best}", flush=True)
    print(f"[dataset] train={len(train_set)} val={len(val_set)} batches="
          f"{len(train_loader)}+{len(val_loader)} device={device} amp={amp}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        planner_only = epoch <= args.planner_warmup_epochs
        stage = "planner_warmup" if planner_only else "joint_predicted_plan"
        model.train()
        sums, batches, overflow = {}, 0, 0
        started = time.perf_counter()
        train_limit = min(len(train_loader), args.max_train_batches) if args.max_train_batches else len(train_loader)
        marks = {max(1, round(train_limit * fraction)) for fraction in (.25, .5, .75, 1.0)}
        for number, batch in enumerate(train_loader, 1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            if probe_batch is None:
                probe_batch = {key: value[:min(32, len(value))].detach().clone()
                               for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                pred = model(batch["image"], batch["ego"], batch["racing_line"],
                             batch["command"], batch["target_speed"])
                loss, metrics = compute_losses(pred, batch, stats, planner_only)
            before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            overflow += int(scaler.get_scale() < before)
            for name, value in metrics.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach())
            batches += 1
            if number in marks:
                print(f"[epoch {epoch}/{args.epochs}] {stage} train {number}/{train_limit} "
                      f"loss={sums['total']/batches:.5f} elapsed={time.perf_counter()-started:.0f}s",
                      flush=True)
            if args.max_train_batches and number >= args.max_train_batches:
                break

        model.eval()
        val_sums, val_batches = {}, 0
        val_source_sums = {name: {} for name in ("normal", "synthetic_recovery", "dagger")}
        val_source_counts = {name: 0 for name in val_source_sums}
        val_phase_sums = {name: {} for name in ("standing", "flying")}
        val_phase_counts = {name: 0 for name in val_phase_sums}
        with torch.no_grad():
            for number, batch in enumerate(val_loader, 1):
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                    pred = model(batch["image"], batch["ego"], batch["racing_line"],
                                 batch["command"], batch["target_speed"])
                    _, metrics = compute_losses(pred, batch, stats, planner_only)
                for name, value in metrics.items():
                    val_sums[name] = val_sums.get(name, 0.0) + float(value)
                val_batches += 1
                for source_name, source_id in (("normal", 0), ("synthetic_recovery", 1),
                                               ("dagger", 2)):
                    source_mask = batch["source_id"] == source_id
                    count = int(source_mask.sum())
                    if count:
                        source_pred = {key: value[source_mask] for key, value in pred.items()}
                        source_batch = {key: value[source_mask] for key, value in batch.items()}
                        _, source_metrics = compute_losses(
                            source_pred, source_batch, stats, planner_only)
                        for name, value in source_metrics.items():
                            target = val_source_sums[source_name]
                            target[name] = target.get(name, 0.0) + float(value) * count
                        val_source_counts[source_name] += count
                for phase_name, command_index in (("standing", 4), ("flying", 5)):
                    phase_mask = batch["command"][:, command_index] > .5
                    count = int(phase_mask.sum())
                    if count:
                        phase_pred = {key: value[phase_mask] for key, value in pred.items()}
                        phase_batch = {key: value[phase_mask] for key, value in batch.items()}
                        _, phase_metrics = compute_losses(
                            phase_pred, phase_batch, stats, planner_only)
                        for name, value in phase_metrics.items():
                            target = val_phase_sums[phase_name]
                            target[name] = target.get(name, 0.0) + float(value) * count
                        val_phase_counts[phase_name] += count
                if args.max_train_batches and number >= args.max_train_batches:
                    break
        scheduler.step()
        train_metrics = {k: v / max(1, batches) for k, v in sums.items()}
        val_metrics = {k: v / max(1, val_batches) for k, v in val_sums.items()}
        val_by_source = {
            source: {name: value / max(1, val_source_counts[source])
                     for name, value in metrics.items()}
            for source, metrics in val_source_sums.items()
        }
        val_by_phase = {
            phase: {name: value / max(1, val_phase_counts[phase])
                    for name, value in metrics.items()}
            for phase, metrics in val_phase_sums.items()
        }
        probe = causal_probe(model, probe_batch, stats)
        row = {"epoch": epoch, "stage": stage, "lr": optimizer.param_groups[0]["lr"],
               "train": train_metrics, "val": val_metrics,
               "val_by_source": val_by_source, "val_by_phase": val_by_phase,
               "causal_probe": probe,
               "amp_overflow_steps": overflow}
        history.append(row)
        print(json.dumps({"epoch": epoch, "stage": stage,
                          "train_total": train_metrics["total"],
                          "val_total": val_metrics["total"],
                          "val_waypoint_mae_m": val_metrics["waypoint_mae_m"],
                          "val_future_speed_mae_mps": val_metrics["future_speed_mae_mps"],
                          "val_by_source_total": {
                              source: metrics.get("total")
                              for source, metrics in val_by_source.items()
                              if val_source_counts[source]
                          },
                          "causal_future_speed_delta_mps": probe["future_speed_delta_mps"],
                          "amp_overflow_steps": overflow}), flush=True)
        state = {
            "epoch": epoch, "stage": stage, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "grad_scaler": scaler.state_dict() if amp else None,
            "model_schema": schema["model_schema"], "artifacts": schema["artifacts"],
            "normalization": normalization, "args": vars(args), "causal_probe": probe,
            "lap_phase": lap_phase_metadata(),
            "val_loss": val_metrics["total"],
        }
        torch.save(state, args.output / "last.pt")
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if epoch == args.planner_warmup_epochs:
            torch.save(state, args.output / "planner_warmup.pt")
            passed = (
                probe["future_speed_delta_mps"] <= -abs(args.causal_probe_min_drop_mps)
                and probe["future_speed_monotonic_fraction"] >= .75
            )
            print(f"[causal-gate] planner target->future_speed passed={passed} probe={probe}",
                  flush=True)
            if args.require_causal_probe and not passed:
                raise RuntimeError("planner causal probe failed; joint training is blocked")
        if not planner_only:
            if val_metrics["total"] < best:
                best, stale = val_metrics["total"], 0
                torch.save(state, args.output / "best.pt")
            else:
                stale += 1
            if args.early_stopping_patience and stale >= args.early_stopping_patience:
                print(f"[early-stop] no joint validation improvement for {stale} epochs")
                break


if __name__ == "__main__":
    main()
