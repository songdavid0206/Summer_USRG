"""V6 training: observational IL plus paired target-speed causal regularization."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from .config import RACING_LINE_LOOKAHEADS_M, lap_phase_metadata
from .train_v5 import TARGET_SHARES, _tensor_stats, compute_losses


RECIPE_VERSION = 6


def _seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lowered_target(target, stats, drop_mps: float, future_only: bool = False):
    """Create uniform or realistic upcoming-corner target perturbations."""
    _, target_std = stats["target_speed_features_mps"]
    changed = target.clone()
    if future_only:
        indices = [index for index, distance in enumerate(RACING_LINE_LOOKAHEADS_M)
                   if distance >= 20.0]
        changed[:, indices] -= float(drop_mps) / target_std[indices]
    else:
        # Includes the final current-target-minus-current-speed feature.
        changed -= float(drop_mps) / target_std
    return changed


def _per_sample_masked_mean(value, mask):
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    dims = tuple(range(1, value.ndim))
    return (value * mask).sum(dim=dims) / mask.sum(dim=dims).clamp_min(1.0)


def causal_pair_metrics(baseline, changed, batch, stats,
                        min_speed_drop_mps: float = .50,
                        min_longitudinal_delta: float = .10):
    """Differentiable penalties for a lower target under identical observations."""
    import torch

    _, speed_std = stats["future_speed_mps"]
    speed_delta = (changed["future_speeds"].float().squeeze(-1) -
                   baseline["future_speeds"].float().squeeze(-1)) * speed_std
    speed_delta = _per_sample_masked_mean(speed_delta, batch["future_mask"].float())

    plan_throttle = changed["plan_controls"][:, 1] - baseline["plan_controls"][:, 1]
    plan_brake = changed["plan_controls"][:, 2] - baseline["plan_controls"][:, 2]
    final_throttle = changed["controls"][:, 1] - baseline["controls"][:, 1]
    final_brake = changed["controls"][:, 2] - baseline["controls"][:, 2]
    # This is a derived training diagnostic only. The network still predicts
    # independent throttle and brake outputs; it is not a signed-action model.
    plan_longitudinal = plan_throttle - plan_brake
    final_longitudinal = final_throttle - final_brake

    # Scale the physical-speed violation by the same 5m/s tolerance used by
    # the supervised future-speed loss. Control violations already use [0,1].
    speed_penalty = torch.relu(speed_delta + min_speed_drop_mps).mean() / 5.0
    longitudinal_penalty = torch.stack((
        torch.relu(plan_longitudinal + min_longitudinal_delta),
        torch.relu(final_longitudinal + min_longitudinal_delta),
    )).mean()
    # Do not obtain the net reduction by making either physical pedal move in
    # the wrong direction. No per-pedal magnitude is imposed because a pedal
    # can validly be saturated at zero; magnitude is enforced on net authority.
    direction_penalty = torch.stack((
        torch.relu(plan_throttle), torch.relu(-plan_brake),
        torch.relu(final_throttle), torch.relu(-final_brake),
    )).mean()
    control_penalty = longitudinal_penalty + direction_penalty
    loss = speed_penalty + control_penalty
    return loss, {
        "causal_loss": loss,
        "causal_speed_penalty": speed_penalty,
        "causal_control_penalty": control_penalty,
        "causal_longitudinal_penalty": longitudinal_penalty,
        "causal_direction_penalty": direction_penalty,
        "causal_future_speed_delta_mps": speed_delta.mean(),
        "causal_plan_throttle_delta": plan_throttle.mean(),
        "causal_plan_brake_delta": plan_brake.mean(),
        "causal_final_throttle_delta": final_throttle.mean(),
        "causal_final_brake_delta": final_brake.mean(),
        "causal_plan_longitudinal_delta": plan_longitudinal.mean(),
        "causal_final_longitudinal_delta": final_longitudinal.mean(),
    }


def _causal_probe_flat(model, batch, stats, drop_mps: float,
                       future_only: bool = False) -> dict[str, float]:
    import torch

    model.eval()
    changed_target = lowered_target(
        batch["target_speed"].float(), stats, drop_mps, future_only=future_only)
    with torch.no_grad():
        visual = model.encode_image(batch["image"])
        baseline = model.forward_from_visual(
            visual, batch["ego"], batch["racing_line"], batch["command"],
            batch["target_speed"])
        changed = model.forward_from_visual(
            visual, batch["ego"], batch["racing_line"], batch["command"], changed_target)
    _, speed_std = stats["future_speed_mps"]
    speed_delta = (changed["future_speeds"].float().squeeze(-1) -
                   baseline["future_speeds"].float().squeeze(-1)) * speed_std
    speed_delta = _per_sample_masked_mean(speed_delta, batch["future_mask"].float())
    plan_throttle = changed["plan_controls"][:, 1] - baseline["plan_controls"][:, 1]
    plan_brake = changed["plan_controls"][:, 2] - baseline["plan_controls"][:, 2]
    final_throttle = changed["controls"][:, 1] - baseline["controls"][:, 1]
    final_brake = changed["controls"][:, 2] - baseline["controls"][:, 2]
    plan_longitudinal = plan_throttle - plan_brake
    final_longitudinal = final_throttle - final_brake
    return {
        "target_drop_mps": float(drop_mps),
        "perturbation": "future_20m_plus" if future_only else "uniform_all_targets",
        "samples": int(speed_delta.shape[0]),
        "future_speed_delta_mps": float(speed_delta.mean()),
        "future_speed_monotonic_fraction": float((speed_delta < 0).float().mean()),
        "plan_throttle_delta": float(plan_throttle.mean()),
        "plan_throttle_monotonic_fraction": float((plan_throttle <= 0).float().mean()),
        "plan_brake_delta": float(plan_brake.mean()),
        "plan_brake_monotonic_fraction": float((plan_brake >= 0).float().mean()),
        "plan_longitudinal_delta": float(plan_longitudinal.mean()),
        "plan_longitudinal_monotonic_fraction": float(
            (plan_longitudinal <= 0).float().mean()),
        "final_throttle_delta": float(final_throttle.mean()),
        "final_throttle_monotonic_fraction": float((final_throttle <= 0).float().mean()),
        "final_brake_delta": float(final_brake.mean()),
        "final_brake_monotonic_fraction": float((final_brake >= 0).float().mean()),
        "final_longitudinal_delta": float(final_longitudinal.mean()),
        "final_longitudinal_monotonic_fraction": float(
            (final_longitudinal <= 0).float().mean()),
        "residual_change_abs_max": float(
            (changed["visual_residual"] - baseline["visual_residual"]).abs().max()),
    }


def causal_probe_v6(model, batch, stats, drop_mps: float = 10.0) -> dict[str, object]:
    """Probe an equal-source validation batch, reporting aggregate and source gates."""
    result = _causal_probe_flat(model, batch, stats, drop_mps)
    anticipation = _causal_probe_flat(
        model, batch, stats, drop_mps, future_only=True)
    if "source_id" in batch:
        names = {0: "normal", 1: "synthetic_recovery", 2: "dagger"}
        by_source = {}
        for source_id, name in names.items():
            mask = batch["source_id"] == source_id
            if int(mask.sum()):
                by_source[name] = _causal_probe_flat(
                    model, {key: value[mask] for key, value in batch.items()},
                    stats, drop_mps)
        anticipation_by_source = {}
        for source_id, name in names.items():
            mask = batch["source_id"] == source_id
            if int(mask.sum()):
                anticipation_by_source[name] = _causal_probe_flat(
                    model, {key: value[mask] for key, value in batch.items()},
                    stats, drop_mps, future_only=True)
        result["by_source"] = by_source
        anticipation["by_source"] = anticipation_by_source
    result["anticipation"] = anticipation
    return result


def planner_gate_passed(probe: dict, min_speed_drop_mps: float = .50) -> bool:
    return (
        probe["future_speed_delta_mps"] <= -abs(min_speed_drop_mps)
        and probe["future_speed_monotonic_fraction"] >= .75
    )


def full_gate_passed(probe: dict, min_speed_drop_mps: float = .50,
                     min_longitudinal_delta: float = .10,
                     anticipation_min_speed_drop_mps: float = .25,
                     anticipation_min_longitudinal_delta: float = .05) -> bool:
    own_pass = (
        planner_gate_passed(probe, min_speed_drop_mps)
        and probe["plan_throttle_delta"] <= 0
        and probe["plan_throttle_monotonic_fraction"] >= .75
        and probe["plan_brake_delta"] >= 0
        and probe["plan_brake_monotonic_fraction"] >= .75
        and probe["plan_longitudinal_delta"] <= -abs(min_longitudinal_delta)
        and probe["plan_longitudinal_monotonic_fraction"] >= .75
        and probe["final_throttle_delta"] <= 0
        and probe["final_throttle_monotonic_fraction"] >= .75
        and probe["final_brake_delta"] >= 0
        and probe["final_brake_monotonic_fraction"] >= .75
        and probe["final_longitudinal_delta"] <= -abs(min_longitudinal_delta)
        and probe["final_longitudinal_monotonic_fraction"] >= .75
        and probe["residual_change_abs_max"] <= 1e-6
    )
    # The aggregate batch is equally stratified, but an average can still hide
    # one failed domain. Every represented source must independently pass.
    sources_pass = all(
        full_gate_passed(source_probe, min_speed_drop_mps, min_longitudinal_delta)
        for source_probe in probe.get("by_source", {}).values()
    )
    anticipation = probe.get("anticipation")
    anticipation_pass = True if anticipation is None else full_gate_passed(
        anticipation, anticipation_min_speed_drop_mps,
        anticipation_min_longitudinal_delta)
    return own_pass and sources_pass and anticipation_pass


def _stratified_probe_batch(dataset, device, per_source: int = 32):
    """Fixed, augmentation-free validation probe spanning all three sources."""
    import torch
    from torch.utils.data import DataLoader, Subset

    indices = []
    for source in ("normal", "synthetic_recovery", "dagger"):
        found = np.flatnonzero(dataset.samples["source_type"].to_numpy() == source)
        if len(found):
            take = min(per_source, len(found))
            indices.extend(found[np.linspace(0, len(found) - 1, take, dtype=int)].tolist())
    if not indices:
        raise RuntimeError("could not construct the v6 stratified validation probe")
    batch = next(iter(DataLoader(Subset(dataset, indices), batch_size=len(indices),
                                 shuffle=False, num_workers=0)))
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def main(recipe_version: int = RECIPE_VERSION,
         default_prepared: Path = Path("training/tcp_lagunaseca_v5"),
         default_output: Path = Path("training/tcp_lagunaseca_v6/checkpoints"),
         description: str = "Train TCP-AC v6 causal plan policy",
         dataset_class=None) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--prepared", type=Path, default=default_prepared)
    parser.add_argument("--output", type=Path,
                        default=default_output)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--planner-warmup-epochs", type=int, default=5)
    parser.add_argument("--init-planner", type=Path,
                        help="initialize from a compatible v5 planner_warmup.pt")
    parser.add_argument("--resume", type=Path, help="resume a v6 last.pt")
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
    parser.add_argument("--causal-drop-mps", type=float, default=10.0)
    parser.add_argument("--causal-min-speed-drop-mps", type=float, default=.50)
    parser.add_argument("--causal-min-longitudinal-delta", type=float, default=.10,
                        help="minimum throttle-minus-brake reduction for a 10m/s target drop")
    parser.add_argument("--anticipation-min-speed-drop-mps", type=float, default=.25)
    parser.add_argument("--anticipation-min-longitudinal-delta", type=float, default=.05,
                        help="minimum net-control reduction when only 20m+ targets drop")
    parser.add_argument("--causal-loss-weight", type=float, default=1.0)
    parser.add_argument("--causal-batch-size", type=int, default=8)
    parser.add_argument("--control-gate-deadline-epochs", type=int, default=5)
    args = parser.parse_args()
    if args.resume and args.init_planner:
        parser.error("--resume and --init-planner are mutually exclusive")
    if args.init_planner and args.planner_warmup_epochs != 0:
        parser.error("use --planner-warmup-epochs 0 with --init-planner")

    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from .data_v5 import TCPDatasetV5
    from .model_v6 import build_model_v6

    if dataset_class is None:
        dataset_class = TCPDatasetV5

    _seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = dataset_class(args.prepared, "train", augment=True)
    val_set = dataset_class(args.prepared, "val", augment=False)
    prepared_schema = json.loads((args.prepared / "schema.json").read_text(encoding="utf-8"))
    label_alignment = prepared_schema.get("control_label_alignment")
    if recipe_version >= 7:
        if not isinstance(label_alignment, dict) or label_alignment.get("delay_ms") != 50:
            raise RuntimeError(
                "v7 requires prepared controls aligned to exactly t+50ms")
    elif label_alignment is not None:
        raise RuntimeError("v6 training refuses delay-aligned prepared data; use train_v7.py")
    normalization = json.loads(
        (args.prepared / "normalization.json").read_text(encoding="utf-8"))
    stats = _tensor_stats(normalization, device)
    sampler = None
    if args.balanced_sampler:
        counts = {name: train_set.sample_buckets.count(name)
                  for name in set(train_set.sample_buckets)}
        available = sum(share for name, share in TARGET_SHARES.items() if counts.get(name, 0))
        weights = [TARGET_SHARES.get(name, 0.0) / available /
                   max(1, counts.get(name, 0)) for name in train_set.sample_buckets]
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
    probe_batch = _stratified_probe_batch(val_set, device)

    model = build_model_v6(
        args.imagenet_pretrained and args.resume is None and args.init_planner is None).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=.5)
    amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=32768.0)
    args.output.mkdir(parents=True, exist_ok=True)
    history, best, stale, start_epoch = [], float("inf"), 0, 1

    if args.init_planner:
        checkpoint = torch.load(args.init_planner, map_location="cpu", weights_only=False)
        if checkpoint.get("stage") != "planner_warmup":
            raise RuntimeError("--init-planner must point to planner_warmup.pt")
        if recipe_version >= 7:
            # V7 changes only control labels. Reuse the frozen V5 planner when
            # every shared planner/input artifact is byte-identical; the extra
            # V7 manifest hash intentionally differs and is checked for all
            # subsequent V7 resume/inference operations.
            old_artifacts = checkpoint.get("artifacts", {})
            shared = {name: value for name, value in prepared_schema["artifacts"].items()
                      if name != "manifest.parquet"}
            if old_artifacts != shared:
                raise RuntimeError("v5 planner/v7 shared artifact mismatch")
            if checkpoint.get("model_schema") != prepared_schema["model_schema"]:
                raise RuntimeError("v5 planner/v7 model schema mismatch")
        elif checkpoint.get("artifacts") != prepared_schema["artifacts"]:
            raise RuntimeError("v5 planner/prepared artifact mismatch")
        if checkpoint.get("normalization") != normalization:
            raise RuntimeError("v5 planner/prepared normalization mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        initial_probe = causal_probe_v6(model, probe_batch, stats, args.causal_drop_mps)
        if not planner_gate_passed(initial_probe, args.causal_min_speed_drop_mps):
            raise RuntimeError(f"initialized planner failed causal gate: {initial_probe}")
        print(f"[init-planner] checkpoint={args.init_planner} probe={initial_probe}", flush=True)
    elif args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("training_recipe", {}).get("version") != recipe_version:
            raise RuntimeError(f"resume checkpoint is not recipe v{recipe_version}")
        if checkpoint.get("artifacts") != prepared_schema["artifacts"]:
            raise RuntimeError("v6 resume/prepared artifact mismatch")
        if checkpoint.get("normalization") != normalization:
            raise RuntimeError("v6 resume/prepared normalization mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if amp and checkpoint.get("grad_scaler") is not None:
            scaler.load_state_dict(checkpoint["grad_scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history_path = args.output / "history.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        passed_rows = [row for row in history if row.get("full_causal_gate_passed")]
        if passed_rows:
            best = min(float(row["val"]["total"]) for row in passed_rows)
            best_epoch = max(int(row["epoch"]) for row in passed_rows
                             if float(row["val"]["total"]) == best)
            stale = int(checkpoint["epoch"]) - best_epoch
        print(f"[resume] checkpoint={args.resume} next_epoch={start_epoch} "
              f"best_causal_val={best}", flush=True)

    print(f"[dataset] train={len(train_set)} val={len(val_set)} "
          f"batches={len(train_loader)}+{len(val_loader)} probe={len(probe_batch['image'])} "
          f"device={device} amp={amp}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        planner_only = epoch <= args.planner_warmup_epochs
        stage = "planner_warmup" if planner_only else "joint_causal_predicted_plan"
        model.train()
        sums, batches, overflow = {}, 0, 0
        started = time.perf_counter()
        limit = min(len(train_loader), args.max_train_batches) if args.max_train_batches else len(train_loader)
        marks = {max(1, round(limit * fraction)) for fraction in (.25, .5, .75, 1.0)}
        for number, batch in enumerate(train_loader, 1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                visual = model.encode_image(batch["image"])
                pred = model.forward_from_visual(
                    visual, batch["ego"], batch["racing_line"], batch["command"],
                    batch["target_speed"])
                supervised, metrics = compute_losses(pred, batch, stats, planner_only)
                metrics["supervised_total"] = supervised
                loss = supervised
                if not planner_only:
                    count = min(args.causal_batch_size, len(batch["image"]))
                    changed_target = lowered_target(
                        batch["target_speed"][:count], stats, args.causal_drop_mps)
                    changed = model.forward_from_visual(
                        visual[:count], batch["ego"][:count], batch["racing_line"][:count],
                        batch["command"][:count], changed_target)
                    baseline = {key: value[:count] for key, value in pred.items()}
                    causal_loss, causal_metrics = causal_pair_metrics(
                        baseline, changed, {key: value[:count] for key, value in batch.items()},
                        stats, args.causal_min_speed_drop_mps,
                        args.causal_min_longitudinal_delta)
                    future_target = lowered_target(
                        batch["target_speed"][:count], stats, args.causal_drop_mps,
                        future_only=True)
                    future_changed = model.forward_from_visual(
                        visual[:count], batch["ego"][:count], batch["racing_line"][:count],
                        batch["command"][:count], future_target)
                    anticipation_loss, anticipation_metrics = causal_pair_metrics(
                        baseline, future_changed,
                        {key: value[:count] for key, value in batch.items()}, stats,
                        args.anticipation_min_speed_drop_mps,
                        args.anticipation_min_longitudinal_delta)
                    loss = supervised + args.causal_loss_weight * .5 * (
                        causal_loss + anticipation_loss)
                    metrics.update(causal_metrics)
                    metrics.update({f"anticipation_{name}": value
                                    for name, value in anticipation_metrics.items()})
                metrics["total"] = loss
            before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            overflow += int(scaler.get_scale() < before)
            for name, value in metrics.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach())
            batches += 1
            if number in marks:
                print(f"[epoch {epoch}/{args.epochs}] {stage} train {number}/{limit} "
                      f"loss={sums['total']/batches:.5f} elapsed={time.perf_counter()-started:.0f}s",
                      flush=True)
            if args.max_train_batches and number >= args.max_train_batches:
                break

        model.eval()
        val_sums, val_batches = {}, 0
        source_sums = {name: {} for name in ("normal", "synthetic_recovery", "dagger")}
        source_counts = {name: 0 for name in source_sums}
        with torch.no_grad():
            for number, batch in enumerate(val_loader, 1):
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                pred = model(batch["image"], batch["ego"], batch["racing_line"],
                             batch["command"], batch["target_speed"])
                _, metrics = compute_losses(pred, batch, stats, planner_only)
                for name, value in metrics.items():
                    val_sums[name] = val_sums.get(name, 0.0) + float(value)
                val_batches += 1
                for source, source_id in (("normal", 0), ("synthetic_recovery", 1),
                                          ("dagger", 2)):
                    mask = batch["source_id"] == source_id
                    count = int(mask.sum())
                    if count:
                        sub_pred = {key: value[mask] for key, value in pred.items()}
                        sub_batch = {key: value[mask] for key, value in batch.items()}
                        _, sub_metrics = compute_losses(sub_pred, sub_batch, stats, planner_only)
                        for name, value in sub_metrics.items():
                            source_sums[source][name] = (
                                source_sums[source].get(name, 0.0) + float(value) * count)
                        source_counts[source] += count
                if args.max_train_batches and number >= args.max_train_batches:
                    break
        scheduler.step()
        train_metrics = {name: value / max(1, batches) for name, value in sums.items()}
        val_metrics = {name: value / max(1, val_batches) for name, value in val_sums.items()}
        val_by_source = {
            source: {name: value / max(1, source_counts[source])
                     for name, value in metrics.items()}
            for source, metrics in source_sums.items()
        }
        probe = causal_probe_v6(model, probe_batch, stats, args.causal_drop_mps)
        planner_pass = planner_gate_passed(probe, args.causal_min_speed_drop_mps)
        full_pass = (not planner_only and full_gate_passed(
            probe, args.causal_min_speed_drop_mps, args.causal_min_longitudinal_delta,
            args.anticipation_min_speed_drop_mps,
            args.anticipation_min_longitudinal_delta))
        row = {
            "epoch": epoch, "stage": stage, "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics, "val": val_metrics,
            "val_by_source": val_by_source, "causal_probe": probe,
            "planner_causal_gate_passed": planner_pass,
            "full_causal_gate_passed": full_pass, "amp_overflow_steps": overflow,
        }
        history.append(row)
        print(json.dumps({
            "epoch": epoch, "stage": stage, "train_total": train_metrics["total"],
            "val_total": val_metrics["total"],
            "val_waypoint_mae_m": val_metrics["waypoint_mae_m"],
            "val_future_speed_mae_mps": val_metrics["future_speed_mae_mps"],
            "val_by_source_total": {name: values.get("total")
                                    for name, values in val_by_source.items()},
            "causal_probe": probe, "planner_gate": planner_pass,
            "full_gate": full_pass, "amp_overflow_steps": overflow,
        }), flush=True)
        state = {
            "epoch": epoch, "stage": stage, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "grad_scaler": scaler.state_dict() if amp else None,
            # Model/data contract is state-dict compatible with v5; recipe is
            # versioned separately so existing shadow/closed-loop stays exact.
            "model_schema": prepared_schema["model_schema"],
            "artifacts": prepared_schema["artifacts"], "normalization": normalization,
            "training_recipe": {
                "version": recipe_version,
                "control_label_alignment": label_alignment,
                "causal_drop_mps": args.causal_drop_mps,
                "causal_min_speed_drop_mps": args.causal_min_speed_drop_mps,
                "causal_min_longitudinal_delta": args.causal_min_longitudinal_delta,
                "anticipation_min_speed_drop_mps": args.anticipation_min_speed_drop_mps,
                "anticipation_min_longitudinal_delta":
                    args.anticipation_min_longitudinal_delta,
                "causal_loss_weight": args.causal_loss_weight,
            },
            "args": vars(args), "causal_probe": probe,
            "lap_phase": lap_phase_metadata(),
            "full_causal_gate_passed": full_pass, "val_loss": val_metrics["total"],
        }
        torch.save(state, args.output / "last.pt")
        (args.output / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")

        if epoch == args.planner_warmup_epochs and args.planner_warmup_epochs:
            torch.save(state, args.output / "planner_warmup.pt")
            print(f"[planner-causal-gate] passed={planner_pass} probe={probe}", flush=True)
            if not planner_pass:
                raise RuntimeError("v6 planner causal gate failed; joint training blocked")
        if not planner_only:
            joint_age = epoch - args.planner_warmup_epochs
            print(f"[full-causal-gate] passed={full_pass} joint_epoch={joint_age} "
                  f"deadline={args.control_gate_deadline_epochs}", flush=True)
            if full_pass:
                if val_metrics["total"] < best:
                    best, stale = val_metrics["total"], 0
                    torch.save(state, args.output / "best.pt")
                else:
                    stale += 1
            if joint_age >= args.control_gate_deadline_epochs and not full_pass:
                raise RuntimeError(
                    "v6 full causal gate failed by deadline; no closed-loop checkpoint saved")
            if best < float("inf") and args.early_stopping_patience and stale >= args.early_stopping_patience:
                print(f"[early-stop] no causal-valid validation improvement for {stale} epochs")
                break


if __name__ == "__main__":
    main()
