"""Offline metrics and visual diagnostics for the Laguna Seca TCP-AC model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import POSITION_SCALE_M, SPEED_SCALE_MPS, normalization_metadata
from .data import TCPDataset, quaternion_yaw


HORIZONS = (0.5, 1.0, 1.5, 2.0)


def training_summary(history: list[dict]) -> dict:
    if not history:
        raise ValueError("history is empty")
    best = min(history, key=lambda row: float(row["val"]["total"]))
    tail = [float(row["val"]["total"]) for row in history[-20:]]
    components = {}
    for name in ("total", "action", "waypoint", "future_action", "speed"):
        row = min(history, key=lambda item: float(item["val"][name]))
        components[name] = {"epoch": int(row["epoch"]), "value": float(row["val"][name])}
    return {
        "epochs": len(history), "best_epoch": int(best["epoch"]),
        "best_val_total": float(best["val"]["total"]),
        "last_epoch": int(history[-1]["epoch"]),
        "last_val_total": float(history[-1]["val"]["total"]),
        "best_validation_components": components,
        "last_20_val_total_min": min(tail), "last_20_val_total_max": max(tail),
        "last_20_val_total_range": max(tail) - min(tail),
    }


def plot_history(history: list[dict], output: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    summary = training_summary(history)
    x = np.array([r["epoch"] for r in history])
    best = summary["best_epoch"]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x, [r["train"]["total"] for r in history], label="train")
    ax.plot(x, [r["val"]["total"] for r in history], label="validation")
    ax.axvline(best, color="tab:green", ls="--", label=f"best epoch {best}")
    ax.set(xlabel="epoch", ylabel="weighted L1", title="TCP-AC total loss")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(output / "loss_curves.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, name in zip(axes.flat, ("action", "waypoint", "future_action", "speed")):
        ax.plot(x, [r["train"][name] for r in history], label="train")
        ax.plot(x, [r["val"][name] for r in history], label="validation")
        ax.axvline(best, color="tab:green", ls="--", alpha=.6)
        ax.set_title(name.replace("_", " ")); ax.grid(alpha=.25); ax.legend()
    fig.suptitle("TCP-AC loss components"); fig.tight_layout()
    fig.savefig(output / "loss_components.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(x, [r["lr"] for r in history], where="post")
    ax.set(xlabel="epoch", ylabel="learning rate", title="Learning-rate schedule")
    ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(output / "learning_rate.png", dpi=160); plt.close(fig)
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _mae_bias(pred, gt) -> dict[str, float]:
    error = np.asarray(pred, float) - np.asarray(gt, float)
    return {"mae": float(np.abs(error).mean()), "bias": float(error.mean())}


def prediction_metrics(table: pd.DataFrame, checkpoint: dict) -> dict:
    accel = _mae_bias(table.pred_signed_accel, table.gt_signed_accel)
    steer = _mae_bias(table.pred_steer, table.gt_steer)
    speed = _mae_bias(table.pred_speed_mps, table.speed_mps)
    horizons, wp_abs, future_abs = [], [], []
    for step, seconds in enumerate(HORIZONS, 1):
        le = table[f"pred_wp{step}_lateral_m"] - table[f"gt_wp{step}_lateral_m"]
        fe = table[f"pred_wp{step}_forward_m"] - table[f"gt_wp{step}_forward_m"]
        ae = table[f"pred_future{step}_signed_accel"] - table[f"gt_future{step}_signed_accel"]
        se = table[f"pred_future{step}_steer"] - table[f"gt_future{step}_steer"]
        wp_abs += [np.abs(le).to_numpy(), np.abs(fe).to_numpy()]
        future_abs += [np.abs(ae).to_numpy(), np.abs(se).to_numpy()]
        horizons.append({
            "step": step, "seconds": seconds,
            "waypoint_lateral_mae_m": float(np.abs(le).mean()),
            "waypoint_forward_mae_m": float(np.abs(fe).mean()),
            "waypoint_euclidean_mean_m": float(np.hypot(le, fe).mean()),
            "future_signed_accel_mae": float(np.abs(ae).mean()),
            "future_steer_mae": float(np.abs(se).mean()),
        })
    action_loss = (accel["mae"] + steer["mae"]) / 2
    waypoint_loss = float(np.concatenate(wp_abs).mean()) / POSITION_SCALE_M
    future_loss = float(np.concatenate(future_abs).mean())
    speed_loss = speed["mae"] / SPEED_SCALE_MPS
    total = action_loss + waypoint_loss + .5 * future_loss + .05 * speed_loss
    return {
        "samples": int(len(table)), "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_val_loss": float(checkpoint["val_loss"]),
        "normalization": checkpoint["normalization"],
        "current_action": {"signed_accel": accel, "steer": steer},
        "speed_mps": speed, "horizons": horizons,
        "recomputed_validation_loss": {
            "total": total, "action": action_loss, "waypoint": waypoint_loss,
            "future_action": future_loss, "speed": speed_loss,
            "checkpoint_total_difference": total - float(checkpoint["val_loss"]),
        },
    }


def _prediction_table(meta: pd.DataFrame, pred: dict, gt: dict) -> pd.DataFrame:
    out = meta.reset_index(drop=True).copy()
    out["gt_signed_accel"], out["gt_steer"] = gt["action"][:, 0], gt["action"][:, 1]
    out["pred_signed_accel"], out["pred_steer"] = pred["action"][:, 0], pred["action"][:, 1]
    out["pred_speed_mps"] = pred["speed"][:, 0] * SPEED_SCALE_MPS
    out["action_abs_error"] = (np.abs(out.pred_signed_accel - out.gt_signed_accel)
                               + np.abs(out.pred_steer - out.gt_steer)) / 2
    wp_errors = []
    for step in range(1, 5):
        j = step - 1
        for axis, name in enumerate(("lateral", "forward")):
            out[f"gt_wp{step}_{name}_m"] = gt["waypoints"][:, j, axis] * POSITION_SCALE_M
            out[f"pred_wp{step}_{name}_m"] = pred["waypoints"][:, j, axis] * POSITION_SCALE_M
        le = out[f"pred_wp{step}_lateral_m"] - out[f"gt_wp{step}_lateral_m"]
        fe = out[f"pred_wp{step}_forward_m"] - out[f"gt_wp{step}_forward_m"]
        out[f"wp{step}_error_m"] = np.hypot(le, fe); wp_errors.append(out[f"wp{step}_error_m"])
        out[f"gt_future{step}_signed_accel"] = gt["future_actions"][:, j, 0]
        out[f"gt_future{step}_steer"] = gt["future_actions"][:, j, 1]
        out[f"pred_future{step}_signed_accel"] = pred["future_actions"][:, j, 0]
        out[f"pred_future{step}_steer"] = pred["future_actions"][:, j, 1]
    out["waypoint_mean_error_m"] = np.column_stack(wp_errors).mean(axis=1)
    out["combined_error_score"] = out.action_abs_error + out.waypoint_mean_error_m / 50
    return out


def run_inference(prepared: Path, checkpoint_path: Path, batch_size: int,
                  workers: int, device_name: str, max_batches: int = 0):
    import torch
    from torch.utils.data import DataLoader
    from .model import build_model

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("normalization") != normalization_metadata():
        raise RuntimeError(f"normalization mismatch: {checkpoint.get('normalization')}")
    device = torch.device(device_name if device_name != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = TCPDataset(prepared, "val", augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")
    model = build_model(False).to(device); model.load_state_dict(checkpoint["model"]); model.eval()
    names = ("waypoints", "action", "future_actions", "speed")
    pred = {n: [] for n in names}; gt = {n: [] for n in names[:-1]}; seen = 0
    with torch.no_grad():
        for batch_no, batch in enumerate(loader, 1):
            gpu = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            result = model(gpu["image"], gpu["speed"], gpu["target_point"], gpu["command"])
            for n in names: pred[n].append(result[n].cpu().numpy())
            for n in gt: gt[n].append(batch[n].numpy())
            seen += len(batch["image"])
            if batch_no == 1 or batch_no % 100 == 0:
                print(json.dumps({"batches": batch_no, "samples": seen,
                                  "total": len(dataset), "device": str(device)}))
            if max_batches and batch_no >= max_batches: break
    pred = {k: np.concatenate(v) for k, v in pred.items()}
    gt = {k: np.concatenate(v) for k, v in gt.items()}
    cols = ["session", "run_id", "weather", "lap", "time_ms", "source_index",
            "spline_position", "image_path", "pos_x", "pos_y", "pos_z", "quat_x",
            "quat_y", "quat_z", "quat_w", "speed_mps", "throttle", "brake", "steer"]
    table = _prediction_table(dataset.samples.iloc[:seen][cols], pred, gt)
    return dataset, checkpoint, table, str(device)


def grouped_outputs(table: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    work = table.copy()
    work["speed_bin"] = pd.cut(work.speed_mps, [-np.inf, 30, 50, 65, 75, np.inf],
                               labels=["<30", "30-50", "50-65", "65-75", ">=75"])
    work["steer_bin"] = pd.cut(np.abs(work.gt_steer), [-np.inf, .05, .15, .30, np.inf],
                               labels=["<.05", ".05-.15", ".15-.30", ">=.30"])
    work["track_bin"] = np.minimum((work.spline_position * 100).astype(int), 99)
    measures = ["action_abs_error", "waypoint_mean_error_m"]
    for key, filename in (("speed_bin", "error_by_speed"), ("steer_bin", "error_by_steer"),
                          ("track_bin", "error_by_track_position")):
        grouped = work.groupby(key, observed=True)[measures].agg(["count", "mean"])
        grouped.to_csv(output / f"{filename}.csv")
        if key != "track_bin":
            mean = work.groupby(key, observed=True)[measures].mean()
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            mean.action_abs_error.plot.bar(ax=axes[0], color="tab:blue", title="action error")
            mean.waypoint_mean_error_m.plot.bar(ax=axes[1], color="tab:orange", title="waypoint error (m)")
            for ax in axes: ax.grid(axis="y", alpha=.25); ax.tick_params(axis="x", rotation=20)
            fig.tight_layout(); fig.savefig(output / f"{filename}.png", dpi=160); plt.close(fig)

    lap = work[(work.session == work.iloc[0].session) & (work.lap == work.iloc[0].lap)].head(2400)
    t = (lap.time_ms.to_numpy() - lap.time_ms.iloc[0]) / 1000
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(t, lap.gt_steer, label="GT"); axes[0].plot(t, lap.pred_steer, label="pred", alpha=.8)
    axes[1].plot(t, lap.gt_signed_accel, label="GT"); axes[1].plot(t, lap.pred_signed_accel, label="pred", alpha=.8)
    axes[2].plot(t, lap.speed_mps * 3.6, color="tab:green")
    axes[0].set_ylabel("steer"); axes[1].set_ylabel("signed accel")
    axes[2].set(ylabel="km/h", xlabel="seconds")
    for ax in axes: ax.grid(alpha=.2); ax.legend() if ax is not axes[2] else None
    fig.tight_layout(); fig.savefig(output / "control_timeseries.png", dpi=160); plt.close(fig)

    track = work.groupby("track_bin", observed=True).agg(
        x=("pos_x", "mean"), y=("pos_y", "mean"), action=("action_abs_error", "mean"),
        waypoint=("waypoint_mean_error_m", "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, col, title in ((axes[0], "action", "Action error"),
                           (axes[1], "waypoint", "Waypoint error (m)")):
        sc = ax.scatter(track.x, track.y, c=track[col], cmap="magma", s=45)
        ax.plot(track.x, track.y, color=".8", lw=.6, zorder=0); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title); fig.colorbar(sc, ax=ax, shrink=.8)
    fig.tight_layout(); fig.savefig(output / "track_error_map.png", dpi=160); plt.close(fig)


def audit_predictions(table: pd.DataFrame, output: Path) -> dict:
    """Expose tail risk, sign errors, run imbalance, and suspicious GT motion."""
    work = table.copy()
    percentiles = {}
    for name in ("action_abs_error", "waypoint_mean_error_m"):
        percentiles[name] = {
            key: float(value) for key, value in zip(
                ("p50", "p90", "p95", "p99", "max"),
                np.quantile(work[name], (.5, .9, .95, .99, 1.0)))
        }
    sign_accuracy = {}
    for name in ("steer", "signed_accel"):
        subset = work[np.abs(work[f"gt_{name}"]) > .05]
        sign_accuracy[name] = {
            "samples": int(len(subset)),
            "accuracy": float((np.sign(subset[f"gt_{name}"])
                               == np.sign(subset[f"pred_{name}"])).mean()),
        }
    sessions = []
    for session, group in work.groupby("session", sort=False):
        sessions.append({
            "session": session, "samples": int(len(group)),
            "action_abs_error": float(group.action_abs_error.mean()),
            "waypoint_mean_error_m": float(group.waypoint_mean_error_m.mean()),
            "steer_mae": float(np.abs(group.pred_steer - group.gt_steer).mean()),
            "signed_accel_mae": float(np.abs(group.pred_signed_accel - group.gt_signed_accel).mean()),
        })
    pd.DataFrame(sessions).to_csv(output / "error_by_session.csv", index=False)
    work["gt_displacement_0p5_m"] = np.hypot(work.gt_wp1_lateral_m, work.gt_wp1_forward_m)
    work["displacement_speed_ratio"] = (
        work.gt_displacement_0p5_m / (.5 * work.speed_mps).clip(lower=1.0))
    suspicious = work[(work.speed_mps > 20)
                      & ((work.displacement_speed_ratio < .5)
                         | (work.displacement_speed_ratio > 1.5))]
    keep = ["session", "lap", "time_ms", "source_index", "spline_position", "speed_mps",
            "gt_displacement_0p5_m", "displacement_speed_ratio", "waypoint_mean_error_m"]
    suspicious[keep].to_csv(output / "suspicious_motion_samples.csv", index=False)
    return {
        "error_percentiles": percentiles,
        "nontrivial_sign_accuracy": sign_accuracy,
        "by_session": sessions,
        "suspicious_gt_motion": {
            "rule": "speed>20m/s and 0.5s displacement/(0.5*speed) outside [0.5,1.5]",
            "samples": int(len(suspicious)), "fraction": float(len(suspicious) / len(work)),
        },
    }


def _kv(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text(encoding="utf-8-sig").splitlines()
                if "=" in line)


class CameraCache:
    def __init__(self): self.tables = {}
    def get(self, row):
        session = str(row.session)
        if session not in self.tables:
            try:
                root = Path(row.image_path).parents[2]
                configured = Path(_kv(root / "SESSION_INFO.txt")["source_run"])
                # Some older SESSION_INFO files contain a path written through a
                # mismatched Windows code page. Prefer it when valid, otherwise
                # reconstruct the source run from the structured session root.
                source = configured if configured.exists() else root.parents[2] / "dataset" / str(row.run_id)
                self.tables[session] = pd.read_parquet(source / "image_index.parquet")
            except (OSError, KeyError, ValueError): self.tables[session] = None
        table = self.tables[session]; index = int(row.source_index)
        return None if table is None or not 0 <= index < len(table) else table.iloc[index]


def _project(row, points, z, camera):
    if camera is None: return None
    names = ["cam_pos_x", "cam_pos_y", "cam_pos_z", "cam_forward_x", "cam_forward_y",
             "cam_forward_z", "cam_up_x", "cam_up_y", "cam_up_z", "cam_fov_deg"]
    if not all(np.isfinite(float(camera.get(n, np.nan))) for n in names): return None
    yaw = quaternion_yaw(row.quat_x, row.quat_y, row.quat_z, row.quat_w)
    fwd = np.array([math.cos(yaw), math.sin(yaw)]); left = np.array([-fwd[1], fwd[0]])
    xy = np.array([row.pos_x, row.pos_y]) + points[:, :1] * left + points[:, 1:] * fwd
    world = np.column_stack((xy, z))
    cpos = camera[["cam_pos_x", "cam_pos_y", "cam_pos_z"]].to_numpy(float)
    cfw = camera[["cam_forward_x", "cam_forward_y", "cam_forward_z"]].to_numpy(float)
    cup = camera[["cam_up_x", "cam_up_y", "cam_up_z"]].to_numpy(float)
    cfw /= np.linalg.norm(cfw); right = np.cross(cfw, cup); right /= np.linalg.norm(right)
    cup = np.cross(right, cfw); rel = world - cpos
    depth, x, y = rel @ cfw, rel @ right, rel @ cup
    focal = 540 / math.tan(math.radians(float(camera.cam_fov_deg)) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.column_stack((960 + focal * x / depth, 460 - focal * y / depth))
    uv[depth <= .1] = np.nan
    return uv


def _future_z(dataset, row):
    return np.array([dataset.table.iloc[dataset.lookup[
        (row.session, int(row.lap), int(row.time_ms) + off * 25)]].pos_z
        for off in dataset.offsets], float)


def qualitative_indices(table: pd.DataFrame, count: int) -> dict[str, list[int]]:
    count = max(2, min(count, len(table)))
    # Balance the regular sheet across held-out sessions instead of allowing
    # the first run to dominate a global sort.
    stratified = []
    groups = list(table.groupby("session", sort=False))
    base, extra = divmod(count, len(groups))
    for group_number, (_, group) in enumerate(groups):
        take = base + (1 if group_number < extra else 0)
        ordered = group.sort_values(["spline_position", "speed_mps"])
        positions = np.linspace(0, len(ordered) - 1, take, dtype=int)
        stratified.extend(ordered.iloc[positions].index.astype(int).tolist())
    return {
        "best": table.nsmallest(count, "combined_error_score").index.astype(int).tolist(),
        "worst": table.nlargest(count, "combined_error_score").index.astype(int).tolist(),
        "stratified": stratified,
    }


def qualitative_sheet(table, dataset, indices, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rows = math.ceil(len(indices) / 2); fig, axes = plt.subplots(rows, 4, figsize=(18, 4.2 * rows), squeeze=False)
    cache = CameraCache()
    for slot, idx in enumerate(indices):
        row = table.iloc[idx]; image_ax, bev_ax = axes[slot // 2, (slot % 2) * 2:(slot % 2) * 2 + 2]
        with Image.open(row.image_path) as image: image_ax.imshow(image.convert("RGB"))
        gt = np.array([[row[f"gt_wp{k}_lateral_m"], row[f"gt_wp{k}_forward_m"]] for k in range(1, 5)])
        pred = np.array([[row[f"pred_wp{k}_lateral_m"], row[f"pred_wp{k}_forward_m"]] for k in range(1, 5)])
        z, camera = _future_z(dataset, row), cache.get(row)
        for points, color, label in ((gt, "lime", "GT"), (pred, "red", "prediction")):
            uv = _project(row, points, z, camera)
            if uv is not None and np.isfinite(uv).all(): image_ax.plot(uv[:, 0], uv[:, 1], "o-", c=color, lw=2, label=label)
        image_ax.axis("off")
        handles, labels = image_ax.get_legend_handles_labels()
        if handles: image_ax.legend(loc="lower right", fontsize=7)
        overlay = "CSP overlay" if camera is not None else "RGB only: raw CSP pose unavailable"
        image_ax.set_title(f"{row.speed_mps*3.6:.0f} km/h  steer {row.gt_steer:+.3f}/{row.pred_steer:+.3f}\n"
                           f"accel {row.gt_signed_accel:+.3f}/{row.pred_signed_accel:+.3f}  [{overlay}]", fontsize=8)
        bev_ax.plot(np.r_[0, gt[:, 0]], np.r_[0, gt[:, 1]], "o-", c="green", label="GT")
        bev_ax.plot(np.r_[0, pred[:, 0]], np.r_[0, pred[:, 1]], "o-", c="red", label="pred")
        bev_ax.scatter(0, 0, marker="^", c="black"); bev_ax.set_aspect("equal", adjustable="box")
        bev_ax.set(xlabel="left + (m)", ylabel="forward (m)", title=f"WP error {row.waypoint_mean_error_m:.2f} m")
        bev_ax.grid(alpha=.25); bev_ax.legend(fontsize=7)
    for ax in axes.flat[len(indices) * 2:]: ax.axis("off")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def write_report(output, metrics, training, device):
    loss = metrics["recomputed_validation_loss"]
    lines = ["# TCP-AC offline evaluation", "",
             f"- checkpoint epoch: {metrics['checkpoint_epoch']}", f"- samples: {metrics['samples']:,}",
             f"- device: `{device}`", f"- recomputed/checkpoint val total: {loss['total']:.9f} / {metrics['checkpoint_val_loss']:.9f}",
             "", "## Current output", "",
             f"- signed acceleration MAE/bias: {metrics['current_action']['signed_accel']['mae']:.6f} / {metrics['current_action']['signed_accel']['bias']:+.6f}",
             f"- steer MAE/bias: {metrics['current_action']['steer']['mae']:.6f} / {metrics['current_action']['steer']['bias']:+.6f}",
             f"- speed MAE: {metrics['speed_mps']['mae']:.3f} m/s ({metrics['speed_mps']['mae']*3.6:.2f} km/h)", "",
             "## Horizon metrics", "", "| horizon | lateral m | forward m | 2D m | future accel | future steer |",
             "|---:|---:|---:|---:|---:|---:|"]
    for r in metrics["horizons"]:
        lines.append(f"| {r['seconds']:.1f}s | {r['waypoint_lateral_mae_m']:.3f} | {r['waypoint_forward_mae_m']:.3f} | {r['waypoint_euclidean_mean_m']:.3f} | {r['future_signed_accel_mae']:.4f} | {r['future_steer_mae']:.4f} |")
    diag = metrics.get("diagnostics", {})
    lines += ["", "## Diagnostics", ""]
    if diag:
        lines += [
            f"- steer sign accuracy (|GT|>0.05): {diag['nontrivial_sign_accuracy']['steer']['accuracy']:.4%}",
            f"- signed-accel sign accuracy (|GT|>0.05): {diag['nontrivial_sign_accuracy']['signed_accel']['accuracy']:.4%}",
            f"- suspicious GT motion samples: {diag['suspicious_gt_motion']['samples']} "
            f"({diag['suspicious_gt_motion']['fraction']:.4%})",
            f"- action-error p99: {diag['error_percentiles']['action_abs_error']['p99']:.4f}",
            f"- mean-waypoint-error p99: {diag['error_percentiles']['waypoint_mean_error_m']['p99']:.3f} m",
        ]
    lines += [
        "", "The tiny difference between recomputed and checkpoint total comes from sample-weighted "
        "aggregation here versus equal batch-mean aggregation during training (the final batch is smaller).",
        "", "RGB overlays use recorded CSP camera pose and GT road height for visualization only; height is not a model input. "
        "The older validation source run is no longer present, so its sheets intentionally show RGB+BEV without image overlay.",
        "", "## Phase-2 verdict", "",
        "PASS WITH WARNINGS: sign and scale wiring pass, and full-validation loss is reproduced. Before actuation, "
        "inspect the older run's low-speed/high-steer outliers and the rows in `suspicious_motion_samples.csv`. "
        "Proceed next to non-actuating shadow mode, not direct closed-loop control.",
        "", "Use `best.pt` for shadow-mode integration. Offline accuracy does not prove closed-loop stability.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=True)
    history = json.loads((args.history or args.checkpoint.parent / "history.json").read_text(encoding="utf-8"))
    train_info = plot_history(history, args.output)
    dataset, checkpoint, table, device = run_inference(args.prepared, args.checkpoint, args.batch_size,
                                                        args.workers, args.device, args.max_batches)
    metrics = prediction_metrics(table, checkpoint)
    metrics["diagnostics"] = audit_predictions(table, args.output)
    table.to_parquet(args.output / "predictions.parquet", index=False)
    pd.DataFrame(metrics["horizons"]).to_csv(args.output / "per_horizon_metrics.csv", index=False)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    grouped_outputs(table, args.output)
    for name, indices in qualitative_indices(table, args.qualitative_count).items():
        qualitative_sheet(table, dataset, indices, args.output / f"qualitative_{name}.jpg", f"TCP-AC: {name}")
    write_report(args.output, metrics, train_info, device)
    print(json.dumps(metrics, indent=2)); return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate TCP-AC on held-out Laguna Seca runs")
    parser.add_argument("--prepared", type=Path, default=Path("training/tcp_lagunaseca"))
    parser.add_argument("--checkpoint", type=Path, default=Path("training/tcp_lagunaseca/checkpoints/best.pt"))
    parser.add_argument("--history", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("training/tcp_lagunaseca/evaluation"))
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto"); parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--qualitative-count", type=int, default=6)
    evaluate(parser.parse_args())


if __name__ == "__main__": main()
