"""Visual and numeric audit of v5 speed, waypoint and train/val inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EGO_FEATURE_NAMES, FUTURE_OFFSETS_FRAMES, PLANNING_HORIZONS_SECONDS
from .data import quaternion_yaw, racing_line_columns, to_local
from .prepare_v5 import _future_values, target_speed_columns_v5
from .profile_v5 import ExpertSpeedProfileV5
from .train_v5 import TARGET_SHARES


def _fill(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, float).copy()
    good = np.isfinite(result)
    if not good.any():
        raise ValueError("no finite bins")
    x = np.arange(len(result), dtype=float)
    if not good.all():
        result[~good] = np.interp(
            x[~good], np.concatenate((x[good] - len(result), x[good], x[good] + len(result))),
            np.tile(result[good], 3))
    return result


def _bin_curve(table: pd.DataFrame, bins: int, reducer: str) -> np.ndarray:
    indices = np.minimum((np.mod(table.spline_position.to_numpy(float), 1) * bins).astype(int),
                         bins - 1)
    speed = np.abs(table.speed_mps.to_numpy(float))
    result = np.full(bins, np.nan)
    for index in range(bins):
        values = speed[indices == index]
        if len(values):
            result[index] = np.median(values) if reducer == "median" else np.mean(values)
    return _fill(result)


def _lookup_curve(curve: np.ndarray, spline: np.ndarray) -> np.ndarray:
    bins = len(curve)
    centers = (np.arange(bins) + .5) / bins
    return np.interp(np.mod(spline, 1), np.concatenate((centers - 1, centers, centers + 1)),
                     np.tile(curve, 3))


def _metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    error = np.asarray(predicted) - np.asarray(actual)
    return {
        "mae_mps": float(np.mean(np.abs(error))),
        "rmse_mps": float(np.sqrt(np.mean(error ** 2))),
        "bias_mps": float(np.mean(error)),
        "p95_abs_mps": float(np.quantile(np.abs(error), .95)),
        "correlation": float(np.corrcoef(predicted, actual)[0, 1]),
    }


def _normalized_family(table: pd.DataFrame, normalization: dict) -> dict[str, np.ndarray]:
    result = {}
    ego = table[list(EGO_FEATURE_NAMES)].to_numpy(np.float32)
    spec = normalization["ego"]
    result["ego"] = (ego - np.asarray(spec["mean"])) / np.asarray(spec["std"])
    target = table[target_speed_columns_v5()].to_numpy(np.float32)
    target = np.column_stack((target, target[:, 0] - np.abs(table.speed_mps.to_numpy(float))))
    spec = normalization["target_speed_features_mps"]
    result["target_speed"] = (target - np.asarray(spec["mean"])) / np.asarray(spec["std"])

    yaw = np.asarray([quaternion_yaw(*q) for q in table[
        ["quat_x", "quat_y", "quat_z", "quat_w"]].itertuples(index=False, name=None)])
    origin = table[["pos_x", "pos_y"]].to_numpy(np.float32)
    line = np.empty((len(table), len(racing_line_columns()), 2), np.float32)
    for point, (x, y) in enumerate(racing_line_columns()):
        world = table[[x, y]].to_numpy(np.float32)
        dx, dy = world[:, 0] - origin[:, 0], world[:, 1] - origin[:, 1]
        line[:, point, 0] = -np.sin(yaw) * dx + np.cos(yaw) * dy
        line[:, point, 1] = np.cos(yaw) * dx + np.sin(yaw) * dy
    spec = normalization["racing_line_local_m"]
    result["racing_line"] = ((line - np.asarray(spec["mean"])) /
                             np.asarray(spec["std"])).reshape(len(table), -1)
    return result


def _waypoint_speed_audit(table: pd.DataFrame, mask: np.ndarray) -> dict[str, object]:
    current_x = table.pos_x.to_numpy(float)
    current_y = table.pos_y.to_numpy(float)
    current_speed = np.abs(table.speed_mps.to_numpy(float))
    previous_x, previous_y, previous_speed = current_x, current_y, current_speed
    previous_valid = np.isfinite(current_x) & np.isfinite(current_y) & np.isfinite(current_speed)
    endpoint_mae, trapezoid_mae, valid_counts = [], [], []
    endpoint_errors, trapezoid_errors = [], []
    previous_t = 0.0
    for offset, horizon in zip(FUTURE_OFFSETS_FRAMES, PLANNING_HORIZONS_SECONDS):
        future_x = _future_values(table, "pos_x", offset).astype(float)
        future_y = _future_values(table, "pos_y", offset).astype(float)
        future_speed = np.abs(_future_values(table, "speed_mps", offset).astype(float))
        current_valid = np.isfinite(future_x) & np.isfinite(future_y) & np.isfinite(future_speed)
        valid = mask & previous_valid & current_valid
        segment_speed = np.hypot(future_x - previous_x, future_y - previous_y) / (horizon - previous_t)
        endpoint_error = segment_speed - future_speed
        trapezoid_target = .5 * (previous_speed + future_speed)
        trapezoid_error = segment_speed - trapezoid_target
        endpoint_mae.append(float(np.mean(np.abs(endpoint_error[valid]))))
        trapezoid_mae.append(float(np.mean(np.abs(trapezoid_error[valid]))))
        valid_counts.append(int(valid.sum()))
        endpoint_errors.append(endpoint_error[valid])
        trapezoid_errors.append(trapezoid_error[valid])
        previous_x, previous_y, previous_speed = future_x, future_y, future_speed
        previous_valid = current_valid
        previous_t = horizon
    return {
        "horizons_s": list(PLANNING_HORIZONS_SECONDS),
        "valid_counts": valid_counts,
        "endpoint_speed_mae_mps": endpoint_mae,
        "trapezoid_average_speed_mae_mps": trapezoid_mae,
        "overall_endpoint_mae_mps": float(np.mean(np.abs(np.concatenate(endpoint_errors)))),
        "overall_trapezoid_mae_mps": float(np.mean(np.abs(np.concatenate(trapezoid_errors)))),
    }


def run(prepared: Path, output: Path, image_samples: int = 48) -> dict[str, object]:
    import matplotlib.pyplot as plt
    from PIL import Image
    from .data import preprocess_rgb

    output.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(prepared / "manifest.parquet")
    split = json.loads((prepared / "split.json").read_text(encoding="utf-8"))
    normalization = json.loads((prepared / "normalization.json").read_text(encoding="utf-8"))
    train_groups, val_groups = set(split["train_groups"]), set(split["val_groups"])
    train_mask = table.trajectory_id.isin(train_groups).to_numpy() & table.eligible.to_numpy(bool)
    val_mask = table.trajectory_id.isin(val_groups).to_numpy() & table.eligible.to_numpy(bool)
    profile = ExpertSpeedProfileV5(prepared / "expert_speed_profile_v5.npz")

    profile_comparison = {}
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    x = (np.arange(profile.bins) + .5) / profile.bins
    for axis, phase in zip(axes, ("standing", "flying")):
        train_normal = table[train_mask & table.source_type.eq("normal").to_numpy() &
                             table.lap_phase.eq(phase).to_numpy()]
        val_normal = table[val_mask & table.source_type.eq("normal").to_numpy() &
                           table.lap_phase.eq(phase).to_numpy()]
        median = _bin_curve(train_normal, profile.bins, "median")
        mean = _bin_curve(train_normal, profile.bins, "mean")
        actual = np.abs(val_normal.speed_mps.to_numpy(float))
        spline = val_normal.spline_position.to_numpy(float)
        median_metrics = _metrics(_lookup_curve(median, spline), actual)
        mean_metrics = _metrics(_lookup_curve(mean, spline), actual)
        profile_comparison[phase] = {"median": median_metrics, "mean": mean_metrics}
        axis.plot(x, median * 3.6, label="train-bin median / v5", lw=1.5)
        axis.plot(x, mean * 3.6, label="train-bin mean", lw=1, alpha=.75)
        val_curve = _bin_curve(val_normal, profile.bins, "median")
        axis.plot(x, val_curve * 3.6, label="held-out val median", lw=1, alpha=.8)
        axis.set_ylabel(f"{phase} km/h")
        axis.grid(alpha=.25)
        axis.legend(ncol=3)
    axes[-1].set_xlabel("normalized spline position")
    fig.suptitle("V5 expert-speed profile: train-only profile vs held-out validation")
    fig.tight_layout()
    fig.savefig(output / "01_speed_profile_train_val.png", dpi=160)
    plt.close(fig)

    grid = np.linspace(0, 1, 1000, endpoint=False)
    lookahead_profiles = np.stack([profile.speeds(s, "flying") for s in grid])
    fig, axis = plt.subplots(figsize=(14, 5))
    for point in (0, 3, 6, 8, 10):
        axis.plot(grid, lookahead_profiles[:, point] * 3.6,
                  label=f"+{int(profile.line_dist[0] * 0 + [0,5,10,20,30,40,50,75,100,125,150][point])}m")
    axis.set(title="Distance-based target speed inputs", xlabel="current spline position",
             ylabel="target speed (km/h)")
    axis.grid(alpha=.25); axis.legend(ncol=5)
    fig.tight_layout(); fig.savefig(output / "02_target_speed_lookaheads.png", dpi=160)
    plt.close(fig)

    waypoint_speed = {
        "train": _waypoint_speed_audit(table, train_mask),
        "val": _waypoint_speed_audit(table, val_mask),
        "by_source": {
            split_name: {
                source: _waypoint_speed_audit(
                    table, split_mask & table.source_type.eq(source).to_numpy())
                for source in ("normal", "synthetic_recovery", "dagger")
            }
            for split_name, split_mask in (("train", train_mask), ("val", val_mask))
        },
    }
    fig, axis = plt.subplots(figsize=(9, 5))
    horizons = np.asarray(PLANNING_HORIZONS_SECONDS)
    for split_name, style in (("train", "-"), ("val", "--")):
        axis.plot(horizons, waypoint_speed[split_name]["endpoint_speed_mae_mps"],
                  style, marker="o", label=f"{split_name}: endpoint speed")
        axis.plot(horizons, waypoint_speed[split_name]["trapezoid_average_speed_mae_mps"],
                  style, marker="s", label=f"{split_name}: segment-average speed")
    axis.set(title="Waypoint segment displacement vs speed labels",
             xlabel="future waypoint time (s)", ylabel="MAE (m/s)")
    axis.grid(alpha=.25); axis.legend()
    fig.tight_layout(); fig.savefig(output / "03_waypoint_speed_consistency.png", dpi=160)
    plt.close(fig)

    distributions = {}
    sample_cap = 60000
    rng = np.random.default_rng(42)
    selected = {}
    for name, mask in (("train", train_mask), ("val", val_mask)):
        indices = np.flatnonzero(mask)
        if len(indices) > sample_cap:
            indices = rng.choice(indices, sample_cap, replace=False)
        selected[name] = table.iloc[np.sort(indices)]
        families = _normalized_family(selected[name], normalization)
        distributions[name] = {
            family: {
                "mean_abs_max": float(np.max(np.abs(values.mean(axis=0)))),
                "std_min": float(values.std(axis=0).min()),
                "std_max": float(values.std(axis=0).max()),
                "p99_abs_max": float(np.max(np.quantile(np.abs(values), .99, axis=0))),
                "finite": bool(np.isfinite(values).all()),
            } for family, values in families.items()
        }
    families = ("ego", "racing_line", "target_speed")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    width = .35; positions = np.arange(len(families))
    for offset, name in ((-.5, "train"), (.5, "val")):
        axes[0].bar(positions + offset * width,
                    [distributions[name][f]["mean_abs_max"] for f in families],
                    width, label=name)
        axes[1].bar(positions + offset * width,
                    [distributions[name][f]["std_max"] for f in families],
                    width, label=name)
    for axis, title in zip(axes, ("maximum |normalized mean|", "maximum normalized std")):
        axis.set_xticks(positions, families); axis.set_title(title); axis.grid(axis="y", alpha=.25)
        axis.legend()
    fig.tight_layout(); fig.savefig(output / "04_input_train_val_distribution.png", dpi=160)
    plt.close(fig)

    # Draw representative pre-brake/braking/corner examples.  Line colors are
    # distance-profile targets; waypoint annotations are time-based GT speeds.
    candidates = []
    for split_name, groups in (("train", train_groups), ("val", val_groups)):
        part = table[table.trajectory_id.isin(groups) & table.source_type.eq("normal")]
        for bucket in ("normal_pre_brake", "normal_braking", "normal_corner"):
            rows = part[part.sampling_bucket.eq(bucket)]
            if len(rows):
                candidates.append((split_name, bucket, rows.iloc[len(rows) // 2]))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    lookup = {(str(r.trajectory_id), int(r.time_ms)): r
              for r in table.itertuples(index=False)}
    for axis, (split_name, bucket, row) in zip(axes.flat, candidates[:6]):
        yaw = quaternion_yaw(row.quat_x, row.quat_y, row.quat_z, row.quat_w)
        origin = np.asarray([row.pos_x, row.pos_y], np.float32)
        world_line = np.asarray([[getattr(row, x), getattr(row, y)]
                                 for x, y in racing_line_columns()], np.float32)
        local_line = to_local(world_line, origin, yaw)
        target_speeds = np.asarray([getattr(row, name) for name in target_speed_columns_v5()])
        future, future_speeds = [], []
        for offset in FUTURE_OFFSETS_FRAMES:
            found = lookup.get((str(row.trajectory_id), int(row.time_ms) + offset * 25), row)
            future.append([found.pos_x, found.pos_y]); future_speeds.append(abs(found.speed_mps))
        local_future = to_local(np.asarray(future, np.float32), origin, yaw)
        scatter = axis.scatter(local_line[:, 0], local_line[:, 1], c=target_speeds * 3.6,
                               cmap="viridis", marker="s", label="line + target speed")
        axis.plot(local_future[:, 0], local_future[:, 1], "r.-", label="future waypoints")
        for point, horizon, speed in zip(local_future, PLANNING_HORIZONS_SECONDS, future_speeds):
            axis.annotate(f"{horizon:g}s\n{speed*3.6:.0f}", point, fontsize=7)
        axis.set_title(f"{split_name} / {bucket}")
        axis.set_xlabel("right (m)"); axis.set_ylabel("forward (m)"); axis.axis("equal")
        axis.grid(alpha=.2)
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label="distance target speed (km/h)")
    fig.savefig(output / "05_waypoints_and_target_speeds_examples.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)

    image_report = {"requested": image_samples, "by_split": {}}
    for split_name, mask, seed in (("train", train_mask, 42), ("val", val_mask, 43)):
        requested = image_samples // 2
        rows = table[mask].sample(min(requested, int(mask.sum())), random_state=seed)
        image_values, failed = [], 0
        for row in rows.itertuples(index=False):
            try:
                tensor = preprocess_rgb(Image.open(row.image_path).convert("RGB")).numpy()
                image_values.append(tensor)
            except (OSError, PermissionError):
                failed += 1
        split_report = {"requested": requested, "loaded": len(image_values), "failed": failed}
        if image_values:
            stacked = np.stack(image_values)
            split_report.update({
                "shape": list(stacked.shape), "finite": bool(np.isfinite(stacked).all()),
                "channel_mean": stacked.mean(axis=(0, 2, 3)).tolist(),
                "channel_std": stacked.std(axis=(0, 2, 3)).tolist(),
                "min": float(stacked.min()), "max": float(stacked.max()),
            })
        image_report["by_split"][split_name] = split_report

    report = {
        "profile_reducer_used": "median per 1/500 spline bin",
        "profile_comparison": profile_comparison,
        "waypoint_speed_consistency": waypoint_speed,
        "normalized_input_distribution": distributions,
        "image_preprocessing_sample": image_report,
        "split": {
            "train_rows": int(train_mask.sum()), "val_rows": int(val_mask.sum()),
            "overlap": len(train_groups & val_groups),
            "train_by_source": table[train_mask].source_type.value_counts().to_dict(),
            "val_by_source": table[val_mask].source_type.value_counts().to_dict(),
            "train_by_phase": table[train_mask].lap_phase.value_counts().to_dict(),
            "val_by_phase": table[val_mask].lap_phase.value_counts().to_dict(),
        },
        "sampler_target_shares": TARGET_SHARES,
    }
    (output / "diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, default=Path("training/tcp_lagunaseca_v5"))
    parser.add_argument("--output", type=Path,
                        default=Path("training/tcp_lagunaseca_v5/input_diagnostics"))
    parser.add_argument("--image-samples", type=int, default=48)
    args = parser.parse_args()
    report = run(args.prepared, args.output, args.image_samples)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
