"""Prepare v5 from the audited v4 cohort while replacing its invalid speed signal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EGO_FEATURE_NAMES, FUTURE_OFFSETS_FRAMES, RACING_LINE_LOOKAHEADS_M
from .config_v5 import model_schema_v5
from .data import quaternion_yaw, racing_line_columns
from .profile_v5 import ExpertSpeedProfileV5, build_expert_profile, sha256_file


def target_speed_columns_v5() -> list[str]:
    return [f"expert_target_speed_{int(x)}m_mps" for x in RACING_LINE_LOOKAHEADS_M]


def _future_values(table: pd.DataFrame, column: str, offset: int) -> np.ndarray:
    lookup = pd.Series(
        table[column].to_numpy(),
        index=pd.MultiIndex.from_arrays(
            [table.trajectory_id.astype(str), table.time_ms.astype(np.int64)]),
    )
    requested = pd.MultiIndex.from_arrays([
        table.trajectory_id.astype(str),
        table.time_ms.astype(np.int64) + int(offset) * 25,
    ])
    return lookup.reindex(requested).to_numpy()


def _transition_buckets(table: pd.DataFrame) -> pd.Series:
    result = table.sampling_bucket.astype(str).copy()
    normal = table.source_type.eq("normal")
    future_throttle = np.column_stack([
        _future_values(table, "throttle", offset) for offset in (10, 20, 40)
    ])
    future_brake = np.column_stack([
        _future_values(table, "brake", offset) for offset in (10, 20, 40)
    ])
    future_steer = np.column_stack([
        _future_values(table, "steer", offset) for offset in (10, 20, 40)
    ])
    throttle = table.throttle.to_numpy(float)
    brake = table.brake.to_numpy(float)
    steer = np.abs(table.steer.to_numpy(float))
    minimum_throttle = np.min(
        np.where(np.isfinite(future_throttle), future_throttle, np.inf), axis=1)
    maximum_brake = np.max(
        np.where(np.isfinite(future_brake), future_brake, -np.inf), axis=1)
    minimum_brake = np.min(
        np.where(np.isfinite(future_brake), future_brake, np.inf), axis=1)
    maximum_steer = np.max(
        np.where(np.isfinite(future_steer), np.abs(future_steer), -np.inf), axis=1)
    labels = np.full(len(table), "normal_straight", dtype=object)
    labels[(throttle >= .90) & (brake < .05) & (steer < .08)] = "normal_full_throttle"
    labels[steer >= .08] = "normal_corner"
    labels[brake > .10] = "normal_braking"
    labels[(brake > .10) & (minimum_brake < .05) & (maximum_steer > .10)] = (
        "normal_brake_release_turnin")
    labels[(brake <= .10) & (maximum_brake > .10)] = "normal_pre_brake"
    labels[(brake <= .10) & (maximum_brake <= .10) & (throttle > .55) &
           (minimum_throttle < throttle - .25)] = "normal_pre_lift"
    result.loc[normal] = labels[normal.to_numpy()]
    return result


def _mask_synthetic_perturbation(table: pd.DataFrame,
                                 transition_margin_s: float = .10) -> tuple[pd.DataFrame, dict]:
    """Never train on the deliberate steer-away portion of synthetic recovery.

    Synthetic episodes first override steering for recovery_perturb_seconds and
    only then return control to the AC AI expert.  Those early actions and paths
    are disturbance generation, not imitation targets.  DAgger episodes begin
    with AC_AI_expert takeover and are intentionally untouched.
    """
    result = table.copy()
    result["spawn_episode_time_s"] = np.nan
    result["recovery_perturb_seconds"] = np.nan
    result["recovery_perturbation_active"] = False
    synthetic = result.source_type.eq("synthetic_recovery")
    missing_rows = 0
    masked_rows = 0
    affected_trajectories = 0
    for trajectory, group in result[synthetic].groupby("trajectory_id", sort=False):
        first = group.iloc[0]
        episode_path = (Path(str(first.session_path)) /
                        f"episode_{int(first.episode):04d}" / "FRAME_INDEX.csv")
        index = pd.read_csv(episode_path)
        required = {"time_ms", "spawn_episode_time_s", "recovery_perturb_seconds"}
        if not required.issubset(index.columns):
            raise ValueError(f"{episode_path}: missing perturbation timing columns")
        timing = index.set_index(index.time_ms.astype(int))
        elapsed = []
        durations = []
        for time_ms in group.time_ms.astype(int):
            if time_ms not in timing.index:
                missing_rows += 1
                elapsed.append(np.nan); durations.append(np.nan)
                continue
            raw = timing.loc[time_ms]
            if isinstance(raw, pd.DataFrame):
                raw = raw.iloc[0]
            elapsed.append(float(raw.spawn_episode_time_s))
            durations.append(float(raw.recovery_perturb_seconds))
        elapsed = np.asarray(elapsed, float)
        durations = np.asarray(durations, float)
        if not np.isfinite(elapsed).all() or not np.isfinite(durations).all():
            raise RuntimeError(f"{trajectory}: incomplete perturbation timing")
        active = elapsed < durations + float(transition_margin_s)
        result.loc[group.index, "spawn_episode_time_s"] = elapsed
        result.loc[group.index, "recovery_perturb_seconds"] = durations
        result.loc[group.index, "recovery_perturbation_active"] = active
        result.loc[group.index[active], "eligible"] = False
        masked_rows += int(active.sum())
        affected_trajectories += int(active.any())
    if missing_rows:
        raise RuntimeError(f"synthetic perturbation timing misses {missing_rows} manifest rows")
    return result, {
        "masked_rows": masked_rows,
        "affected_trajectories": affected_trajectories,
        "transition_margin_s": transition_margin_s,
        "rule": "synthetic eligible only when spawn_episode_time_s >= perturb_seconds + margin",
        "dagger_untouched": True,
    }


def _stats(values: np.ndarray, minimum_std: float = 1e-3) -> dict[str, object]:
    array = np.asarray(values, np.float64)
    mean = np.nanmean(array, axis=0)
    std = np.nanstd(array, axis=0)
    std = np.maximum(std, minimum_std)
    return {"mean": np.asarray(mean).tolist(), "std": np.asarray(std).tolist()}


def _normalization(table: pd.DataFrame, train_groups: set[str]) -> dict[str, object]:
    train = table[table.trajectory_id.isin(train_groups) & table.eligible].copy()
    yaw = np.asarray([
        quaternion_yaw(*values) for values in train[
            ["quat_x", "quat_y", "quat_z", "quat_w"]].itertuples(index=False, name=None)
    ])
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    line = np.empty((len(train), len(RACING_LINE_LOOKAHEADS_M), 2), np.float32)
    for point, (x_col, y_col) in enumerate(racing_line_columns()):
        dx = train[x_col].to_numpy(float) - train.pos_x.to_numpy(float)
        dy = train[y_col].to_numpy(float) - train.pos_y.to_numpy(float)
        line[:, point, 0] = -sin_yaw * dx + cos_yaw * dy
        line[:, point, 1] = cos_yaw * dx + sin_yaw * dy

    all_index = table.index.to_numpy()
    train_indices = train.index.to_numpy()
    position = {int(value): offset for offset, value in enumerate(all_index)}
    waypoints = np.full((len(train), len(FUTURE_OFFSETS_FRAMES), 2), np.nan, np.float32)
    future_speed = np.full((len(train), len(FUTURE_OFFSETS_FRAMES)), np.nan, np.float32)
    table_keys = {(str(r.trajectory_id), int(r.time_ms)): int(i)
                  for i, r in table[["trajectory_id", "time_ms"]].iterrows()}
    for sample_index, (_, row) in enumerate(train.iterrows()):
        for horizon, offset in enumerate(FUTURE_OFFSETS_FRAMES):
            found = table_keys.get((str(row.trajectory_id), int(row.time_ms) + offset * 25))
            if found is None:
                continue
            future = table.loc[found]
            dx, dy = float(future.pos_x - row.pos_x), float(future.pos_y - row.pos_y)
            waypoints[sample_index, horizon] = (
                -sin_yaw[sample_index] * dx + cos_yaw[sample_index] * dy,
                cos_yaw[sample_index] * dx + sin_yaw[sample_index] * dy,
            )
            future_speed[sample_index, horizon] = abs(float(future.speed_mps))

    target = train[target_speed_columns_v5()].to_numpy(np.float32)
    target = np.column_stack((target, target[:, 0] - np.abs(train.speed_mps.to_numpy(float))))
    return {
        "version": 5,
        "computed_from": "train eligible rows only",
        "ego": _stats(train[list(EGO_FEATURE_NAMES)].to_numpy(np.float32)),
        "racing_line_local_m": _stats(line),
        "waypoints_local_m": _stats(waypoints),
        "future_speed_mps": _stats(future_speed),
        "target_speed_features_mps": _stats(target),
        "image": {
            "resize_hw": [256, 512], "color": "RGB",
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        },
    }


def prepare_v5(source_prepared: Path, output: Path) -> dict[str, object]:
    source_prepared = source_prepared.resolve()
    output.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(source_prepared / "manifest.parquet").copy()
    split = json.loads((source_prepared / "split.json").read_text(encoding="utf-8"))
    train_groups = set(split["train_groups"])
    val_groups = set(split["val_groups"])
    if train_groups & val_groups:
        raise RuntimeError("train/val trajectory overlap")

    line_csv = next(
        Path(path) / "TRACK_INFO" / "AI_RACING_LINE.csv"
        for path in table.session_path.astype(str).unique()
        if (Path(path) / "TRACK_INFO" / "AI_RACING_LINE.csv").exists()
    )
    profile_path = output / "expert_speed_profile_v5.npz"
    profile_summary = build_expert_profile(table, train_groups, line_csv, profile_path)
    profile = ExpertSpeedProfileV5(profile_path)

    table, perturbation_mask = _mask_synthetic_perturbation(table)

    columns = target_speed_columns_v5()
    targets = np.empty((len(table), len(columns)), np.float32)
    line_points = np.empty((len(table), len(RACING_LINE_LOOKAHEADS_M), 2), np.float32)
    for index, row in enumerate(table.itertuples(index=False)):
        phase = str(getattr(row, "lap_phase", "flying"))
        targets[index] = profile.speeds(float(row.spline_position), phase)
        line_points[index] = profile.line_points(float(row.spline_position))
    for index, column in enumerate(columns):
        table[column] = targets[:, index]
    for index, (x_column, y_column) in enumerate(racing_line_columns()):
        table[x_column] = line_points[:, index, 0]
        table[y_column] = line_points[:, index, 1]
    table["target_world_x"] = line_points[:, -1, 0]
    table["target_world_y"] = line_points[:, -1, 1]
    # Remove the invalid v4 signal so it cannot accidentally be selected downstream.
    invalid = [name for name in table.columns if name.startswith("line_") and
               name.endswith("_target_speed_mps")]
    table.drop(columns=invalid, inplace=True, errors="ignore")
    table["sampling_bucket"] = _transition_buckets(table)

    normalization = _normalization(table, train_groups)
    normalization_path = output / "normalization.json"
    normalization_path.write_text(json.dumps(normalization, indent=2), encoding="utf-8")
    normalization_sha = sha256_file(normalization_path)

    split["mode"] = "solo_v5"
    split["speed_profile_build_groups"] = sorted(
        set(table.loc[
            table.trajectory_id.isin(train_groups) & table.source_type.eq("normal"),
            "trajectory_id"]))
    split["speed_profile_excludes"] = "all validation and all recovery/DAgger trajectories"
    table.to_parquet(output / "manifest.parquet", index=False)
    (output / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    schema = {
        "model_schema": model_schema_v5(profile.sha256, normalization_sha),
        "artifacts": {
            "expert_speed_profile_v5.npz": profile.sha256,
            "normalization.json": normalization_sha,
        },
    }
    (output / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    source_exclusions = source_prepared / "exclusions.json"
    if source_exclusions.exists():
        (output / "exclusions.json").write_text(
            source_exclusions.read_text(encoding="utf-8"), encoding="utf-8")

    summary = {
        "schema_version": 5, "frames": int(len(table)),
        "train_samples": int((table.trajectory_id.isin(train_groups) & table.eligible).sum()),
        "val_samples": int((table.trajectory_id.isin(val_groups) & table.eligible).sum()),
        "split_overlap": len(train_groups & val_groups),
        "profile": profile_summary,
        "sampling_buckets": {str(k): int(v) for k, v in
                             table.loc[table.eligible, "sampling_bucket"].value_counts().items()},
        "synthetic_perturbation_mask": perturbation_mask,
        "v4_preserved_at": str(source_prepared),
        "invalid_ai_racing_line_speed_columns_present": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare frozen TCP-AC v5 data")
    parser.add_argument("--source-prepared", type=Path,
                        default=Path("training/tcp_lagunaseca_v4"))
    parser.add_argument("--output", type=Path,
                        default=Path("training/tcp_lagunaseca_v5"))
    args = parser.parse_args()
    print(json.dumps(prepare_v5(args.source_prepared, args.output),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
