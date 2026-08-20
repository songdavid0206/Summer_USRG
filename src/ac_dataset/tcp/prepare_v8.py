"""Prepare the single-session 5-lap/2-lap V8 convergence experiment."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FUTURE_OFFSETS_FRAMES, RACING_LINE_LOOKAHEADS_M
from .config_v5 import model_schema_v5
from .data import racing_line_columns
from .prepare_v5 import _normalization, _transition_buckets, target_speed_columns_v5
from .profile_v5 import ExpertSpeedProfileV5, build_expert_profile, sha256_file


DEFAULT_SESSION = "20260727_153344_lagunaseca_ferrarisf15t_clear_solo_40hz_stollrevised"
TRAIN_LAPS = (1, 2, 3, 4, 5)
VAL_LAPS = (6, 7)
MAX_GAME_TIME_RESIDUAL_MS = 20.0
MAX_CAMERA_POSE_DELTA_MS = 50.0


def _quality_filter(table: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Remove rows that violate the image/state alignment contract.

    V8 intentionally keeps partial future horizons at lap ends (the loss masks
    those horizons).  This filter is only for a bad *current observation*:
    failed synchronization, a repeated capture, invalid numeric labels, or a
    camera-pose lookup that came from the preceding lap.
    """
    session_paths = {Path(str(value)) for value in table.session_path.unique()}
    if len(session_paths) != 1:
        raise RuntimeError("V8 quality filtering requires exactly one session path")
    session_path = next(iter(session_paths))
    frame_index = pd.read_csv(session_path / "FRAME_INDEX.csv")
    required_index = {
        "dataset_lap_number", "time_ms", "source_frame", "sync_ok",
        "game_time_residual_ms",
    }
    missing = required_index - set(frame_index.columns)
    if missing:
        raise RuntimeError(f"FRAME_INDEX is missing V8 QC fields: {sorted(missing)}")

    reasons: dict[tuple[int, int], set[str]] = {}

    def flag(rows: pd.DataFrame, reason: str) -> None:
        for row in rows.itertuples(index=False):
            key = (int(row.dataset_lap_number), int(row.time_ms))
            reasons.setdefault(key, set()).add(reason)

    flag(frame_index[~frame_index.sync_ok.astype(bool)], "sync_false")
    flag(frame_index[
        pd.to_numeric(frame_index.game_time_residual_ms, errors="coerce").abs()
        .gt(MAX_GAME_TIME_RESIDUAL_MS)
    ], "game_time_residual_gt_20ms")
    flag(frame_index[frame_index.duplicated(
        ["dataset_lap_number", "source_frame"], keep="first")],
         "repeated_source_frame")

    image_index_path = session_path / "image_index.parquet"
    if not image_index_path.exists():
        raise FileNotFoundError(image_index_path)
    image_index = pd.read_parquet(image_index_path)
    required_image = {"dataset_lap_number", "time_ms", "cam_pose_game_delta_ms"}
    missing = required_image - set(image_index.columns)
    if missing:
        raise RuntimeError(f"image_index is missing V8 QC fields: {sorted(missing)}")
    pose_delta = pd.to_numeric(image_index.cam_pose_game_delta_ms, errors="coerce")
    flag(image_index[pose_delta.isna() | pose_delta.abs().gt(MAX_CAMERA_POSE_DELTA_MS)],
         "camera_pose_time_mismatch")

    numeric_columns = [
        "speed_mps", "pos_x", "pos_y", "quat_x", "quat_y", "quat_z", "quat_w",
        "steer", "throttle", "brake", "spline_position",
    ]
    nonfinite = ~np.isfinite(table[numeric_columns].to_numpy(float)).all(axis=1)
    for row in table.loc[nonfinite, ["lap", "time_ms"]].itertuples(index=False):
        reasons.setdefault((int(row.lap), int(row.time_ms)), set()).add(
            "nonfinite_training_value")
    bounds = (
        table.steer.astype(float).abs().gt(1.0001) |
        ~table.throttle.astype(float).between(0.0, 1.0001) |
        ~table.brake.astype(float).between(0.0, 1.0001) |
        ~table.spline_position.astype(float).between(0.0, 1.0) |
        ~table.speed_mps.astype(float).between(0.0, 100.0)
    )
    for row in table.loc[bounds, ["lap", "time_ms"]].itertuples(index=False):
        reasons.setdefault((int(row.lap), int(row.time_ms)), set()).add(
            "physical_bounds_violation")
    for row in table.loc[
            ~table.image_path.map(lambda value: Path(str(value)).is_file()),
            ["lap", "time_ms"]].itertuples(index=False):
        reasons.setdefault((int(row.lap), int(row.time_ms)), set()).add("missing_image")

    excluded = []
    keep = np.ones(len(table), dtype=bool)
    for position, row in enumerate(table.itertuples(index=False)):
        key = (int(row.lap), int(row.time_ms))
        if key in reasons:
            keep[position] = False
            excluded.append({
                "lap": key[0], "time_ms": key[1],
                "reasons": sorted(reasons[key]),
            })
    return table.loc[keep].copy(), excluded


def prepare_v8(source_prepared: Path, output: Path,
               session: str = DEFAULT_SESSION) -> dict[str, object]:
    source_prepared = source_prepared.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = pd.read_parquet(source_prepared / "manifest.parquet")
    table = source[(source.session.astype(str) == session) &
                   source.source_type.eq("normal")].copy()
    if table.empty:
        raise ValueError(f"V8 session is absent from source manifest: {session}")
    source_rows = int(len(table))
    table, quality_exclusions = _quality_filter(table)
    observed_laps = tuple(sorted(table.lap.astype(int).unique()))
    expected_laps = tuple(sorted(TRAIN_LAPS + VAL_LAPS))
    if observed_laps != expected_laps:
        raise RuntimeError(f"V8 requires laps {expected_laps}, observed {observed_laps}")
    if table.trajectory_id.nunique() != 7:
        raise RuntimeError("V8 requires exactly one trajectory per lap")

    train_groups = set(table.loc[table.lap.isin(TRAIN_LAPS), "trajectory_id"])
    val_groups = set(table.loc[table.lap.isin(VAL_LAPS), "trajectory_id"])
    if train_groups & val_groups or len(train_groups) != 5 or len(val_groups) != 2:
        raise RuntimeError("invalid V8 5/2 trajectory split")
    standing_groups = set(table.loc[table.lap_phase.eq("standing"), "trajectory_id"])
    if not standing_groups or not standing_groups <= train_groups:
        raise RuntimeError("standing lap must exist and be wholly assigned to train")

    line_csv = Path(str(table.iloc[0].session_path)) / "TRACK_INFO" / "AI_RACING_LINE.csv"
    if not line_csv.exists():
        raise FileNotFoundError(line_csv)
    profile_path = output / "expert_speed_profile_v5.npz"
    profile_summary = build_expert_profile(table, train_groups, line_csv, profile_path)
    profile = ExpertSpeedProfileV5(profile_path)

    target_columns = target_speed_columns_v5()
    targets = np.empty((len(table), len(target_columns)), np.float32)
    line_points = np.empty((len(table), len(RACING_LINE_LOOKAHEADS_M), 2), np.float32)
    for index, row in enumerate(table.itertuples(index=False)):
        phase = str(row.lap_phase)
        targets[index] = profile.speeds(float(row.spline_position), phase)
        line_points[index] = profile.line_points(float(row.spline_position))
    for index, column in enumerate(target_columns):
        table[column] = targets[:, index]
    for index, (x_column, y_column) in enumerate(racing_line_columns()):
        table[x_column] = line_points[:, index, 0]
        table[y_column] = line_points[:, index, 1]
    table["target_world_x"] = line_points[:, -1, 0]
    table["target_world_y"] = line_points[:, -1, 1]
    table["sampling_bucket"] = _transition_buckets(table)
    table.sort_values(["lap", "time_ms"], inplace=True, ignore_index=True)

    normalization = _normalization(table, train_groups)
    normalization_path = output / "normalization.json"
    normalization_path.write_text(json.dumps(normalization, indent=2), encoding="utf-8")
    normalization_sha = sha256_file(normalization_path)
    manifest_path = output / "manifest.parquet"
    table.to_parquet(manifest_path, index=False)
    quality_path = output / "quality_exclusions.json"
    quality_path.write_text(json.dumps({
        "source_rows": source_rows,
        "retained_rows": int(len(table)),
        "excluded_rows": len(quality_exclusions),
        "thresholds": {
            "max_game_time_residual_ms": MAX_GAME_TIME_RESIDUAL_MS,
            "max_camera_pose_delta_ms": MAX_CAMERA_POSE_DELTA_MS,
        },
        "rows": quality_exclusions,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    split = {
        "track": "ks_laguna_seca",
        "mode": "solo_v8_single_session_5_train_2_val",
        "unit": "lap_trajectory",
        "session": session,
        "train_laps": list(TRAIN_LAPS), "val_laps": list(VAL_LAPS),
        "train_sessions": [session], "val_sessions": [session],
        "train_groups": sorted(train_groups), "val_groups": sorted(val_groups),
        "future_offsets_frames": list(FUTURE_OFFSETS_FRAMES),
        "future_offsets_seconds": [value / 40.0 for value in FUTURE_OFFSETS_FRAMES],
        "standing_lap_rule": "standing lap is always train",
        "phase_training": "one shared model and loss; phase is an input bit, not a split model",
    }
    split_path = output / "split.json"
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    schema = {
        "model_schema": model_schema_v5(profile.sha256, normalization_sha),
        "artifacts": {
            "expert_speed_profile_v5.npz": profile.sha256,
            "normalization.json": normalization_sha,
            "manifest.parquet": sha256_file(manifest_path),
            "quality_exclusions.json": sha256_file(quality_path),
        },
        "experiment": {
            "version": 8, "session": session,
            "train_laps": list(TRAIN_LAPS), "val_laps": list(VAL_LAPS),
            "recovery_or_dagger_included": False,
        },
    }
    (output / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    exclusions = source_prepared / "exclusions.json"
    if exclusions.exists():
        shutil.copy2(exclusions, output / "source_exclusions.json")

    summary = {
        "schema_version": 8, "session": session,
        "source_rows": source_rows,
        "rows": int(len(table)),
        "quality_excluded_rows": len(quality_exclusions),
        "quality_exclusions": quality_exclusions,
        "train_samples": int((table.trajectory_id.isin(train_groups) & table.eligible).sum()),
        "val_samples": int((table.trajectory_id.isin(val_groups) & table.eligible).sum()),
        "train_laps": list(TRAIN_LAPS), "val_laps": list(VAL_LAPS),
        "standing_train_rows": int((table.trajectory_id.isin(train_groups) &
                                     table.lap_phase.eq("standing")).sum()),
        "train_val_trajectory_overlap": len(train_groups & val_groups),
        "sources": {str(k): int(v) for k, v in table.source_type.value_counts().items()},
        "profile": profile_summary,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TCP-AC V8 single-session 5/2 split")
    parser.add_argument("--source-prepared", type=Path,
                        default=Path("training/tcp_lagunaseca_v5"))
    parser.add_argument("--output", type=Path, default=Path("training/tcp_lagunaseca_v8"))
    parser.add_argument("--session", default=DEFAULT_SESSION)
    args = parser.parse_args()
    print(json.dumps(prepare_v8(args.source_prepared, args.output, args.session),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
