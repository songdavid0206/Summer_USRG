"""Build the v4 solo + recovery + DAgger cohort without crossing trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    FUTURE_OFFSETS_FRAMES, RACING_LINE_LOOKAHEADS_M,
    model_schema_metadata, normalization_metadata,
)
from .data import racing_line_columns, target_speed_columns
from .prepare import VAL_SESSIONS, _read_kv, parse_ego_info


EXCLUDED_TRAJECTORIES = {
    ("20260813_140715_lagunaseca_ferrarisf15t_midclouds_recovery_ai_40hz", 50):
        "all frames include a CSP settings sidebar (visual contamination)",
}

RECOVERY_VAL_SESSIONS = {
    # Whole-session holdouts: one synthetic recovery run and one DAgger run.
    "20260812_130916_lagunaseca_ferrarisf15t_lightfog_recovery_ai_40hz",
    "20260813_140715_lagunaseca_ferrarisf15t_midclouds_recovery_ai_40hz",
}

WEATHER_ALIASES = {
    # Proven identical by AC weather type in the raw exports.
    "3_clear": "clear",                 # type 15
    "5_light_clouds": "scattered_clouds",  # type 17
    "2_light_fog": "mist",             # type 21
}


def _canonical_weather(name: object) -> str:
    raw = str(name or "unknown")
    return WEATHER_ALIASES.get(raw, raw)


def _line_path(session: Path, fallback: Path) -> Path:
    candidate = session / "TRACK_INFO" / "AI_RACING_LINE.csv"
    return candidate if candidate.exists() else fallback


def _add_line_and_speed_targets(table: pd.DataFrame, fallback: Path) -> pd.DataFrame:
    result = table.copy()
    for _, group in result.groupby("session", sort=False):
        session = Path(str(group.iloc[0].session_path))
        line = pd.read_csv(_line_path(session, fallback))
        required = {"normalized_position", "dist", "x", "y"}
        if not required.issubset(line.columns):
            raise ValueError(f"{session}: racing line misses {sorted(required - set(line.columns))}")
        speed_col = next((name for name in ("ai_speed_mps", "ai_speed_ms", "ai_speed_kmh")
                          if name in line), "")
        if speed_col not in line:
            raise ValueError(f"{session}: racing line has no AI target speed")
        speed = line[speed_col].to_numpy(float)
        if speed_col.endswith("kmh"):
            speed = speed / 3.6
        norm = line.normalized_position.to_numpy(float)
        dist = line.dist.to_numpy(float)
        xy = line[["x", "y"]].to_numpy(float)
        length = float(dist[-1] + np.linalg.norm(xy[0] - xy[-1]))
        current = np.interp(group.spline_position.to_numpy(float) % 1.0, norm, dist)
        for ahead, (x_col, y_col), speed_out in zip(
                RACING_LINE_LOOKAHEADS_M, racing_line_columns(), target_speed_columns()):
            target_dist = (current + ahead) % length
            indices = np.minimum(np.searchsorted(dist, target_dist), len(line) - 1)
            result.loc[group.index, x_col] = xy[indices, 0]
            result.loc[group.index, y_col] = xy[indices, 1]
            result.loc[group.index, speed_out] = speed[indices]
    result["target_world_x"] = result[racing_line_columns()[-1][0]]
    result["target_world_y"] = result[racing_line_columns()[-1][1]]
    return result


def _normal_manifest(source: Path) -> pd.DataFrame:
    table = pd.read_parquet(source / "manifest.parquet").copy()
    table["source_type"] = "normal"
    table["weather_raw"] = table["weather"].astype(str)
    table["weather"] = table["weather_raw"].map(_canonical_weather)
    table["episode"] = -1
    table["trajectory_id"] = (
        table.session.astype(str) + "/lap_" + table.lap.astype(int).astype(str).str.zfill(2))
    table["session_path"] = [str(Path(p).parents[2]) for p in table.image_path]
    table["expert_variant"] = "normal_ai"
    table["pedal_loss_weight"] = 1.0
    table["recovery_severity"] = "none"
    brake = table.brake.astype(float) > .10
    corner = table.steer.astype(float).abs() > .08
    table["sampling_bucket"] = np.where(
        brake, "normal_brake", np.where(corner, "normal_corner", "normal_straight"))
    return table


def _float(row, name: str, default: float = 0.0) -> float:
    value = row.get(name, default) if isinstance(row, dict) else getattr(row, name, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _recovery_manifest(dataset_v2: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for folder, source_type in (("solo_recovery", "synthetic_recovery"),
                                ("solo_Dagger", "dagger")):
        root = dataset_v2 / folder
        for session in sorted(root.iterdir()):
            if not session.is_dir() or "lagunaseca" not in session.name.lower():
                continue
            info = _read_kv(session / "SESSION_INFO.txt")
            episode_dirs = {int(p.name.split("_")[-1]): p
                            for p in session.glob("episode_*") if p.is_dir()}
            top = pd.read_csv(session / "FRAME_INDEX.csv")
            indexed = set(top.episode_number.astype(int).unique())
            for missing in sorted(indexed - set(episode_dirs)):
                excluded.append({"session": session.name, "episode": int(missing),
                                 "reason": "listed by session index but episode directory is missing"})
            for episode, ep_path in sorted(episode_dirs.items()):
                explicit = EXCLUDED_TRAJECTORIES.get((session.name, episode))
                if explicit:
                    excluded.append({"session": session.name, "episode": episode,
                                     "reason": explicit})
                    continue
                frame_index_path = ep_path / "FRAME_INDEX.csv"
                if not frame_index_path.exists():
                    excluded.append({"session": session.name, "episode": episode,
                                     "reason": "episode FRAME_INDEX.csv is missing"})
                    continue
                index = pd.read_csv(frame_index_path)
                errors = 0
                episode_records = []
                for row in index.itertuples(index=False):
                    time_ms = int(row.time_ms)
                    image = ep_path / "CAMERA" / f"{time_ms}.jpg"
                    ego_path = ep_path / "EGO_INFO" / f"{time_ms}.txt"
                    if not image.exists():
                        errors += 1
                        continue
                    try:
                        ego = parse_ego_info(ego_path)
                    except (OSError, ValueError):
                        errors += 1
                        continue
                    lateral = abs(_float(row, "recovery_lateral_error_m"))
                    heading = abs(_float(row, "recovery_heading_error_deg"))
                    severity = ("extreme" if lateral > 8 or heading > 45 else
                                "severe" if lateral > 3 or heading > 15 else "mild")
                    supervisor = _float(row, "recovery_longitudinal_active") > .5
                    weather = str(getattr(row, "spawn_weather_name", "") or
                                  info.get("weather_name", "unknown"))
                    episode_records.append({
                        "session": session.name,
                        "session_path": str(session.resolve()),
                        "run_id": info.get("run_id", session.name[:15]),
                        "weather_raw": weather,
                        "weather": _canonical_weather(weather),
                        "lap": int(_float(row, "lap_count", 0)),
                        "lap_phase": "flying",
                        "episode": episode,
                        "trajectory_id": f"{session.name}/episode_{episode:04d}",
                        "time_ms": time_ms,
                        "source_index": int(_float(row, "source_index", -1)),
                        "spline_position": _float(row, "spline_position") % 1.0,
                        "image_path": str(image.resolve()),
                        "source_type": source_type,
                        "expert_variant": f"lookahead_{int(_float(row, 'spawn_ai_lookahead_base', _float(info, 'ai_lookahead_base', 0)))}",
                        "recovery_lateral_error_m": _float(row, "recovery_lateral_error_m"),
                        "recovery_heading_error_deg": _float(row, "recovery_heading_error_deg"),
                        "recovery_severity": severity,
                        "supervisor_active": supervisor,
                        "pedal_loss_weight": .5 if supervisor else 1.0,
                        "sampling_bucket": f"{source_type}_{severity}",
                        **ego,
                    })
                if errors or len(episode_records) != len(index):
                    excluded.append({"session": session.name, "episode": episode,
                                     "reason": f"incomplete episode ({errors} unreadable/missing frames)"})
                    continue
                records.extend(episode_records)
    if not records:
        raise RuntimeError("no usable recovery/DAgger episodes found")
    return pd.DataFrame.from_records(records), excluded


def _future_masks(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["eligible"] = True
    for offset in FUTURE_OFFSETS_FRAMES:
        result[f"future_valid_{offset}"] = False
    for _, group in result.groupby("trajectory_id", sort=False):
        times = set(group.time_ms.astype(int))
        for offset in FUTURE_OFFSETS_FRAMES:
            result.loc[group.index, f"future_valid_{offset}"] = [
                int(t) + int(offset) * 25 in times for t in group.time_ms]
    return result


def _split(table: pd.DataFrame) -> dict[str, object]:
    normal_sessions = sorted(table.loc[table.source_type == "normal", "session"].unique())
    val_sessions = [s for s in normal_sessions if s in VAL_SESSIONS]
    if len(val_sessions) != 2:
        raise RuntimeError(f"fixed normal validation sessions missing: {val_sessions}")
    val_groups = set(table.loc[table.session.isin(val_sessions), "trajectory_id"].unique())
    recovery_sessions = set(table.loc[table.source_type != "normal", "session"].unique())
    missing_holdouts = RECOVERY_VAL_SESSIONS - recovery_sessions
    if missing_holdouts:
        raise RuntimeError(f"recovery validation sessions missing: {sorted(missing_holdouts)}")
    val_groups.update(table.loc[
        table.session.isin(RECOVERY_VAL_SESSIONS), "trajectory_id"].unique())
    all_groups = set(table.trajectory_id.unique())
    train_groups = all_groups - val_groups
    if train_groups & val_groups or train_groups | val_groups != all_groups:
        raise RuntimeError("trajectory split overlap or coverage failure")
    split = {
        "track": "ks_laguna_seca", "mode": "solo_v4", "unit": "trajectory",
        "train_sessions": sorted(set(table.loc[table.trajectory_id.isin(train_groups), "session"])),
        "val_sessions": sorted(set(table.loc[table.trajectory_id.isin(val_groups), "session"])),
        "train_groups": sorted(train_groups), "val_groups": sorted(val_groups),
        "future_offsets_frames": list(FUTURE_OFFSETS_FRAMES),
        "future_offsets_seconds": [x / 40.0 for x in FUTURE_OFFSETS_FRAMES],
        "recovery_validation_rule": "whole held-out sessions; no raw session appears in both sets",
    }
    return split


def _distribution(table: pd.DataFrame, groups: set[str]) -> dict[str, object]:
    part = table[table.trajectory_id.isin(groups)]
    return {
        "frames": int(len(part)), "trajectories": int(part.trajectory_id.nunique()),
        "by_source": {str(k): int(v) for k, v in part.source_type.value_counts().items()},
        "by_weather": {str(k): int(v) for k, v in part.weather.value_counts().items()},
        "by_bucket": {str(k): int(v) for k, v in part.sampling_bucket.value_counts().items()},
        "future_valid": {str(o): int(part[f"future_valid_{o}"].sum())
                         for o in FUTURE_OFFSETS_FRAMES},
    }


def prepare_v4(dataset_v2: Path, normal_prepared: Path, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    normal = _normal_manifest(normal_prepared)
    recovery, excluded = _recovery_manifest(dataset_v2)
    common = sorted(set(normal.columns) | set(recovery.columns))
    table = pd.concat([normal.reindex(columns=common), recovery.reindex(columns=common)],
                      ignore_index=True)
    fallback = next(Path(p) / "TRACK_INFO" / "AI_RACING_LINE.csv"
                    for p in table.session_path if (Path(p) / "TRACK_INFO" / "AI_RACING_LINE.csv").exists())
    table = _add_line_and_speed_targets(table, fallback)
    table = _future_masks(table)
    table.sort_values(["source_type", "session", "trajectory_id", "time_ms"],
                      inplace=True, ignore_index=True)
    split = _split(table)
    train_groups, val_groups = set(split["train_groups"]), set(split["val_groups"])
    # Hard assertions make leakage a preparation error, never a silent training issue.
    assert not train_groups.intersection(val_groups)
    for trajectory, group in table.groupby("trajectory_id"):
        if not group.time_ms.is_unique:
            raise RuntimeError(f"duplicate time within {trajectory}")
    table.to_parquet(output / "manifest.parquet", index=False)
    (output / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    schema = {"model_schema": model_schema_metadata(True),
              "normalization": normalization_metadata(True)}
    (output / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (output / "exclusions.json").write_text(
        json.dumps(excluded, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "schema_version": 4, "frames": int(len(table)),
        "trajectories": int(table.trajectory_id.nunique()),
        "excluded_trajectories": len(excluded),
        "train": _distribution(table, train_groups),
        "val": _distribution(table, val_groups),
        "split_overlap": 0,
        "notes": [
            "No future label crosses a lap or episode boundary.",
            "Target speeds come from AI_RACING_LINE.csv, not future observed speed.",
            "DAgger supervisor pedal labels receive 0.5 loss weight.",
            "Weather aliases with identical AC type are canonicalized; weather_raw is preserved.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TCP-AC v4 solo/recovery/DAgger data")
    parser.add_argument("--dataset-v2", type=Path, default=Path("dataset_v2"))
    parser.add_argument("--normal-prepared", type=Path,
                        default=Path("training/tcp_lagunaseca_v3_dense"))
    parser.add_argument("--output", type=Path, default=Path("training/tcp_lagunaseca_v4"))
    args = parser.parse_args()
    print(json.dumps(prepare_v4(args.dataset_v2, args.normal_prepared, args.output),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
