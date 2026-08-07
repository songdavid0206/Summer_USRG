"""Prepare the Laguna Seca solo dataset for TCP-style training.

This command is deliberately conservative: it only discovers structured solo
sessions for Laguna Seca, packs the small EGO_INFO text labels into one parquet
manifest, samples RGB quality statistics, creates a visual contact sheet, and
splits by complete run. It never mutates the source dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DEFAULT_FUTURE_OFFSETS = (20, 40, 60, 80)  # 0.5, 1.0, 1.5, 2.0 s at 40 Hz
MIN_SESSION_NAME = "20260724_141917_lagunaseca_ferrarisf15t_solo_40hz"
VAL_SESSIONS = {
    "20260724_154559_lagunaseca_ferrarisf15t_clear_solo_40hz",
    "20260727_153344_lagunaseca_ferrarisf15t_clear_solo_40hz_stollrevised",
}


def _read_kv(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def parse_ego_info(path: Path) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        values[key.strip()] = [float(item) for item in raw.split()]
    required = {"position", "orientation", "velocity", "throttle", "brake", "steer_angle"}
    missing = required - values.keys()
    if missing:
        raise ValueError(f"{path}: missing fields {sorted(missing)}")
    p, q, v = values["position"], values["orientation"], values["velocity"]
    if len(p) != 3 or len(q) != 4 or len(v) != 3:
        raise ValueError(f"{path}: invalid vector length")
    return {
        "pos_x": p[0], "pos_y": p[1], "pos_z": p[2],
        "quat_x": q[0], "quat_y": q[1], "quat_z": q[2], "quat_w": q[3],
        "vel_x": v[0], "vel_y": v[1], "vel_z": v[2],
        "speed_mps": float(math.sqrt(sum(item * item for item in v))),
        "throttle": values["throttle"][0],
        "brake": values["brake"][0],
        "steer": values["steer_angle"][0],
    }


def discover_sessions(dataset_v2: Path) -> list[Path]:
    solo = dataset_v2 / "solo"
    sessions = []
    for path in sorted(solo.iterdir()):
        if not path.is_dir() or "lagunaseca" not in path.name.lower():
            continue
        # The 2026-07-23 five-lap runs use a different capture/lap-start
        # convention.  Keep the raw sessions intact, but exclude them from
        # this coherent seven-lap training cohort.
        if path.name < MIN_SESSION_NAME:
            continue
        info = _read_kv(path / "SESSION_INFO.txt")
        if info.get("mode") == "solo" and info.get("track") == "ks_laguna_seca":
            sessions.append(path)
    return sessions


def build_manifest(sessions: list[Path]) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    shared_line_path = next(
        path for path in (session / "TRACK_INFO" / "AI_RACING_LINE.csv" for session in sessions)
        if path.exists()
    )
    for session in sessions:
        info = _read_kv(session / "SESSION_INFO.txt")
        index = pd.read_csv(session / "FRAME_INDEX.csv")
        line_path = session / "TRACK_INFO" / "AI_RACING_LINE.csv"
        racing_line = pd.read_csv(line_path if line_path.exists() else shared_line_path)
        line_norm = racing_line["normalized_position"].to_numpy(float)
        line_dist = racing_line["dist"].to_numpy(float)
        line_length = float(line_dist[-1] + np.linalg.norm(
            racing_line[["x", "y"]].iloc[0].to_numpy(float)
            - racing_line[["x", "y"]].iloc[-1].to_numpy(float)))
        session_count = 0
        errors = 0
        legacy_lap_numbers = "dataset_lap_number" not in index.columns
        for row in index.itertuples(index=False):
            lap_number = int(row.lap_count if legacy_lap_numbers else row.dataset_lap_number)
            if lap_number < 1:
                errors += 1
                continue
            time_ms = int(row.time_ms)
            lap = session / f"lap_{lap_number:02d}"
            image_path = lap / "CAMERA" / f"{time_ms}.jpg"
            ego_path = lap / "EGO_INFO" / f"{time_ms}.txt"
            try:
                ego = parse_ego_info(ego_path)
            except (OSError, ValueError):
                errors += 1
                continue
            spline_position = float(row.spline_position) % 1.0
            current_line_dist = float(np.interp(spline_position, line_norm, line_dist))
            target_dist = (current_line_dist + 50.0) % line_length
            target_index = min(int(np.searchsorted(line_dist, target_dist)), len(racing_line) - 1)
            target = racing_line.iloc[target_index]
            records.append({
                "session": session.name,
                "run_id": info.get("run_id", session.name[:15]),
                "weather": info.get("weather_name", "unknown"),
                "lap": lap_number,
                "lap_phase": "standing" if lap_number == 1 else "flying",
                "time_ms": time_ms,
                "source_index": int(row.source_index),
                "spline_position": spline_position,
                "image_path": str(image_path.resolve()),
                "target_world_x": float(target.x),
                "target_world_y": float(target.y),
                **ego,
            })
            session_count += 1
        summary.append({"session": session.name, "frames": session_count, "label_errors": errors})
    manifest = pd.DataFrame.from_records(records)
    if manifest.empty:
        raise RuntimeError("No readable Laguna Seca solo frames were found")
    manifest.sort_values(["session", "lap", "time_ms"], inplace=True, ignore_index=True)
    return manifest, summary


def add_training_eligibility(manifest: pd.DataFrame, offsets: tuple[int, ...]) -> pd.DataFrame:
    result = manifest.copy()
    result["eligible"] = False
    max_offset = max(offsets)
    for _, group in result.groupby(["session", "lap"], sort=False):
        positions = {int(t): i for i, t in zip(group.index, group["time_ms"])}
        for index, time_ms in zip(group.index, group["time_ms"]):
            if all(int(time_ms) + offset * 25 in positions for offset in offsets):
                result.at[index, "eligible"] = True
    return result


def _rgb_metrics(path: Path) -> tuple[float, float, float]:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((160, 80)), dtype=np.float32)
    mean = float(gray.mean())
    std = float(gray.std())
    sharpness = float(np.var(np.diff(gray, axis=0)) + np.var(np.diff(gray, axis=1)))
    return mean, std, sharpness


def sample_rgb_qc(manifest: pd.DataFrame, stride: int) -> pd.DataFrame:
    rows = []
    for _, group in manifest.groupby("session", sort=False):
        for row in group.iloc[::stride].itertuples(index=False):
            try:
                mean, std, sharpness = _rgb_metrics(Path(row.image_path))
                status = "flag_dark" if mean < 5.0 or std < 2.0 else "ok"
            except (OSError, ValueError):
                mean = std = sharpness = float("nan")
                status = "unreadable"
            rows.append({
                "session": row.session, "lap": row.lap, "time_ms": row.time_ms,
                "image_path": row.image_path, "mean": mean, "std": std,
                "sharpness": sharpness, "status": status,
            })
    return pd.DataFrame(rows)


def make_contact_sheet(manifest: pd.DataFrame, output: Path, samples_per_run: int = 8) -> None:
    thumb_w, thumb_h, label_h = 240, 118, 28
    sessions = list(manifest["session"].drop_duplicates())
    canvas = Image.new("RGB", (thumb_w * samples_per_run, (thumb_h + label_h) * len(sessions)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row_number, session in enumerate(sessions):
        group = manifest[manifest["session"] == session]
        # Random-looking but deterministic positions avoid repeatedly landing
        # on the same track location in equal-length laps.
        rng = np.random.default_rng(2026 + row_number)
        choices = np.sort(rng.choice(len(group), size=min(samples_per_run, len(group)), replace=False))
        for column, position in enumerate(choices):
            sample = group.iloc[int(position)]
            with Image.open(sample.image_path) as image:
                image = image.convert("RGB")
                image.thumbnail((thumb_w, thumb_h))
                x = column * thumb_w + (thumb_w - image.width) // 2
                y = row_number * (thumb_h + label_h)
                canvas.paste(image, (x, y))
        label = f"{session[:15]}  {session.split('_')[-3:] if len(session.split('_')) > 3 else ''}"
        draw.text((4, row_number * (thumb_h + label_h) + thumb_h + 5), label, fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=90)


def write_split(sessions: list[Path], output: Path) -> dict[str, object]:
    names = [path.name for path in sessions]
    val = [name for name in names if name in VAL_SESSIONS]
    if len(val) != 2:
        raise RuntimeError(f"Expected the two fixed validation runs, found {val}")
    train = [name for name in names if name not in VAL_SESSIONS]
    split = {
        "track": "ks_laguna_seca", "mode": "solo", "unit": "run",
        "train_sessions": train, "val_sessions": val,
        "future_offsets_frames": list(DEFAULT_FUTURE_OFFSETS),
        "future_offsets_seconds": [offset / 40.0 for offset in DEFAULT_FUTURE_OFFSETS],
        "lap_phase": {
            "standing_dataset_lap": 1,
            "flying_dataset_lap_min": 2,
        },
    }
    output.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    return split


def prepare(dataset_v2: Path, output: Path, rgb_stride: int = 40) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    sessions = discover_sessions(dataset_v2)
    manifest, session_summary = build_manifest(sessions)
    manifest = add_training_eligibility(manifest, DEFAULT_FUTURE_OFFSETS)
    manifest.to_parquet(output / "manifest.parquet", index=False)
    qc = sample_rgb_qc(manifest, max(1, rgb_stride))
    qc.to_parquet(output / "rgb_qc_samples.parquet", index=False)
    make_contact_sheet(manifest, output / "contact_sheet.jpg")
    split = write_split(sessions, output / "split.json")
    summary = {
        "sessions": len(sessions), "frames": int(len(manifest)),
        "eligible_samples": int(manifest["eligible"].sum()),
        "rgb_samples": int(len(qc)),
        "rgb_flagged": int((qc["status"] != "ok").sum()),
        "train_sessions": len(split["train_sessions"]),
        "val_sessions": len(split["val_sessions"]),
        "session_summary": session_summary,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def prepare_from_manifest(source: Path, output: Path) -> dict[str, object]:
    """Create the coherent seven-lap cohort from an already verified manifest.

    This avoids reparsing the immutable raw EGO_INFO tree when only cohort and
    phase metadata changed.  Image paths and labels remain byte-for-byte the
    values from the previously validated manifest.
    """
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(source / "manifest.parquet")
    manifest = manifest[manifest["session"] >= MIN_SESSION_NAME].copy()
    if manifest.empty:
        raise RuntimeError("filtered manifest is empty")
    manifest["lap_phase"] = np.where(manifest["lap"] == 1, "standing", "flying")
    manifest.to_parquet(output / "manifest.parquet", index=False)

    names = sorted(manifest["session"].unique())
    split = write_split([Path(name) for name in names], output / "split.json")
    source_qc = source / "rgb_qc_samples.parquet"
    rgb_samples = rgb_flagged = 0
    if source_qc.exists():
        qc = pd.read_parquet(source_qc)
        qc = qc[qc["session"].isin(names)].copy()
        qc.to_parquet(output / "rgb_qc_samples.parquet", index=False)
        rgb_samples = int(len(qc))
        rgb_flagged = int((qc["status"] != "ok").sum())
    phase_counts = {
        str(key): int(value)
        for key, value in manifest.loc[manifest["eligible"], "lap_phase"].value_counts().items()
    }
    summary = {
        "source_prepared": str(source.resolve()),
        "cohort_min_session": MIN_SESSION_NAME,
        "sessions": len(names), "frames": int(len(manifest)),
        "eligible_samples": int(manifest["eligible"].sum()),
        "eligible_phase_counts": phase_counts,
        "rgb_samples": rgb_samples, "rgb_flagged": rgb_flagged,
        "train_sessions": len(split["train_sessions"]),
        "val_sessions": len(split["val_sessions"]),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Laguna Seca solo TCP training data")
    parser.add_argument("--dataset-v2", type=Path, default=Path("dataset_v2"))
    parser.add_argument("--output", type=Path, default=Path("training/tcp_lagunaseca"))
    parser.add_argument("--source-prepared", type=Path,
                        help="filter a previously verified manifest instead of reparsing raw files")
    parser.add_argument("--rgb-stride", type=int, default=40,
                        help="RGB statistics sampling stride (40 = approximately 1 Hz)")
    args = parser.parse_args()
    summary = (prepare_from_manifest(args.source_prepared, args.output)
               if args.source_prepared else
               prepare(args.dataset_v2, args.output, args.rgb_stride))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
