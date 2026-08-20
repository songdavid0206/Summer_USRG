"""Fail-fast consistency audit for a prepared v4 cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import FUTURE_OFFSETS_FRAMES
from .data import target_speed_columns
from .shadow import RacingLineTarget


def audit(prepared: Path, output: Path | None = None) -> dict[str, object]:
    table = pd.read_parquet(prepared / "manifest.parquet")
    split = json.loads((prepared / "split.json").read_text(encoding="utf-8"))
    schema = json.loads((prepared / "schema.json").read_text(encoding="utf-8"))
    train, val = set(split["train_groups"]), set(split["val_groups"])
    all_groups = set(table.trajectory_id.unique())
    if train & val or train | val != all_groups:
        raise RuntimeError("train/val group leakage or incomplete coverage")
    train_sessions = set(table.loc[table.trajectory_id.isin(train), "session"])
    val_sessions = set(table.loc[table.trajectory_id.isin(val), "session"])
    session_overlap = train_sessions & val_sessions
    if session_overlap:
        raise RuntimeError(f"train/val raw session leakage: {sorted(session_overlap)}")
    if table.duplicated(["trajectory_id", "time_ms"]).any():
        raise RuntimeError("duplicate trajectory/time key")

    mask_mismatches = 0
    for _, group in table.groupby("trajectory_id", sort=False):
        times = set(group.time_ms.astype(int))
        for offset in FUTURE_OFFSETS_FRAMES:
            expected = np.asarray([int(t) + offset * 25 in times for t in group.time_ms])
            mask_mismatches += int(np.count_nonzero(
                expected != group[f"future_valid_{offset}"].to_numpy(bool)))
    if mask_mismatches:
        raise RuntimeError(f"future mask mismatches: {mask_mismatches}")

    fallback = next(Path(p) / "TRACK_INFO" / "AI_RACING_LINE.csv"
                    for p in table.session_path
                    if (Path(p) / "TRACK_INFO" / "AI_RACING_LINE.csv").exists())
    target_max_error = 0.0
    checked_targets = 0
    for _, group in table.groupby("session", sort=False):
        line_path = Path(str(group.iloc[0].session_path)) / "TRACK_INFO" / "AI_RACING_LINE.csv"
        provider = RacingLineTarget.from_csv(line_path if line_path.exists() else fallback)
        sample = group.iloc[np.linspace(0, len(group) - 1, min(64, len(group)), dtype=int)]
        for row in sample.itertuples(index=False):
            expected, _ = provider.target_speed_features(row.spline_position, row.speed_mps)
            actual = np.asarray([getattr(row, name) for name in target_speed_columns()], float)
            target_max_error = max(target_max_error, float(np.max(np.abs(expected - actual))))
            checked_targets += 1
    if target_max_error > 1e-5:
        raise RuntimeError(f"offline/live target speed mismatch: {target_max_error}")

    rgb_shapes: dict[str, int] = {}
    unreadable: list[str] = []
    for row in table.groupby("trajectory_id", sort=False).first().itertuples():
        try:
            with Image.open(row.image_path) as image:
                key = f"{image.mode}:{image.width}x{image.height}"
                rgb_shapes[key] = rgb_shapes.get(key, 0) + 1
        except OSError:
            unreadable.append(str(row.image_path))
    if unreadable:
        raise RuntimeError(f"unreadable trajectory sample images: {len(unreadable)}")

    source_split = {}
    for name, groups in (("train", train), ("val", val)):
        part = table[table.trajectory_id.isin(groups)]
        source_split[name] = {
            "frames": int(len(part)), "trajectories": int(part.trajectory_id.nunique()),
            "sources": {str(k): int(v) for k, v in part.source_type.value_counts().items()},
            "sessions": int(part.session.nunique()), "weather": int(part.weather.nunique()),
        }
    result = {
        "ok": True, "schema_version": schema["model_schema"]["version"],
        "frames": int(len(table)), "trajectories": int(len(all_groups)),
        "train_val_overlap": 0, "train_val_session_overlap": 0,
        "future_mask_mismatches": 0,
        "target_speed_rows_checked": checked_targets,
        "target_speed_live_max_abs_error_mps": target_max_error,
        "rgb_trajectory_samples": int(sum(rgb_shapes.values())), "rgb_shapes": rgb_shapes,
        "split": source_split,
    }
    destination = output or prepared / "audit_v4.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit prepared TCP-AC v4 data")
    parser.add_argument("--prepared", type=Path, default=Path("training/tcp_lagunaseca_v4"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.prepared, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
