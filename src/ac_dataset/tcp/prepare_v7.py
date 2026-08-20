"""Build a delay-compensated control-label view of the frozen v5 data.

The observation, ego state, racing-line features, waypoints, and speed targets
remain anchored at time t.  Only the expert control targets are replaced with
the controls from t + delay.  This teaches the externally actuated policy to
request the action that should be taking effect after the measured
sense/infer/vJoy/AC response delay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .prepare_v5 import _future_values, _transition_buckets
from .profile_v5 import sha256_file


CONTROL_COLUMNS = ("steer", "throttle", "brake")


def align_control_labels(table: pd.DataFrame, delay_frames: int = 2,
                         frame_period_ms: int = 25) -> tuple[pd.DataFrame, dict]:
    """Shift control supervision forward without crossing trajectory boundaries."""
    if delay_frames <= 0 or frame_period_ms <= 0:
        raise ValueError("delay_frames and frame_period_ms must be positive")
    required = {"trajectory_id", "time_ms", "eligible", *CONTROL_COLUMNS}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"manifest missing control-alignment columns: {sorted(missing)}")

    result = table.copy()
    shifted = {
        name: _future_values(table, name, delay_frames)
        for name in CONTROL_COLUMNS
    }
    if "pedal_loss_weight" in table.columns:
        shifted["pedal_loss_weight"] = _future_values(
            table, "pedal_loss_weight", delay_frames)
    valid = np.logical_and.reduce([
        np.isfinite(np.asarray(shifted[name], dtype=float)) for name in CONTROL_COLUMNS
    ])
    originally_eligible = result.eligible.astype(bool).to_numpy()
    result["control_label_valid"] = valid
    result["control_label_delay_frames"] = int(delay_frames)
    result["control_label_delay_ms"] = int(delay_frames * frame_period_ms)
    result["control_label_time_ms"] = (
        result.time_ms.astype(np.int64) + int(delay_frames * frame_period_ms))
    for name, values in shifted.items():
        result[name] = values
    result["eligible"] = originally_eligible & valid
    # Transition buckets must describe the shifted action target, not the
    # observation-time action that is no longer supervised.
    result["sampling_bucket"] = _transition_buckets(result)

    return result, {
        "delay_frames": int(delay_frames),
        "frame_period_ms": int(frame_period_ms),
        "delay_ms": int(delay_frames * frame_period_ms),
        "semantics": "observation_t_supervised_with_expert_control_t_plus_delay",
        "originally_eligible": int(originally_eligible.sum()),
        "aligned_eligible": int(result.eligible.astype(bool).sum()),
        "dropped_missing_future_label": int((originally_eligible & ~valid).sum()),
        "trajectory_boundary_crossing_allowed": False,
    }


def prepare_v7(source_prepared: Path, output: Path, delay_frames: int = 2) -> dict:
    source_prepared = source_prepared.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_manifest = pd.read_parquet(source_prepared / "manifest.parquet")
    table, alignment = align_control_labels(source_manifest, delay_frames, 25)
    split = json.loads((source_prepared / "split.json").read_text(encoding="utf-8"))
    train_groups = set(split["train_groups"])
    val_groups = set(split["val_groups"])
    if train_groups & val_groups:
        raise RuntimeError("train/val trajectory overlap")

    manifest_path = output / "manifest.parquet"
    table.to_parquet(manifest_path, index=False)
    for name in ("normalization.json", "expert_speed_profile_v5.npz"):
        shutil.copy2(source_prepared / name, output / name)
    for name in ("split.json", "exclusions.json", "rgb_qc_samples.parquet",
                 "contact_sheet.jpg"):
        source = source_prepared / name
        if source.exists():
            shutil.copy2(source, output / name)

    source_schema = json.loads(
        (source_prepared / "schema.json").read_text(encoding="utf-8"))
    schema = {
        "model_schema": source_schema["model_schema"],
        "artifacts": {
            "expert_speed_profile_v5.npz": sha256_file(
                output / "expert_speed_profile_v5.npz"),
            "normalization.json": sha256_file(output / "normalization.json"),
            "manifest.parquet": sha256_file(manifest_path),
        },
        "control_label_alignment": alignment,
    }
    (output / "schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8")

    split["mode"] = "solo_v7_delay_aligned_controls"
    split["control_label_alignment"] = alignment
    (output / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    summary = {
        "schema_version": 7,
        "source_prepared": str(source_prepared),
        "frames": int(len(table)),
        "train_samples": int((table.trajectory_id.isin(train_groups) & table.eligible).sum()),
        "val_samples": int((table.trajectory_id.isin(val_groups) & table.eligible).sum()),
        "split_overlap": len(train_groups & val_groups),
        "control_label_alignment": alignment,
        "sampling_buckets": {
            str(key): int(value) for key, value in
            table.loc[table.eligible, "sampling_bucket"].value_counts().items()
        },
        "inputs_planner_targets_and_normalization_unchanged": True,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare v7 delay-aligned control labels")
    parser.add_argument("--source-prepared", type=Path,
                        default=Path("training/tcp_lagunaseca_v5"))
    parser.add_argument("--output", type=Path,
                        default=Path("training/tcp_lagunaseca_v7"))
    parser.add_argument("--delay-frames", type=int, default=2,
                        help="40Hz frames of expert-control lead (2 = 50ms)")
    args = parser.parse_args()
    print(json.dumps(prepare_v7(args.source_prepared, args.output, args.delay_frames),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
