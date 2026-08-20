"""Audit v2 training inputs and the exact schema consumed by live inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import (
    EGO_FEATURE_NAMES, EGO_FEATURE_NORMALIZED_CLIP, model_schema_metadata,
    normalization_metadata,
)
from .data import (
    normalize_ego_features, preprocess_rgb, quaternion_yaw,
    racing_line_columns, to_local,
)


def audit(prepared: Path, output: Path, sample_count: int = 512) -> dict[str, object]:
    schema_path = prepared / "schema.json"
    recorded = json.loads(schema_path.read_text(encoding="utf-8"))
    expected = {
        "model_schema": model_schema_metadata(),
        "normalization": normalization_metadata(),
    }
    if recorded != expected:
        raise RuntimeError("prepared schema.json differs from current train/live schema")
    table = pd.read_parquet(prepared / "manifest.parquet")
    eligible = table[table["eligible"]].reset_index(drop=True)
    required = set(EGO_FEATURE_NAMES)
    required.update(name for pair in racing_line_columns() for name in pair)
    missing = required - set(eligible.columns)
    if missing:
        raise RuntimeError(f"manifest missing v2 inputs: {sorted(missing)}")

    raw_ego = eligible[list(EGO_FEATURE_NAMES)].to_numpy(np.float32)
    normalized_ego = normalize_ego_features(raw_ego)
    ego_stats = {}
    for index, name in enumerate(EGO_FEATURE_NAMES):
        raw = raw_ego[:, index]
        norm = normalized_ego[:, index]
        ego_stats[name] = {
            "raw_min": float(raw.min()), "raw_max": float(raw.max()),
            "raw_abs_p99": float(np.quantile(np.abs(raw), .99)),
            "normalized_abs_p99": float(np.quantile(np.abs(norm), .99)),
            "clip_fraction": float((np.abs(norm) >= EGO_FEATURE_NORMALIZED_CLIP).mean()),
        }

    take = min(max(1, int(sample_count)), len(eligible))
    indices = np.linspace(0, len(eligible) - 1, take, dtype=int)
    sampled = eligible.iloc[indices]
    local_lines = []
    for row in sampled.itertuples(index=False):
        yaw = quaternion_yaw(row.quat_x, row.quat_y, row.quat_z, row.quat_w)
        world = np.asarray([[getattr(row, x), getattr(row, y)]
                            for x, y in racing_line_columns()], np.float32)
        local_lines.append(to_local(world, np.asarray([row.pos_x, row.pos_y]), yaw))
    local_lines = np.asarray(local_lines, np.float32)
    line_normalized = local_lines / float(
        normalization_metadata()["racing_line_scale_m"])

    image_stats = []
    for row in sampled.iloc[np.linspace(0, len(sampled) - 1, min(8, len(sampled)), dtype=int)].itertuples():
        with Image.open(row.image_path) as image:
            tensor = preprocess_rgb(image)
        image_stats.append({
            "path": str(row.image_path), "shape": list(tensor.shape),
            "mean": float(tensor.mean()), "std": float(tensor.std()),
            "finite": bool(np.isfinite(tensor.numpy()).all()),
        })

    result = {
        "prepared": str(prepared.resolve()), "samples": int(len(eligible)),
        "schema_match": True,
        "absolute_pose_model_input": False,
        "ego": ego_stats,
        "racing_line": {
            "frame": "ego_local_right_forward",
            "sampled": take,
            "shape": list(local_lines.shape),
            "raw_min_m": float(local_lines.min()),
            "raw_max_m": float(local_lines.max()),
            "normalized_abs_max": float(np.abs(line_normalized).max()),
        },
        "controls": {
            name: {"min": float(eligible[name].min()), "max": float(eligible[name].max())}
            for name in ("steer", "throttle", "brake")
        },
        "simultaneous_pedals_fraction": float(
            ((eligible.throttle > .05) & (eligible.brake > .05)).mean()),
        "image_preprocessing": image_stats,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "input_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit v2 model inputs and normalization")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=512)
    args = parser.parse_args()
    audit(args.prepared, args.output, args.sample_count)


if __name__ == "__main__":
    main()
