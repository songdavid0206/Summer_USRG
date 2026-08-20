"""Strict audit for delay-aligned V7 prepared data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .prepare_v5 import _future_values
from .prepare_v7 import CONTROL_COLUMNS
from .profile_v5 import sha256_file


def audit_v7(prepared: Path) -> dict:
    prepared = prepared.resolve()
    summary = json.loads((prepared / "summary.json").read_text(encoding="utf-8"))
    source_dir = Path(summary["source_prepared"])
    source = pd.read_parquet(source_dir / "manifest.parquet")
    table = pd.read_parquet(prepared / "manifest.parquet")
    schema = json.loads((prepared / "schema.json").read_text(encoding="utf-8"))
    split = json.loads((prepared / "split.json").read_text(encoding="utf-8"))
    alignment = schema.get("control_label_alignment", {})
    delay_frames = int(alignment.get("delay_frames", -1))
    if len(source) != len(table) or delay_frames <= 0:
        raise RuntimeError("invalid V7 row count or delay metadata")

    expected = {
        name: _future_values(source, name, delay_frames) for name in CONTROL_COLUMNS
    }
    expected_valid = np.logical_and.reduce([
        np.isfinite(np.asarray(values, dtype=float)) for values in expected.values()
    ])
    eligible_expected = source.eligible.astype(bool).to_numpy() & expected_valid
    exact_errors = {}
    for name, values in expected.items():
        mask = expected_valid
        error = np.abs(table.loc[mask, name].to_numpy(float) - np.asarray(values)[mask])
        exact_errors[name] = float(error.max(initial=0.0))
    if any(value != 0.0 for value in exact_errors.values()):
        raise RuntimeError(f"shifted control mismatch: {exact_errors}")
    if not np.array_equal(table.eligible.astype(bool).to_numpy(), eligible_expected):
        raise RuntimeError("V7 eligible mask does not equal V5 eligible AND future-label-valid")

    unchanged_exclusions = {
        *CONTROL_COLUMNS, "pedal_loss_weight", "sampling_bucket", "eligible",
        "control_label_valid", "control_label_delay_frames", "control_label_delay_ms",
        "control_label_time_ms",
    }
    unchanged = [name for name in source.columns
                 if name in table.columns and name not in unchanged_exclusions]
    changed_unexpectedly = []
    for name in unchanged:
        left, right = source[name], table[name]
        if pd.api.types.is_numeric_dtype(left):
            same = np.array_equal(left.to_numpy(), right.to_numpy(), equal_nan=True)
        else:
            same = left.fillna("<NA>").astype(str).equals(
                right.fillna("<NA>").astype(str))
        if not same:
            changed_unexpectedly.append(name)
    if changed_unexpectedly:
        raise RuntimeError(f"non-label columns changed: {changed_unexpectedly}")

    train_groups, val_groups = set(split["train_groups"]), set(split["val_groups"])
    overlap = train_groups & val_groups
    if overlap:
        raise RuntimeError(f"train/val trajectory overlap: {sorted(overlap)[:3]}")
    artifact_errors = {}
    for name, expected_hash in schema["artifacts"].items():
        actual = sha256_file(prepared / name)
        if actual != expected_hash:
            artifact_errors[name] = {"expected": expected_hash, "actual": actual}
    if artifact_errors:
        raise RuntimeError(f"artifact hash mismatch: {artifact_errors}")

    originally_eligible = source.eligible.astype(bool).to_numpy()
    shift_mae = {
        name: float(np.mean(np.abs(
            np.asarray(expected[name])[eligible_expected] -
            source.loc[eligible_expected, name].to_numpy(float))))
        for name in CONTROL_COLUMNS
    }
    result = {
        "passed": True,
        "prepared": str(prepared),
        "source_prepared": str(source_dir),
        "delay_frames": delay_frames,
        "delay_ms": int(alignment["delay_ms"]),
        "rows": int(len(table)),
        "eligible_before": int(originally_eligible.sum()),
        "eligible_after": int(eligible_expected.sum()),
        "dropped_at_trajectory_tails": int((originally_eligible & ~expected_valid).sum()),
        "train_val_overlap": 0,
        "max_control_alignment_error": exact_errors,
        "same_time_to_shifted_target_mae": shift_mae,
        "unexpected_non_label_changes": changed_unexpectedly,
        "artifact_hashes_valid": True,
    }
    (prepared / "audit_v7.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V7 delay-aligned prepared data")
    parser.add_argument("--prepared", type=Path,
                        default=Path("training/tcp_lagunaseca_v7"))
    args = parser.parse_args()
    print(json.dumps(audit_v7(args.prepared), indent=2))


if __name__ == "__main__":
    main()
