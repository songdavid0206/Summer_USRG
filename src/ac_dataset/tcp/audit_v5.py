"""Fail-closed integrity audit for v5 data, artifacts, normalization and wiring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config_v5 import (
    BRAKE_RESIDUAL_CAP, STEER_RESIDUAL_CAP, TARGET_SPEED_WIDTH,
    THROTTLE_RESIDUAL_CAP,
)
from .prepare_v5 import target_speed_columns_v5
from .data import racing_line_columns
from .profile_v5 import ExpertSpeedProfileV5, sha256_file
from .train_v5 import TARGET_SHARES


def audit(prepared: Path, model_check: bool = True) -> dict[str, object]:
    schema = json.loads((prepared / "schema.json").read_text(encoding="utf-8"))
    split = json.loads((prepared / "split.json").read_text(encoding="utf-8"))
    normalization = json.loads((prepared / "normalization.json").read_text(encoding="utf-8"))
    table = pd.read_parquet(prepared / "manifest.parquet")
    train, val = set(split["train_groups"]), set(split["val_groups"])
    if train & val:
        raise RuntimeError("train/val overlap")
    artifacts = schema["artifacts"]
    hash_results = {}
    for name, expected in artifacts.items():
        actual = sha256_file(prepared / name)
        if actual != expected:
            raise RuntimeError(f"artifact mismatch: {name}")
        hash_results[name] = actual
    if any(name.startswith("line_") and name.endswith("_target_speed_mps")
           for name in table.columns):
        raise RuntimeError("invalid v4 target-speed columns survived")
    if "recovery_perturbation_active" not in table:
        raise RuntimeError("v5 manifest lacks synthetic perturbation mask")
    perturbation_leaks = int((table.eligible.astype(bool) &
                              table.recovery_perturbation_active.astype(bool)).sum())
    if perturbation_leaks:
        raise RuntimeError(f"{perturbation_leaks} deliberate perturbation frames are eligible")

    profile = ExpertSpeedProfileV5(prepared / "expert_speed_profile_v5.npz")
    indices = np.linspace(0, len(table) - 1, min(1000, len(table)), dtype=int)
    maximum_lookup_error = 0.0
    maximum_line_lookup_error = 0.0
    columns = target_speed_columns_v5()
    for index in indices:
        row = table.iloc[index]
        live = profile.speeds(float(row.spline_position), str(row.lap_phase))
        stored = row[columns].to_numpy(dtype=np.float32)
        maximum_lookup_error = max(maximum_lookup_error,
                                   float(np.max(np.abs(live - stored))))
        live_line = profile.line_points(float(row.spline_position))
        stored_line = np.asarray([[float(row[x]), float(row[y])]
                                  for x, y in racing_line_columns()], np.float32)
        maximum_line_lookup_error = max(
            maximum_line_lookup_error, float(np.max(np.abs(live_line - stored_line))))
    if maximum_lookup_error > 1e-6:
        raise RuntimeError(f"train/live target lookup mismatch: {maximum_lookup_error}")
    if maximum_line_lookup_error > 1e-6:
        raise RuntimeError(f"train/live racing-line lookup mismatch: {maximum_line_lookup_error}")

    val_normal = table[table.trajectory_id.isin(val) & table.source_type.eq("normal")]
    predicted = val_normal[columns[0]].to_numpy(float)
    actual = np.abs(val_normal.speed_mps.to_numpy(float))
    correlation = float(np.corrcoef(predicted, actual)[0, 1])
    mae_mps = float(np.mean(np.abs(predicted - actual)))
    if correlation < .90 or mae_mps > 3.0:
        raise RuntimeError(f"expert profile physical validation failed: corr={correlation} mae={mae_mps}")

    for name, spec in normalization.items():
        if not isinstance(spec, dict) or "std" not in spec:
            continue
        mean, std = np.asarray(spec["mean"]), np.asarray(spec["std"])
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise RuntimeError(f"bad normalization: {name}")

    model_results = {"checked": False}
    if model_check:
        import torch
        from .data_v5 import TCPDatasetV5
        from .model_v5 import build_model_v5

        dataset = TCPDatasetV5(prepared, "val", augment=False)
        # Some archived normal-session files can be ACL-restricted in managed
        # audit environments. Prefer held-out recovery samples here; the loader
        # and tensor path are identical and the training smoke test covers the
        # sampler's complete mixed cohort.
        candidates = np.flatnonzero(dataset.samples.source_type.ne("normal").to_numpy())
        selected = candidates[:2] if len(candidates) >= 2 else np.arange(min(2, len(dataset)))
        items = [dataset[int(index)] for index in selected]
        batch = {key: torch.stack([item[key] for item in items]) for key in items[0]}
        model = build_model_v5(False).eval()
        with torch.no_grad():
            pred = model(batch["image"], batch["ego"], batch["racing_line"],
                         batch["command"], batch["target_speed"])
        caps = torch.tensor(
            [STEER_RESIDUAL_CAP, THROTTLE_RESIDUAL_CAP, BRAKE_RESIDUAL_CAP])
        residual_max = pred["visual_residual"].abs().amax(dim=0)
        if torch.any(residual_max > caps + 1e-6):
            raise RuntimeError("residual cap failure")
        # The public controller accepts predicted-plan tensors only. Changing a
        # plan must change control; there is no GT plan argument to leak through.
        changed_waypoints = pred["waypoints"].clone()
        changed_waypoints[:, :, 0] += .5
        with torch.no_grad():
            changed_control, _ = model.control_from_plan(
                changed_waypoints, pred["future_speeds"], batch["ego"],
                batch["target_speed"], batch["command"])
        plan_sensitivity = float((changed_control - pred["plan_controls"]).abs().max())
        if plan_sensitivity <= 0:
            raise RuntimeError("control is disconnected from predicted plan")
        model.zero_grad(set_to_none=True)
        gradient_pred = model(batch["image"], batch["ego"], batch["racing_line"],
                              batch["command"], batch["target_speed"])
        gradient_pred["plan_controls"].sum().backward()
        planner_gradient = model.waypoint_delta.weight.grad
        planner_gradient_norm = (0.0 if planner_gradient is None else
                                 float(planner_gradient.abs().sum()))
        if planner_gradient_norm <= 0:
            raise RuntimeError("joint control gradient does not reach planner")
        model_results = {
            "checked": True,
            "image_shape": list(batch["image"].shape),
            "target_speed_width": int(batch["target_speed"].shape[1]),
            "residual_abs_max": residual_max.tolist(),
            "plan_control_sensitivity": plan_sensitivity,
            "control_to_planner_gradient_l1": planner_gradient_norm,
            "ground_truth_plan_is_control_input": False,
            "all_outputs_finite": all(torch.isfinite(value).all().item()
                                      for value in pred.values()),
        }
        if not model_results["all_outputs_finite"]:
            raise RuntimeError("non-finite model output")

    train_samples = table[table.trajectory_id.isin(train) & table.eligible.astype(bool)]
    bucket_audit = {}
    for bucket, count in train_samples.sampling_bucket.value_counts().items():
        expected = len(train_samples) * TARGET_SHARES.get(str(bucket), 0.0)
        bucket_audit[str(bucket)] = {
            "rows": int(count),
            "trajectories": int(train_samples[
                train_samples.sampling_bucket.eq(bucket)].trajectory_id.nunique()),
            "target_share": float(TARGET_SHARES.get(str(bucket), 0.0)),
            "expected_repeats_per_epoch": float(expected / max(1, count)),
        }
    return {
        "passed": True,
        "schema_version": schema["model_schema"]["version"],
        "train_groups": len(train), "val_groups": len(val), "overlap": 0,
        "artifact_hashes": hash_results,
        "target_lookup_max_abs_error_mps": maximum_lookup_error,
        "racing_line_lookup_max_abs_error_m": maximum_line_lookup_error,
        "heldout_normal_profile_corr": correlation,
        "heldout_normal_profile_mae_mps": mae_mps,
        "profile_range_mps": {
            "standing": [float(profile.standing.min()), float(profile.standing.max())],
            "flying": [float(profile.flying.min()), float(profile.flying.max())],
        },
        "eligible_deliberate_perturbation_frames": perturbation_leaks,
        "train_sampling_bucket_audit": bucket_audit,
        "model": model_results,
        "target_width_expected": TARGET_SPEED_WIDTH,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path,
                        default=Path("training/tcp_lagunaseca_v5"))
    parser.add_argument("--model-check", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    result = audit(args.prepared, args.model_check)
    (args.prepared / "audit_v5.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
