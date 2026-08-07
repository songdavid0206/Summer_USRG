"""Reproducible preflight audit for a prepared TCP-AC training cohort."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import POSITION_SCALE_M, SPEED_SCALE_MPS


OFFSETS = (20, 40, 60, 80)


def _stats(values: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "min": float(values.min()), "mean": float(values.mean()),
        "p01": float(values.quantile(.01)), "p50": float(values.quantile(.50)),
        "p95": float(values.quantile(.95)), "p99": float(values.quantile(.99)),
        "max": float(values.max()),
    }


def _future_integrity(table: pd.DataFrame, eligible: pd.DataFrame) -> dict[str, int]:
    available = pd.MultiIndex.from_frame(table[["session", "lap", "time_ms"]])
    result: dict[str, int] = {}
    for offset in OFFSETS:
        requested = pd.MultiIndex.from_arrays([
            eligible["session"], eligible["lap"], eligible["time_ms"] + offset * 25,
        ])
        result[f"t_plus_{offset * 25}ms_missing"] = int((~requested.isin(available)).sum())
    return result


def _waypoint_stats(table: pd.DataFrame, eligible: pd.DataFrame) -> dict[str, dict[str, float]]:
    future = table[["session", "lap", "time_ms", "pos_x", "pos_y"]].copy()
    out: dict[str, dict[str, float]] = {}
    base = eligible[["session", "lap", "time_ms", "pos_x", "pos_y"]]
    for offset in OFFSETS:
        query = base.assign(future_time_ms=base["time_ms"] + offset * 25)
        joined = query.merge(
            future, left_on=["session", "lap", "future_time_ms"],
            right_on=["session", "lap", "time_ms"], suffixes=("", "_future"),
            how="left", validate="many_to_one",
        )
        distance = np.hypot(
            joined["pos_x_future"] - joined["pos_x"],
            joined["pos_y_future"] - joined["pos_y"],
        )
        stats = _stats(pd.Series(distance))
        stats["normalized_p99"] = stats["p99"] / POSITION_SCALE_M
        out[f"{offset / 40:.1f}s"] = stats
    return out


def audit(prepared: Path, output: Path, batch_size: int = 32) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(prepared / "manifest.parquet")
    split = json.loads((prepared / "split.json").read_text(encoding="utf-8"))
    eligible = table[table["eligible"].astype(bool)].copy()
    train_sessions, val_sessions = set(split["train_sessions"]), set(split["val_sessions"])
    eligible["split"] = np.where(eligible["session"].isin(val_sessions), "val", "train")
    eligible["signed_accel"] = eligible["throttle"] - eligible["brake"]
    eligible["speed_kmh"] = eligible["speed_mps"] * 3.6
    eligible["track_bin"] = np.minimum((eligible["spline_position"] * 100).astype(int), 99)

    errors: list[str] = []
    warnings: list[str] = []
    key_cols = ["session", "lap", "time_ms"]
    duplicate_keys = int(table.duplicated(key_cols).sum())
    if duplicate_keys:
        errors.append(f"duplicate frame keys: {duplicate_keys}")
    missing_image_paths = int(sum(
        not os.path.isfile(path) for path in table["image_path"].drop_duplicates()
    ))
    if missing_image_paths:
        errors.append(f"missing image paths: {missing_image_paths}")
    split_overlap = sorted(train_sessions & val_sessions)
    if split_overlap:
        errors.append(f"train/val session overlap: {split_overlap}")
    assigned = train_sessions | val_sessions
    unassigned = sorted(set(table["session"]) - assigned)
    if unassigned:
        errors.append(f"sessions missing from split: {unassigned}")

    numeric_cols = [
        "spline_position", "pos_x", "pos_y", "pos_z", "quat_x", "quat_y",
        "quat_z", "quat_w", "speed_mps", "throttle", "brake", "steer",
        "target_world_x", "target_world_y",
    ]
    nonfinite = {
        col: int((~np.isfinite(pd.to_numeric(table[col], errors="coerce"))).sum())
        for col in numeric_cols
    }
    if any(nonfinite.values()):
        errors.append(f"non-finite numeric labels: {nonfinite}")
    ranges = {
        "throttle_outside_0_1": int(((table.throttle < 0) | (table.throttle > 1)).sum()),
        "brake_outside_0_1": int(((table.brake < 0) | (table.brake > 1)).sum()),
        "steer_outside_minus1_1": int(((table.steer < -1) | (table.steer > 1)).sum()),
        "negative_speed": int((table.speed_mps < 0).sum()),
        "spline_outside_0_1": int(((table.spline_position < 0) | (table.spline_position >= 1)).sum()),
    }
    if any(ranges.values()):
        errors.append(f"physical range violations: {ranges}")

    phase_bad = int((
        ((table.lap == 1) & (table.lap_phase != "standing")) |
        ((table.lap >= 2) & (table.lap_phase != "flying"))
    ).sum())
    if phase_bad:
        errors.append(f"lap phase mapping violations: {phase_bad}")
    session_laps = table.groupby("session")["lap"].agg(lambda x: sorted(set(map(int, x))))
    bad_laps = {session: laps for session, laps in session_laps.items()
                if laps != list(range(1, 8))}
    if bad_laps:
        errors.append(f"sessions without exact laps 1..7: {bad_laps}")

    future = _future_integrity(table, eligible)
    if any(future.values()):
        errors.append(f"eligible samples with missing future labels: {future}")

    target_distance = np.hypot(
        eligible.target_world_x - eligible.pos_x,
        eligible.target_world_y - eligible.pos_y,
    )
    speed_stats = _stats(eligible.speed_mps)
    speed_stats["normalized_max"] = speed_stats["max"] / SPEED_SCALE_MPS
    target_stats = _stats(pd.Series(target_distance))
    target_stats["normalized_p99"] = target_stats["p99"] / POSITION_SCALE_M
    waypoint_stats = _waypoint_stats(table, eligible)

    session_summary = eligible.groupby(["split", "session"], as_index=False).agg(
        weather=("weather", "first"),
        samples=("session", "size"), laps=("lap", "nunique"),
        standing_samples=("lap_phase", lambda x: int((x == "standing").sum())),
        flying_samples=("lap_phase", lambda x: int((x == "flying").sum())),
        speed_mean_kmh=("speed_kmh", "mean"), speed_p99_kmh=("speed_kmh", lambda x: x.quantile(.99)),
        steer_abs_p99=("steer", lambda x: x.abs().quantile(.99)),
        brake_fraction=("brake", lambda x: (x > .05).mean()),
    )
    session_summary.to_csv(output / "session_summary.csv", index=False)

    track = eligible.groupby(["split", "lap_phase", "track_bin"], as_index=False).agg(
        samples=("session", "size"), speed_mean_kmh=("speed_kmh", "mean"),
        speed_p50_kmh=("speed_kmh", "median"), steer_mean=("steer", "mean"),
        signed_accel_mean=("signed_accel", "mean"),
    )
    track.to_csv(output / "track_coverage.csv", index=False)

    # Fine 0.1%-lap bins reveal same-location action differences between phases.
    eligible["fine_track_bin"] = np.minimum((eligible.spline_position * 1000).astype(int), 999)
    conflicts = eligible.groupby(["split", "fine_track_bin", "lap_phase"], as_index=False).agg(
        samples=("session", "size"), speed_p50_kmh=("speed_kmh", "median"),
        signed_accel_p50=("signed_accel", "median"), steer_p50=("steer", "median"),
    )
    conflicts.to_csv(output / "phase_track_profiles.csv", index=False)
    train_profiles = conflicts[conflicts["split"] == "train"].pivot(
        index="fine_track_bin", columns="lap_phase",
        values=["samples", "speed_p50_kmh", "signed_accel_p50"],
    ).dropna()
    comparable = (
        (train_profiles[("samples", "standing")] >= 5) &
        (train_profiles[("samples", "flying")] >= 5) &
        ((train_profiles[("speed_p50_kmh", "standing")] -
          train_profiles[("speed_p50_kmh", "flying")]).abs() <= 5)
    )
    opposite_context = comparable & (
        (train_profiles[("signed_accel_p50", "standing")] -
         train_profiles[("signed_accel_p50", "flying")]).abs() >= .3
    )
    phase_conflict_bins = int(opposite_context.sum())

    qc_summary: dict[str, object] = {"available": False}
    qc_path = prepared / "rgb_qc_samples.parquet"
    if qc_path.exists():
        qc = pd.read_parquet(qc_path)
        counts = {str(k): int(v) for k, v in qc.status.value_counts().items()}
        qc_summary = {"available": True, "samples": int(len(qc)), "status_counts": counts}
        if int((qc.status != "ok").sum()):
            warnings.append("sampled RGB QC contains flagged frames")
    warnings.extend([
        "40 Hz adjacent frames are strongly temporally correlated; sample count is not independent-scene count.",
        "validation sessions are clear-weather only, so weather robustness is not validated.",
        "expert-line behavior data does not cover learner recovery states or off-track recovery.",
        "RGB QC is sampled rather than a full semantic inspection of every image.",
    ])

    phase_counts = eligible.groupby(["split", "lap_phase"]).size().unstack(fill_value=0)
    coverage: dict[str, object] = {}
    for (split_name, phase), group in eligible.groupby(["split", "lap_phase"]):
        counts = group.groupby("track_bin").size().reindex(range(100), fill_value=0)
        coverage[f"{split_name}_{phase}"] = {
            "bins_present": int((counts > 0).sum()), "min_samples_per_bin": int(counts.min()),
            "median_samples_per_bin": float(counts.median()),
            "zero_bins": [int(x) for x in counts[counts == 0].index],
        }

    train_count = int((eligible["split"] == "train").sum())
    val_count = int((eligible["split"] == "val").sum())
    batches = {
        "batch_size": batch_size,
        "train_batches": math.ceil(train_count / batch_size),
        "val_batches": math.ceil(val_count / batch_size),
        "batches_per_epoch": math.ceil(train_count / batch_size) + math.ceil(val_count / batch_size),
        "optimizer_steps_50_epochs": math.ceil(train_count / batch_size) * 50,
    }
    rare = {
        "hard_left_steer_lt_minus_0_5": int(((eligible.split == "train") & (eligible.steer < -.5)).sum()),
        "hard_right_steer_gt_0_5": int(((eligible.split == "train") & (eligible.steer > .5)).sum()),
        "hard_brake_gt_0_7": int(((eligible.split == "train") & (eligible.brake > .7)).sum()),
        "low_speed_lt_60kmh": int(((eligible.split == "train") & (eligible.speed_kmh < 60)).sum()),
    }

    verdict = "PASS_FOR_PHASE_BASELINE" if not errors else "BLOCKED"
    summary: dict[str, object] = {
        "verdict": verdict, "errors": errors, "warnings": warnings,
        "prepared": str(prepared.resolve()), "sessions": int(table.session.nunique()),
        "frames": int(len(table)), "eligible_samples": int(len(eligible)),
        "train_sessions": len(train_sessions), "val_sessions": len(val_sessions),
        "train_samples": train_count, "val_samples": val_count,
        "phase_counts": {split_name: {phase: int(value) for phase, value in row.items()}
                         for split_name, row in phase_counts.to_dict("index").items()},
        "duplicate_frame_keys": duplicate_keys, "nonfinite": nonfinite,
        "missing_image_paths": missing_image_paths,
        "range_violations": ranges, "phase_mapping_violations": phase_bad,
        "future_label_integrity": future, "speed_mps": speed_stats,
        "target_distance_m": target_stats, "waypoint_distance_m": waypoint_stats,
        "rgb_qc": qc_summary, "track_coverage": coverage,
        "rare_train_samples": rare, "batches": batches,
        "phase_conflict_diagnostic": {
            "fine_bins_with_both_phases": int(len(train_profiles)),
            "similar_speed_bins": int(comparable.sum()),
            "similar_speed_but_accel_difference_ge_0_3": phase_conflict_bins,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    train_hours = train_count / 40 / 3600
    val_hours = val_count / 40 / 3600
    lines = [
        "# TCP-AC phase 데이터셋 본학습 전 종합 감사",
        "", f"- 판정: **{verdict}**", f"- blocking error: {len(errors)}개",
        f"- 세션: {table.session.nunique()}개 (train {len(train_sessions)}, val {len(val_sessions)})",
        f"- eligible: {len(eligible):,}개 (train {train_count:,}, val {val_count:,})",
        f"- 40 Hz 환산: train {train_hours:.2f}시간, val {val_hours:.2f}시간",
        "", "## 핵심 무결성", "",
        f"- 중복 `(session, lap, time_ms)`: {duplicate_keys}",
        f"- 존재하지 않는 RGB 경로: {missing_image_paths}",
        f"- 비유한 숫자 label: {sum(nonfinite.values())}",
        f"- 물리 범위 위반: {sum(ranges.values())}",
        f"- phase mapping 위반: {phase_bad}",
        f"- eligible future label 누락: {sum(future.values())}",
        f"- train/val run 누수: {len(split_overlap)}",
        "", "## standing/flying 구성", "",
        "| split | standing | flying | 합계 |", "|---|---:|---:|---:|",
    ]
    for split_name in ("train", "val"):
        standing = int(phase_counts.loc[split_name].get("standing", 0))
        flying = int(phase_counts.loc[split_name].get("flying", 0))
        lines.append(f"| {split_name} | {standing:,} | {flying:,} | {standing + flying:,} |")
    lines += [
        "",
        f"- 두 phase가 모두 존재하는 train 0.1% 트랙 구간: {len(train_profiles)}개",
        f"- 속도 중앙값 차이가 5 km/h 이하인 비교 가능 구간: {int(comparable.sum())}개",
        f"- 그중 signed-accel 중앙값 차이가 0.3 이상인 구간: {phase_conflict_bins}개",
        "- 이 충돌 구간을 새 one-hot lap phase 입력이 명시적으로 분리한다.",
    ]
    lines += [
        "", "## 물리량과 정규화", "",
        f"- speed p99/max: {speed_stats['p99']:.3f}/{speed_stats['max']:.3f} m/s",
        f"- speed/90 최대: {speed_stats['normalized_max']:.3f}",
        f"- target distance p99: {target_stats['p99']:.2f} m (정규화 {target_stats['normalized_p99']:.3f})",
    ]
    for horizon, stats in waypoint_stats.items():
        lines.append(
            f"- waypoint {horizon}: mean {stats['mean']:.2f} m, p99 {stats['p99']:.2f} m, "
            f"p99/50 {stats['normalized_p99']:.3f}")
    lines += [
        "", "## 1 epoch 작업량", "",
        f"- batch size: {batch_size}", f"- train batch: {batches['train_batches']:,}",
        f"- validation batch: {batches['val_batches']:,}",
        f"- 합계: {batches['batches_per_epoch']:,} batch/epoch",
        f"- 50 epoch optimizer step: {batches['optimizer_steps_50_epochs']:,}",
        "", "## 판정 해석", "",
        "- 현재 데이터는 standing/flying 라벨 충돌 원인을 검증하는 phase-aware baseline 본학습에 충분하다.",
        "- 6개 train standing lap과 36개 train flying lap이 있고 두 phase 모두 전 트랙에 가까운 범위를 덮는다.",
        "- 다만 이 판정은 폐루프 복구 데이터가 충분하다는 뜻이 아니다. 전문가 라인 밖 상태는 별도 보강이 필요하다.",
        "", "## 남은 주의사항", "",
    ]
    lines.extend(f"- {warning}" for warning in warnings)
    lines += [
        "", "## 산출물", "",
        "- `summary.json`: 기계 판독 가능한 전체 결과",
        "- `session_summary.csv`: run별 sample/phase/속도/action 요약",
        "- `track_coverage.csv`: split·phase·트랙 1% 구간별 커버리지",
        "- `phase_track_profiles.csv`: 트랙 0.1% 구간별 standing/flying action profile",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit prepared TCP-AC data before full training")
    parser.add_argument("--prepared", type=Path, default=Path("training/tcp_lagunaseca_phase"))
    parser.add_argument("--output", type=Path,
                        default=Path("training/tcp_lagunaseca_phase/preflight_audit"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    summary = audit(args.prepared, args.output, args.batch_size)
    print(json.dumps({
        "verdict": summary["verdict"], "errors": summary["errors"],
        "train_samples": summary["train_samples"], "val_samples": summary["val_samples"],
        "batches": summary["batches"], "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
