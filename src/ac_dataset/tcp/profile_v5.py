"""Train-only expert speed profile shared byte-for-byte by v5 offline/live code."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RACING_LINE_LOOKAHEADS_M
from .config_v5 import PROFILE_BINS


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fill_circular(values: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    result = np.asarray(values, np.float64).copy()
    valid = np.isfinite(result)
    if not valid.all() and fallback is not None:
        result[~valid] = np.asarray(fallback, np.float64)[~valid]
        valid = np.isfinite(result)
    if not valid.any():
        raise ValueError("speed profile has no finite bins")
    if not valid.all():
        x = np.arange(len(result), dtype=np.float64)
        xv = x[valid]
        yv = result[valid]
        result[~valid] = np.interp(
            x[~valid], np.concatenate((xv - len(result), xv, xv + len(result))),
            np.tile(yv, 3))
    return result.astype(np.float32)


def _binned_median(table: pd.DataFrame, bins: int) -> tuple[np.ndarray, np.ndarray]:
    spline = np.mod(table.spline_position.to_numpy(np.float64), 1.0)
    speed = np.abs(table.speed_mps.to_numpy(np.float64))
    indices = np.minimum((spline * bins).astype(np.int64), bins - 1)
    values = np.full(bins, np.nan, np.float64)
    counts = np.zeros(bins, np.int64)
    for index in range(bins):
        sample = speed[indices == index]
        sample = sample[np.isfinite(sample)]
        counts[index] = len(sample)
        if len(sample):
            values[index] = np.median(sample)
    return values, counts


def build_expert_profile(table: pd.DataFrame, train_groups: set[str], line_csv: Path,
                         output: Path, bins: int = PROFILE_BINS) -> dict[str, object]:
    """Build only from normal train trajectories; val/recovery/DAgger are forbidden."""
    train = table[
        table.trajectory_id.isin(train_groups) & (table.source_type == "normal")
    ].copy()
    if train.empty:
        raise ValueError("no normal train rows for expert profile")
    forbidden = set(train.trajectory_id) - train_groups
    if forbidden:
        raise RuntimeError("validation trajectory leaked into speed profile")

    flying_raw, flying_counts = _binned_median(train[train.lap_phase == "flying"], bins)
    flying = _fill_circular(flying_raw)
    standing_raw, standing_counts = _binned_median(train[train.lap_phase == "standing"], bins)
    standing = _fill_circular(standing_raw, flying)

    line = pd.read_csv(line_csv)
    required = {"normalized_position", "dist", "x", "y"}
    if not required.issubset(line):
        raise ValueError(f"racing line misses {sorted(required - set(line))}")
    line = line.sort_values("normalized_position")
    norm = line.normalized_position.to_numpy(np.float64)
    dist = line.dist.to_numpy(np.float64)
    xy = line[["x", "y"]].to_numpy(np.float64)
    length = float(dist[-1] + np.linalg.norm(xy[-1] - xy[0]))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, version=np.int64(5), bins=np.int64(bins),
        standing_speed_mps=standing, flying_speed_mps=flying,
        standing_counts=standing_counts, flying_counts=flying_counts,
        line_normalized_position=norm, line_distance_m=dist,
        line_xy=xy, track_length_m=np.float64(length),
    )
    return {
        "sha256": sha256_file(output), "bins": bins,
        "train_normal_rows": int(len(train)),
        "train_trajectories": int(train.trajectory_id.nunique()),
        "standing_observed_bins": int((standing_counts > 0).sum()),
        "flying_observed_bins": int((flying_counts > 0).sum()),
        "standing_range_mps": [float(standing.min()), float(standing.max())],
        "flying_range_mps": [float(flying.min()), float(flying.max())],
        "line_csv_speed_columns_used": [],
    }


@dataclass
class ExpertSpeedProfileV5:
    path: Path

    def __post_init__(self) -> None:
        payload = np.load(self.path, allow_pickle=False)
        if int(payload["version"]) != 5:
            raise ValueError("not a v5 speed profile")
        self.bins = int(payload["bins"])
        self.standing = payload["standing_speed_mps"].astype(np.float32)
        self.flying = payload["flying_speed_mps"].astype(np.float32)
        self.line_norm = payload["line_normalized_position"].astype(np.float64)
        self.line_dist = payload["line_distance_m"].astype(np.float64)
        self.line_xy = payload["line_xy"].astype(np.float64)
        self.track_length_m = float(payload["track_length_m"])
        self.sha256 = sha256_file(self.path)

    def spline_at_lookaheads(self, spline_position: float,
                             lookaheads=RACING_LINE_LOOKAHEADS_M) -> np.ndarray:
        spline = float(spline_position) % 1.0
        current_dist = float(np.interp(spline, self.line_norm, self.line_dist))
        target_dist = np.mod(current_dist + np.asarray(lookaheads), self.track_length_m)
        return np.interp(target_dist, self.line_dist, self.line_norm).astype(np.float64)

    def speeds(self, spline_position: float, phase: str,
               lookaheads=RACING_LINE_LOOKAHEADS_M) -> np.ndarray:
        positions = self.spline_at_lookaheads(spline_position, lookaheads)
        profile = self.standing if phase == "standing" else self.flying
        centers = (np.arange(self.bins, dtype=np.float64) + .5) / self.bins
        return np.interp(
            positions,
            np.concatenate((centers - 1.0, centers, centers + 1.0)),
            np.tile(profile.astype(np.float64), 3),
        ).astype(np.float32)

    def line_points(self, spline_position: float,
                    lookaheads=RACING_LINE_LOOKAHEADS_M) -> np.ndarray:
        positions = self.spline_at_lookaheads(spline_position, lookaheads)
        x = np.interp(positions, self.line_norm, self.line_xy[:, 0])
        y = np.interp(positions, self.line_norm, self.line_xy[:, 1])
        return np.column_stack((x, y)).astype(np.float32)

