"""Validated physical steering-angle to vJoy-command interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SteerMap:
    actions: np.ndarray
    angles_deg: np.ndarray
    source: Path

    @classmethod
    def from_csv(cls, path: str | Path) -> "SteerMap":
        source = Path(path)
        table = pd.read_csv(source)
        required = {"action_steerAngle", "steerAngle"}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"steer map missing columns: {sorted(missing)}")
        clean = table[["action_steerAngle", "steerAngle"]].dropna().astype(float)
        clean = clean.sort_values("steerAngle").drop_duplicates("steerAngle")
        actions = clean["action_steerAngle"].to_numpy(float)
        angles = clean["steerAngle"].to_numpy(float)
        if len(actions) < 2 or not np.isfinite(actions).all() or not np.isfinite(angles).all():
            raise ValueError("steer map needs at least two finite rows")
        if not np.all(np.diff(angles) > 0) or not np.all(np.diff(actions) > 0):
            raise ValueError("steer map angle and action columns must both be strictly increasing")
        if angles[0] >= 0 or angles[-1] <= 0:
            raise ValueError("steer map must span negative and positive physical angles")
        if np.max(np.abs(actions)) > 1.0:
            raise ValueError("steer map actions must stay in vJoy range [-1,1]")
        return cls(actions.astype(np.float32), angles.astype(np.float32), source.resolve())

    @property
    def lock_deg(self) -> float:
        return float(max(abs(float(self.angles_deg[0])), abs(float(self.angles_deg[-1]))))

    def action_for_angle(self, angle_deg: float) -> float:
        angle = float(np.clip(angle_deg, self.angles_deg[0], self.angles_deg[-1]))
        action = float(np.interp(angle, self.angles_deg, self.actions))
        return 0.0 if abs(action) < 1e-12 else action

    def action_for_normalized(self, steer_norm: float) -> float:
        return self.action_for_angle(float(np.clip(steer_norm, -1.0, 1.0)) * self.lock_deg)

    def metadata(self) -> dict[str, object]:
        return {
            "source": str(self.source), "rows": int(len(self.actions)),
            "lock_deg": self.lock_deg,
            "action_min": float(self.actions[0]),
            "action_max": float(self.actions[-1]),
        }
