"""V5 dataset using train-only feature statistics and future speed targets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import POSITION_SCALE_M, RACING_LINE_SCALE_M
from .data import TCPDataset
from .prepare_v5 import target_speed_columns_v5
from .profile_v5 import sha256_file


class TCPDatasetV5(TCPDataset):
    def __init__(self, prepared_dir: str | Path, split: str,
                 image_size: tuple[int, int] = (256, 512), augment: bool = False):
        super().__init__(prepared_dir, split, image_size=image_size, augment=augment)
        self.v4 = False
        schema = json.loads((self.root / "schema.json").read_text(encoding="utf-8"))
        artifacts = schema.get("artifacts", {})
        for name, expected in artifacts.items():
            actual = sha256_file(self.root / name)
            if actual != expected:
                raise RuntimeError(f"v5 artifact hash mismatch for {name}: {actual} != {expected}")
        self.normalization = json.loads(
            (self.root / "normalization.json").read_text(encoding="utf-8"))
        self._stats = {}
        for name in ("ego", "racing_line_local_m", "waypoints_local_m",
                     "future_speed_mps", "target_speed_features_mps"):
            spec = self.normalization[name]
            self._stats[name] = (
                np.asarray(spec["mean"], np.float32), np.asarray(spec["std"], np.float32))

    def _normalize(self, name: str, value: np.ndarray) -> np.ndarray:
        mean, std = self._stats[name]
        return ((np.asarray(value, np.float32) - mean) / std).astype(np.float32)

    def __getitem__(self, index: int):
        import torch

        item = super().__getitem__(index)
        row = self.samples.iloc[index]
        waypoint_m = item["waypoints"].numpy() * POSITION_SCALE_M
        line_m = item["racing_line"].numpy() * RACING_LINE_SCALE_M
        ego_raw = item["ego_raw"].numpy()
        profile = row[target_speed_columns_v5()].to_numpy(dtype=np.float32)
        target_physical = np.concatenate((
            profile,
            np.asarray([profile[0] - abs(float(row.speed_mps))], np.float32),
        ))

        future_speed = []
        for offset in self.offsets:
            found = self.lookup.get(
                (str(row.trajectory_id), int(row.time_ms) + int(offset) * 25))
            future = row if found is None else self.table.iloc[found]
            future_speed.append(abs(float(future.speed_mps)))
        future_speed = np.asarray(future_speed, np.float32)[:, None]

        item["ego"] = torch.from_numpy(self._normalize("ego", ego_raw))
        item["racing_line"] = torch.from_numpy(
            self._normalize("racing_line_local_m", line_m))
        item["waypoints"] = torch.from_numpy(
            self._normalize("waypoints_local_m", waypoint_m))
        item["future_speeds"] = torch.from_numpy(
            self._normalize("future_speed_mps", future_speed[:, 0])[:, None])
        item["target_speed"] = torch.from_numpy(
            self._normalize("target_speed_features_mps", target_physical))
        item["waypoints_m"] = torch.from_numpy(waypoint_m.astype(np.float32))
        item["future_speeds_mps"] = torch.from_numpy(future_speed)
        item["target_speed_mps"] = torch.from_numpy(target_physical)
        return item

