"""PyTorch dataset for the Assetto Corsa TCP baseline."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    FLYING_LAP_COMMAND_INDEX, POSITION_SCALE_M, STANDING_LAP_COMMAND_INDEX,
)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def route_command(lap_phase: str) -> np.ndarray:
    if lap_phase not in {"standing", "flying"}:
        raise ValueError(f"unknown lap phase: {lap_phase}")
    command = np.zeros(6, np.float32)
    command[3] = 1.0
    command[
        STANDING_LAP_COMMAND_INDEX if lap_phase == "standing"
        else FLYING_LAP_COMMAND_INDEX
    ] = 1.0
    return command


def preprocess_rgb(image, image_size: tuple[int, int] = (256, 512)):
    """Apply the exact RGB preprocessing shared by training and live inference."""
    import torch
    from PIL import Image

    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("RGB input must be an HxWx3 uint8 array")
        image = Image.fromarray(image, mode="RGB")
    else:
        image = image.convert("RGB")
    image = image.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def quaternion_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def to_local(points_xy: np.ndarray, origin_xy: np.ndarray, yaw: float) -> np.ndarray:
    delta = np.asarray(points_xy, dtype=np.float32) - np.asarray(origin_xy, dtype=np.float32)
    forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
    right = np.array([-forward[1], forward[0]], dtype=np.float32)
    return np.column_stack((delta @ right, delta @ forward)).astype(np.float32)


class TCPDataset:
    """Lazy torch Dataset; torch is imported only when samples are requested."""

    def __init__(self, prepared_dir: str | Path, split: str,
                 image_size: tuple[int, int] = (256, 512), augment: bool = False):
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        self.root = Path(prepared_dir)
        config = json.loads((self.root / "split.json").read_text(encoding="utf-8"))
        sessions = set(config[f"{split}_sessions"])
        self.offsets = tuple(int(value) for value in config["future_offsets_frames"])
        table = pd.read_parquet(self.root / "manifest.parquet")
        table = table[table["session"].isin(sessions)].copy()
        table.sort_values(["session", "lap", "time_ms"], inplace=True, ignore_index=True)
        self.table = table
        self.samples = table[table["eligible"]].reset_index(drop=True)
        self.lookup = {
            (row.session, int(row.lap), int(row.time_ms)): index
            for index, row in table.iterrows()
        }
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import torch
        from PIL import Image, ImageEnhance

        row = self.samples.iloc[index]
        image = Image.open(row.image_path).convert("RGB")
        if self.augment:
            factor = float(np.random.uniform(0.85, 1.15))
            image = ImageEnhance.Brightness(image).enhance(factor)
        image_tensor = preprocess_rgb(image, self.image_size)

        future_rows = []
        for offset in self.offsets:
            key = (row.session, int(row.lap), int(row.time_ms) + offset * 25)
            future_rows.append(self.table.iloc[self.lookup[key]])
        future_xy = np.array([[item.pos_x, item.pos_y] for item in future_rows], np.float32)
        yaw = quaternion_yaw(row.quat_x, row.quat_y, row.quat_z, row.quat_w)
        waypoints = to_local(
            future_xy, np.array([row.pos_x, row.pos_y]), yaw,
        ) / POSITION_SCALE_M

        # Fixed-route solo setup: target point comes from the AI racing line,
        # not from the demonstrated future trajectory, so the same input is
        # available during closed-loop inference.
        target_point = to_local(
            np.array([[row.target_world_x, row.target_world_y]], np.float32),
            np.array([row.pos_x, row.pos_y], np.float32), yaw,
        )[0] / POSITION_SCALE_M
        phase = str(row.lap_phase) if "lap_phase" in self.samples.columns else (
            "standing" if int(row.lap) == 1 else "flying")
        command = route_command(phase)
        signed_accel = float(row.throttle) - float(row.brake)
        action = np.array([signed_accel, float(row.steer)], np.float32)
        future_actions = np.array(
            [[float(item.throttle) - float(item.brake), float(item.steer)] for item in future_rows],
            np.float32,
        )
        return {
            "image": image_tensor,
            "speed": torch.tensor([float(row.speed_mps)], dtype=torch.float32),
            "target_point": torch.from_numpy(target_point.copy()),
            "command": torch.from_numpy(command),
            "waypoints": torch.from_numpy(waypoints),
            "action": torch.from_numpy(action),
            "future_actions": torch.from_numpy(future_actions),
        }
