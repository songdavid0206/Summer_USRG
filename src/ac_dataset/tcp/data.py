"""PyTorch dataset for the Assetto Corsa TCP baseline."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    EGO_FEATURE_NAMES, EGO_FEATURE_NORMALIZED_CLIP, EGO_FEATURE_SCALES,
    FLYING_LAP_COMMAND_INDEX, POSITION_SCALE_M, RACING_LINE_LOOKAHEADS_M,
    RACING_LINE_SCALE_M, SPEED_SCALE_MPS, STANDING_LAP_COMMAND_INDEX,
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


def racing_line_columns() -> list[tuple[str, str]]:
    return [(f"line_{int(distance)}m_x", f"line_{int(distance)}m_y")
            for distance in RACING_LINE_LOOKAHEADS_M]


def target_speed_columns() -> list[str]:
    return [f"line_{int(distance)}m_target_speed_mps"
            for distance in RACING_LINE_LOOKAHEADS_M]


def normalize_ego_features(values) -> np.ndarray:
    """Normalize the documented local ego vector; absolute pose is never included."""
    array = np.asarray(values, dtype=np.float32)
    if array.shape[-1] != len(EGO_FEATURE_NAMES):
        raise ValueError(
            f"ego feature width {array.shape[-1]} != {len(EGO_FEATURE_NAMES)}")
    scales = np.asarray(EGO_FEATURE_SCALES, dtype=np.float32)
    return np.clip(
        array / scales, -EGO_FEATURE_NORMALIZED_CLIP,
        EGO_FEATURE_NORMALIZED_CLIP).astype(np.float32)


def add_temporal_ego_features(table: pd.DataFrame) -> pd.DataFrame:
    """Derive local velocity/yaw-rate/acceleration identically for offline data.

    EGO_INFO velocity_x/y are vehicle-local longitudinal/lateral velocity.  Pose
    is used only to derive yaw rate and local racing-line coordinates; absolute
    pose columns remain metadata, never model measurements.
    """
    result = table.copy()
    derived = {
        "longitudinal_speed_mps", "lateral_speed_mps", "yaw_rate_rad_s",
        "longitudinal_accel_mps2", "lateral_accel_mps2",
    }
    if derived.issubset(result.columns):
        # Full v2 preparation reads the exact local signals exported in EGO_INFO.
        return result
    result["longitudinal_speed_mps"] = result["vel_x"].astype(float)
    result["lateral_speed_mps"] = result["vel_y"].astype(float)
    result["yaw_rad"] = np.array([
        quaternion_yaw(x, y, z, w) for x, y, z, w in result[
            ["quat_x", "quat_y", "quat_z", "quat_w"]].itertuples(index=False, name=None)
    ])
    result["yaw_rate_rad_s"] = 0.0
    result["longitudinal_accel_mps2"] = 0.0
    result["lateral_accel_mps2"] = 0.0
    for _, group in result.groupby(["session", "lap"], sort=False):
        dt = group["time_ms"].diff().to_numpy(float) / 1000.0
        yaw = group["yaw_rad"].to_numpy(float)
        dyaw = np.arctan2(np.sin(np.diff(yaw, prepend=yaw[0])),
                          np.cos(np.diff(yaw, prepend=yaw[0])))
        valid = np.isfinite(dt) & (dt > 0.0) & (dt <= 0.2)
        yaw_rate = np.zeros(len(group), float)
        long_accel = np.zeros(len(group), float)
        lateral_accel = np.zeros(len(group), float)
        yaw_rate[valid] = dyaw[valid] / dt[valid]
        long_delta = group["longitudinal_speed_mps"].diff().to_numpy(float)
        lat_delta = group["lateral_speed_mps"].diff().to_numpy(float)
        long_accel[valid] = long_delta[valid] / dt[valid]
        lateral_accel[valid] = lat_delta[valid] / dt[valid]
        result.loc[group.index, "yaw_rate_rad_s"] = yaw_rate
        result.loc[group.index, "longitudinal_accel_mps2"] = long_accel
        result.loc[group.index, "lateral_accel_mps2"] = lateral_accel
    return result


class EgoFeatureBuilder:
    """Stateful live equivalent of :func:`add_temporal_ego_features`."""

    def __init__(self):
        self.previous: tuple[float, float, float, float] | None = None

    def reset(self) -> None:
        self.previous = None

    def update(self, speed_mps: float, longitudinal_speed_mps: float,
               lateral_speed_mps: float, yaw_rad: float,
               timestamp_s: float) -> tuple[np.ndarray, np.ndarray]:
        yaw_rate = longitudinal_accel = lateral_accel = 0.0
        if self.previous is not None:
            previous_longitudinal, previous_lateral, previous_yaw, previous_time = self.previous
            dt = float(timestamp_s) - previous_time
            if 0.0 < dt <= 0.2:
                yaw_delta = math.atan2(
                    math.sin(float(yaw_rad) - previous_yaw),
                    math.cos(float(yaw_rad) - previous_yaw))
                yaw_rate = yaw_delta / dt
                longitudinal_accel = (
                    float(longitudinal_speed_mps) - previous_longitudinal) / dt
                lateral_accel = (float(lateral_speed_mps) - previous_lateral) / dt
        self.previous = (float(longitudinal_speed_mps), float(lateral_speed_mps),
                         float(yaw_rad), float(timestamp_s))
        raw = np.asarray([
            abs(float(speed_mps)), float(longitudinal_speed_mps),
            float(lateral_speed_mps), yaw_rate,
            longitudinal_accel, lateral_accel,
        ], dtype=np.float32)
        return raw, normalize_ego_features(raw)


class TCPDataset:
    """Lazy torch Dataset; torch is imported only when samples are requested."""

    def __init__(self, prepared_dir: str | Path, split: str,
                 image_size: tuple[int, int] = (256, 512), augment: bool = False):
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        self.root = Path(prepared_dir)
        config = json.loads((self.root / "split.json").read_text(encoding="utf-8"))
        sessions = set(config.get(f"{split}_sessions", []))
        groups = set(config.get(f"{split}_groups", []))
        self.offsets = tuple(int(value) for value in config["future_offsets_frames"])
        table = pd.read_parquet(self.root / "manifest.parquet")
        if "trajectory_id" not in table:
            table["trajectory_id"] = (
                table["session"].astype(str) + "/lap_" +
                table["lap"].astype(int).astype(str).str.zfill(2))
        selected = table["trajectory_id"].isin(groups) if groups else table["session"].isin(sessions)
        table = table[selected].copy()
        table.sort_values(["trajectory_id", "time_ms"], inplace=True, ignore_index=True)
        self.table = table
        self.samples = table[table["eligible"]].reset_index(drop=True)
        required = set(EGO_FEATURE_NAMES)
        required.update(name for pair in racing_line_columns() for name in pair)
        missing = required - set(table.columns)
        if missing:
            raise ValueError(
                f"prepared manifest uses the old schema; missing {sorted(missing)}. "
                "Run prepare.py --source-prepared ... into a new output directory.")
        self.lookup = {
            (str(row.trajectory_id), int(row.time_ms)): index
            for index, row in table.iterrows()
        }
        self.image_size = image_size
        self.augment = augment
        self.v4 = all(name in table.columns for name in target_speed_columns())
        self.sample_buckets = self.samples.get(
            "sampling_bucket", pd.Series("normal", index=self.samples.index)).astype(str).tolist()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import torch
        from PIL import Image, ImageEnhance

        row = self.samples.iloc[index]
        image = Image.open(row.image_path).convert("RGB")
        if self.augment:
            # The network has always consumed 512x256. Apply train-only
            # photometric augmentation at that actual input resolution instead
            # of making three full-resolution 1920x945 CPU passes. The shared
            # preprocess_rgb function remains unchanged for live inference.
            image = image.resize(
                (self.image_size[1], self.image_size[0]), Image.Resampling.BILINEAR)
            image = ImageEnhance.Brightness(image).enhance(float(np.random.uniform(.82, 1.18)))
            image = ImageEnhance.Contrast(image).enhance(float(np.random.uniform(.86, 1.14)))
            image = ImageEnhance.Color(image).enhance(float(np.random.uniform(.88, 1.12)))
        image_tensor = preprocess_rgb(image, self.image_size)

        future_rows = []
        future_mask = []
        for offset in self.offsets:
            key = (str(row.trajectory_id), int(row.time_ms) + offset * 25)
            found = self.lookup.get(key)
            future_mask.append(float(found is not None))
            future_rows.append(row if found is None else self.table.iloc[found])
        future_xy = np.array([[item.pos_x, item.pos_y] for item in future_rows], np.float32)
        yaw = quaternion_yaw(row.quat_x, row.quat_y, row.quat_z, row.quat_w)
        waypoints = to_local(
            future_xy, np.array([row.pos_x, row.pos_y]), yaw,
        ) / POSITION_SCALE_M

        # Eleven fixed-distance racing-line points are transformed from global
        # storage into the ego-local right/forward frame before entering the model.
        line_world = np.array(
            [[float(row[x]), float(row[y])] for x, y in racing_line_columns()],
            np.float32)
        racing_line = to_local(
            line_world, np.array([row.pos_x, row.pos_y], np.float32), yaw,
        ) / RACING_LINE_SCALE_M
        ego_raw = np.array([float(row[name]) for name in EGO_FEATURE_NAMES], np.float32)
        ego = normalize_ego_features(ego_raw)
        phase = str(row.lap_phase) if "lap_phase" in self.samples.columns else (
            "standing" if int(row.lap) == 1 else "flying")
        command = route_command(phase)
        controls = np.array(
            [float(row.steer), float(row.throttle), float(row.brake)], np.float32)
        future_controls = np.array([
            [float(item.steer), float(item.throttle), float(item.brake)]
            for item in future_rows
        ], np.float32)
        target_speed = np.empty(0, np.float32)
        if self.v4:
            profile = np.asarray(
                [float(row[name]) for name in target_speed_columns()], np.float32)
            target_speed = np.concatenate((
                profile / SPEED_SCALE_MPS,
                np.asarray([(profile[0] - float(row.speed_mps)) / SPEED_SCALE_MPS],
                           np.float32)))
        current_pedal_weight = float(getattr(row, "pedal_loss_weight", 1.0))
        future_pedal_weight = np.asarray([
            float(getattr(item, "pedal_loss_weight", 1.0)) for item in future_rows
        ], np.float32)
        source_id = {"normal": 0, "synthetic_recovery": 1, "dagger": 2}.get(
            str(getattr(row, "source_type", "normal")), 0)
        return {
            "image": image_tensor,
            "ego": torch.from_numpy(ego.copy()),
            "ego_raw": torch.from_numpy(ego_raw.copy()),
            "racing_line": torch.from_numpy(racing_line.copy()),
            "command": torch.from_numpy(command),
            "target_speed": torch.from_numpy(target_speed.copy()),
            "waypoints": torch.from_numpy(waypoints),
            "future_mask": torch.tensor(future_mask, dtype=torch.float32),
            "controls": torch.from_numpy(controls),
            "future_controls": torch.from_numpy(future_controls),
            "pedal_loss_weight": torch.tensor(current_pedal_weight, dtype=torch.float32),
            "future_pedal_loss_weight": torch.from_numpy(future_pedal_weight),
            "source_id": torch.tensor(source_id, dtype=torch.int64),
            "speed": torch.tensor([float(row.speed_mps)], dtype=torch.float32),
        }
