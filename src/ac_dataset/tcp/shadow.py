"""Read-only live TCP-AC inference for Assetto Corsa.

This module deliberately has no controller/vJoy output. It captures the same
RGB crop used for training, reads telemetry, predicts controls, and records
the predictions beside the driver's actual inputs for later inspection.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .config import (
    EGO_FEATURE_NAMES, FLYING_LAP_COMMAND_INDEX, POSITION_SCALE_M,
    RACING_LINE_LOOKAHEADS_M, RACING_LINE_SCALE_M, SPEED_SCALE_MPS,
    STANDING_LAP_COMMAND_INDEX,
    model_schema_metadata, normalization_metadata, target_speed_input_width,
)
from .data import normalize_ego_features, preprocess_rgb, to_local


TRAINING_FRAME_SHAPE = (945, 1920, 3)


def parse_region(value: str) -> tuple[int, int, int, int]:
    fields = tuple(int(item.strip()) for item in value.split(","))
    if len(fields) != 4 or fields[2] <= 0 or fields[3] <= 0:
        raise argparse.ArgumentTypeError("region must be left,top,width,height")
    return fields


def observed_controls(state: dict) -> dict[str, float | str | None]:
    throttle = _finite(state.get("accStatus"))
    brake = _finite(state.get("brakeStatus"))
    steer = _finite(state.get("steerNorm"))
    steer_angle_deg = _finite(state.get("steerAngle"))
    source = "steerNorm"
    if steer is None:
        candidate = steer_angle_deg
        if candidate is not None and abs(candidate) <= 1.05:
            steer, source = candidate, "steerAngle"
        else:
            source = "unavailable"
    signed = None if throttle is None or brake is None else throttle - brake
    return {
        "actual_steer": steer, "actual_steer_source": source,
        "actual_steer_angle_deg": steer_angle_deg,
        "actual_throttle": throttle, "actual_brake": brake,
        "actual_signed_accel": signed,
    }


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class RacingLineTarget:
    def __init__(self, table: pd.DataFrame,
                 lookaheads_m=RACING_LINE_LOOKAHEADS_M,
                 expert_profile=None, normalization: dict | None = None):
        required = {"normalized_position", "x", "y", "dist"}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"racing line is missing columns: {sorted(missing)}")
        self.norm = table["normalized_position"].to_numpy(float)
        self.dist = table["dist"].to_numpy(float)
        self.xy = table[["x", "y"]].to_numpy(float)
        speed_name = next((name for name in ("ai_speed_mps", "ai_speed_ms", "ai_speed_kmh")
                           if name in table), None)
        self.speed_mps = (np.zeros(len(table), float) if speed_name is None
                          else table[speed_name].to_numpy(float))
        if speed_name is not None and speed_name.endswith("kmh"):
            self.speed_mps /= 3.6
        self.length = float(self.dist[-1] + np.linalg.norm(self.xy[0] - self.xy[-1]))
        # Keep the spline-indexed samples for the learned model (the exact
        # training distribution), but also build a closed physical polyline for
        # controller feedback and safety distances.  AC's spline parameter and
        # the AI-line normalized_position are not guaranteed to identify the
        # same nearest point on track.
        self._segment_start = self.xy
        self._segment_end = np.roll(self.xy, -1, axis=0)
        self._segment_vector = self._segment_end - self._segment_start
        self._segment_length = np.linalg.norm(self._segment_vector, axis=1)
        self._segment_length_sq = np.sum(self._segment_vector ** 2, axis=1)
        self._segment_distance = np.concatenate((
            np.asarray([0.0]), np.cumsum(self._segment_length[:-1])))
        self.length = float(np.sum(self._segment_length))
        self.lookaheads_m = tuple(float(value) for value in lookaheads_m)
        self.expert_profile = expert_profile
        self.normalization = normalization
        if self.lookaheads_m != tuple(RACING_LINE_LOOKAHEADS_M):
            raise ValueError(
                f"live lookaheads {self.lookaheads_m} != training {RACING_LINE_LOOKAHEADS_M}")

    @classmethod
    def from_csv(cls, path: Path):
        return cls(pd.read_csv(path))

    @classmethod
    def from_prepared(cls, prepared: Path, line_path: Path):
        profile_path = prepared / "expert_speed_profile_v5.npz"
        if not profile_path.exists():
            return cls.from_csv(line_path)
        from .profile_v5 import ExpertSpeedProfileV5
        normalization = json.loads(
            (prepared / "normalization.json").read_text(encoding="utf-8"))
        return cls(pd.read_csv(line_path), expert_profile=ExpertSpeedProfileV5(profile_path),
                   normalization=normalization)

    def target_world(self, spline_position: float) -> np.ndarray:
        return self.target_world_points(spline_position)[-1]

    def target_world_points(self, spline_position: float) -> np.ndarray:
        if self.expert_profile is not None:
            return self.expert_profile.line_points(spline_position, self.lookaheads_m)
        indices = self.target_indices(spline_position)
        return np.asarray(self.xy[indices], dtype=np.float32)

    def target_indices(self, spline_position: float) -> np.ndarray:
        spline = float(spline_position) % 1.0
        current_dist = float(np.interp(spline, self.norm, self.dist))
        indices = []
        for lookahead in self.lookaheads_m:
            target_dist = (current_dist + lookahead) % self.length
            index = min(int(np.searchsorted(self.dist, target_dist)), len(self.xy) - 1)
            indices.append(index)
        return np.asarray(indices, dtype=np.int64)

    def target_speed_features(self, spline_position: float,
                              current_speed_mps: float,
                              phase: str = "flying") -> tuple[np.ndarray, np.ndarray]:
        if self.expert_profile is not None:
            profile = self.expert_profile.speeds(
                spline_position, phase, self.lookaheads_m)
            physical = np.concatenate((profile, np.asarray(
                [profile[0] - abs(float(current_speed_mps))], np.float32)))
            spec = self.normalization["target_speed_features_mps"]
            normalized = ((physical - np.asarray(spec["mean"], np.float32)) /
                          np.asarray(spec["std"], np.float32)).astype(np.float32)
            return profile, normalized
        profile = self.speed_mps[self.target_indices(spline_position)].astype(np.float32)
        normalized = np.concatenate((
            profile / SPEED_SCALE_MPS,
            np.asarray([(profile[0] - abs(float(current_speed_mps))) / SPEED_SCALE_MPS],
                       np.float32)))
        return profile, normalized

    def reference_world(self, spline_position: float) -> np.ndarray:
        spline = float(spline_position) % 1.0
        return np.asarray([
            np.interp(spline, self.norm, self.xy[:, 0]),
            np.interp(spline, self.norm, self.xy[:, 1]),
        ], dtype=np.float32)

    def indexed_deviation_m(self, spline_position: float, origin_xy) -> float:
        """Legacy diagnostic: distance to the point selected by AC spline."""
        reference = self.reference_world(spline_position)
        return float(np.linalg.norm(np.asarray(origin_xy, float) - reference))

    def nearest_projection(self, origin_xy) -> tuple[np.ndarray, float, float]:
        """Return nearest line point, closed-path distance, and Euclidean error."""
        origin = np.asarray(origin_xy, dtype=float).reshape(2)
        relative = origin - self._segment_start
        denominator = np.where(self._segment_length_sq > 1e-12,
                               self._segment_length_sq, 1.0)
        fraction = np.clip(
            np.sum(relative * self._segment_vector, axis=1) / denominator,
            0.0, 1.0)
        projected = self._segment_start + fraction[:, None] * self._segment_vector
        errors_sq = np.sum((projected - origin) ** 2, axis=1)
        index = int(np.argmin(errors_sq))
        distance = (self._segment_distance[index]
                    + fraction[index] * self._segment_length[index]) % self.length
        return (projected[index].astype(np.float32), float(distance),
                float(np.sqrt(errors_sq[index])))

    def world_points_from_distance(self, distance_m: float,
                                   lookaheads_m=None) -> np.ndarray:
        """Interpolate physical AI-line points along the closed polyline."""
        lookaheads = self.lookaheads_m if lookaheads_m is None else tuple(lookaheads_m)
        targets = (float(distance_m) + np.asarray(lookaheads, float)) % self.length
        indices = np.searchsorted(self._segment_distance, targets, side="right") - 1
        indices = np.clip(indices, 0, len(self._segment_start) - 1)
        offsets = targets - self._segment_distance[indices]
        lengths = self._segment_length[indices]
        fractions = np.divide(offsets, lengths, out=np.zeros_like(offsets),
                              where=lengths > 1e-12)
        return np.asarray(
            self._segment_start[indices]
            + fractions[:, None] * self._segment_vector[indices],
            dtype=np.float32)

    def deviation_m(self, spline_position: float, origin_xy) -> float:
        """Physical nearest-line deviation; spline is retained for API stability."""
        del spline_position
        return self.nearest_projection(origin_xy)[2]

    def control_local_points(self, origin_xy,
                             yaw: float) -> tuple[np.ndarray, float, float]:
        """Physical racing-line reference for classical feedback control."""
        _, distance, deviation = self.nearest_projection(origin_xy)
        world = self.world_points_from_distance(distance)
        return (to_local(world, np.asarray(origin_xy, np.float32), float(yaw)),
                distance, deviation)

    def target_local(self, spline_position: float, origin_xy, yaw: float) -> tuple[np.ndarray, np.ndarray]:
        local_m, normalized = self.local_points(spline_position, origin_xy, yaw)
        return local_m[-1], normalized[-1]

    def local_points(self, spline_position: float, origin_xy,
                     yaw: float) -> tuple[np.ndarray, np.ndarray]:
        world = self.target_world_points(spline_position)
        local_m = to_local(world, np.asarray(origin_xy, np.float32), float(yaw))
        if self.expert_profile is not None:
            spec = self.normalization["racing_line_local_m"]
            normalized = ((local_m - np.asarray(spec["mean"], np.float32)) /
                          np.asarray(spec["std"], np.float32)).astype(np.float32)
            return local_m, normalized
        return local_m, local_m / RACING_LINE_SCALE_M


def discover_racing_line(prepared: Path) -> Path:
    manifest = pd.read_parquet(
        prepared / "manifest.parquet", columns=["session", "image_path"])
    if manifest.empty:
        raise RuntimeError("prepared manifest is empty")
    checked: list[Path] = []
    # Early legacy sessions did not export TRACK_INFO. Search every prepared
    # session rather than assuming the first manifest row owns the shared line.
    for raw_path in manifest.drop_duplicates("session")["image_path"]:
        image = Path(str(raw_path))
        if len(image.parents) < 3:
            continue
        candidate = image.parents[2] / "TRACK_INFO" / "AI_RACING_LINE.csv"
        if candidate in checked:
            continue
        checked.append(candidate)
        if candidate.exists():
            return candidate
    preview = ", ".join(str(path) for path in checked[:3])
    raise FileNotFoundError(
        f"could not find AI_RACING_LINE.csv in any prepared session; checked {len(checked)} "
        f"session path(s), including: {preview}")


class ShadowPredictor:
    def __init__(self, checkpoint_path: Path, device_name: str = "auto",
                 prepared: Path | None = None):
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        recipe = checkpoint.get("training_recipe", {})
        recipe_version = int(recipe.get("version", 0))
        requires_causal_gate = bool(recipe.get(
            "requires_full_causal_gate", recipe_version in (6, 7)))
        if requires_causal_gate and not checkpoint.get("full_causal_gate_passed", False):
            raise RuntimeError(
                "v6 checkpoint failed or did not run the full causal gate; "
                "shadow/closed-loop inference is blocked")
        schema = checkpoint.get("model_schema")
        self.v5 = isinstance(schema, dict) and int(schema.get("version", 0)) == 5
        if self.v5:
            if prepared is None:
                raise ValueError("v5 live inference requires --prepared")
            from .config_v5 import TARGET_SPEED_WIDTH
            from .model_v5 import build_model_v5
            from .profile_v5 import sha256_file
            disk_schema = json.loads((prepared / "schema.json").read_text(encoding="utf-8"))
            if schema != disk_schema["model_schema"]:
                raise RuntimeError("v5 checkpoint/prepared schema mismatch")
            for name, expected_hash in checkpoint["artifacts"].items():
                if sha256_file(prepared / name) != expected_hash:
                    raise RuntimeError(f"v5 live artifact mismatch: {name}")
            self.normalization_log = json.loads(
                (prepared / "normalization.json").read_text(encoding="utf-8"))
            if checkpoint.get("normalization") != self.normalization_log:
                raise RuntimeError("v5 checkpoint/live normalization mismatch")
            self.target_speed_width = TARGET_SPEED_WIDTH
            model_builder = lambda: build_model_v5(False)
        else:
            from .model import build_model
            self.target_speed_width = target_speed_input_width(schema)
            actual = checkpoint.get("normalization")
            expected = normalization_metadata(bool(self.target_speed_width))
            if actual != expected:
                raise RuntimeError(f"checkpoint normalization mismatch: {actual} != {expected}")
            if schema != model_schema_metadata(bool(self.target_speed_width)):
                raise RuntimeError(
                    "checkpoint uses the old model schema; retrain with v2 prepared data")
            self.normalization_log = expected
            model_builder = lambda: build_model(False, target_speed_width=self.target_speed_width)
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.model = model_builder().to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self._image_mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32, device=self.device,
        ).view(1, 3, 1, 1)
        self._image_std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32, device=self.device,
        ).view(1, 3, 1, 1)
        self.preprocess_backend = "torch_gpu_bilinear" if self.device.type == "cuda" else "pil_cpu"
        self.epoch = int(checkpoint["epoch"])
        self.val_loss = float(checkpoint.get("val_loss", float("nan")))
        # Every v5-schema training sample contains a standing/flying bit in the
        # six-wide route command (see data.route_command).  Early v5/v6
        # checkpoints unfortunately omitted the descriptive ``lap_phase``
        # metadata, so checking that metadata alone silently produced the
        # never-trained live command [0, 0, 0, 1, 0, 0].  Schema v5 is itself
        # sufficient evidence that the two phase bits are part of the input.
        phase_metadata = checkpoint.get("lap_phase")
        if isinstance(phase_metadata, dict) and "enabled" in phase_metadata:
            self.lap_phase_enabled = bool(phase_metadata["enabled"])
        else:
            self.lap_phase_enabled = self.v5 or phase_metadata is not None
        # Pay CUDA/cuDNN's one-time setup cost before the first logged frame.
        dummy_image = torch.zeros((1, 3, 256, 512), dtype=torch.float32, device=self.device)
        dummy_ego = torch.zeros(
            (1, len(EGO_FEATURE_NAMES)), dtype=torch.float32, device=self.device)
        dummy_line = torch.zeros(
            (1, len(RACING_LINE_LOOKAHEADS_M), 2),
            dtype=torch.float32, device=self.device)
        dummy_command = torch.zeros((1, 6), dtype=torch.float32, device=self.device)
        dummy_command[0, 3] = 1.0
        dummy_target_speed = torch.zeros(
            (1, self.target_speed_width), dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            self.model(dummy_image, dummy_ego, dummy_line, dummy_command,
                       dummy_target_speed if self.target_speed_width else None)
            # Warm the full-resolution upload/resize kernel as well; otherwise
            # the first logged frame pays a one-time preprocessing setup cost.
            self._preprocess(np.zeros(TRAINING_FRAME_SHAPE, dtype=np.uint8))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _preprocess(self, rgb: np.ndarray):
        import torch
        import torch.nn.functional as F

        if self.device.type != "cuda":
            return preprocess_rgb(rgb).unsqueeze(0).to(self.device)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("RGB input must be an HxWx3 uint8 array")
        if not rgb.flags.c_contiguous:
            rgb = np.ascontiguousarray(rgb)
        image = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).unsqueeze(0)
        image = image.float().div_(255.0)
        image = F.interpolate(
            image, size=(256, 512), mode="bilinear", align_corners=False, antialias=True,
        )
        return (image - self._image_mean) / self._image_std

    def predict(self, rgb: np.ndarray, ego: np.ndarray, racing_line: np.ndarray,
                standing_lap: bool | None = None,
                target_speed: np.ndarray | None = None,
                full_diagnostics: bool = True,
                actuation_waypoints: bool = False) -> dict:
        import torch

        preprocess_started = time.perf_counter()
        image = self._preprocess(rgb)
        ego_tensor = torch.as_tensor(
            ego, dtype=torch.float32, device=self.device).reshape(1, len(EGO_FEATURE_NAMES))
        line_tensor = torch.as_tensor(
            racing_line, dtype=torch.float32, device=self.device).reshape(
                1, len(RACING_LINE_LOOKAHEADS_M), 2)
        command = torch.zeros((1, 6), dtype=torch.float32, device=self.device)
        command[0, 3] = 1.0
        if self.lap_phase_enabled:
            if standing_lap is None:
                raise ValueError("phase-aware checkpoint requires standing_lap")
            command[0, STANDING_LAP_COMMAND_INDEX if standing_lap
                    else FLYING_LAP_COMMAND_INDEX] = 1.0
        target_tensor = None
        if self.target_speed_width:
            if target_speed is None:
                raise ValueError("v4 checkpoint requires target_speed features")
            target_tensor = torch.as_tensor(
                target_speed, dtype=torch.float32, device=self.device).reshape(
                    1, self.target_speed_width)
        # Detailed timing needs a hard barrier after preprocessing. In the
        # real-time actuation path this barrier only serializes the CPU/GPU;
        # the final controls copy below already provides the required barrier.
        if self.device.type == "cuda" and full_diagnostics:
            torch.cuda.synchronize(self.device)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
        started = time.perf_counter()
        with torch.inference_mode():
            pred = self.model(image, ego_tensor, line_tensor, command, target_tensor)
        if self.device.type == "cuda" and full_diagnostics:
            torch.cuda.synchronize(self.device)
        controls = pred["controls"][0].detach().cpu().numpy()
        # In lite mode the blocking controls copy is the single synchronization
        # point and inference_ms includes all queued GPU work needed to actuate.
        inference_ms = (time.perf_counter() - started) * 1000.0
        basic = {
            "pred_steer": float(controls[0]),
            "pred_throttle": float(controls[1]),
            "pred_brake": float(controls[2]),
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
        }
        # Actuation needs only the current controls. The other heads require
        # several extra GPU-to-CPU copies and are diagnostic-only.
        if not full_diagnostics and actuation_waypoints:
            if self.v5:
                wp_spec = self.normalization_log["waypoints_local_m"]
                waypoints = (pred["waypoints"][0].detach().cpu().numpy() *
                             np.asarray(wp_spec["std"], np.float32) +
                             np.asarray(wp_spec["mean"], np.float32))
            else:
                waypoints = (pred["waypoints"][0].detach().cpu().numpy()
                             * POSITION_SCALE_M)
            return {**basic, "pred_waypoints_m": waypoints.tolist()}
        if not full_diagnostics:
            return basic
        if self.v5:
            wp_spec = self.normalization_log["waypoints_local_m"]
            speed_spec = self.normalization_log["future_speed_mps"]
            waypoints = (pred["waypoints"][0].detach().cpu().numpy() *
                         np.asarray(wp_spec["std"], np.float32) +
                         np.asarray(wp_spec["mean"], np.float32))
            future_speeds = (pred["future_speeds"][0, :, 0].detach().cpu().numpy() *
                             np.asarray(speed_spec["std"], np.float32) +
                             np.asarray(speed_spec["mean"], np.float32))
            plan = pred["plan_controls"][0].detach().cpu().numpy()
            residual = pred["visual_residual"][0].detach().cpu().numpy()
            return {
                **basic,
                "pred_waypoints_m": waypoints.tolist(),
                "pred_future_speeds_mps": future_speeds.tolist(),
                "pred_future_controls": pred["future_controls"][0].detach().cpu().numpy().tolist(),
                "plan_steer": float(plan[0]), "plan_throttle": float(plan[1]),
                "plan_brake": float(plan[2]),
                "visual_residual_steer": float(residual[0]),
                "visual_residual_throttle": float(residual[1]),
                "visual_residual_brake": float(residual[2]),
            }
        return {
            **basic,
            "pred_speed_mps": float(pred["speed"][0, 0].detach().cpu()) * SPEED_SCALE_MPS,
            "pred_waypoints_m": (pred["waypoints"][0].detach().cpu().numpy() * POSITION_SCALE_M).tolist(),
            "pred_future_controls": pred["future_controls"][0].detach().cpu().numpy().tolist(),
        }

    def normalize_ego(self, raw: np.ndarray, legacy_normalized: np.ndarray) -> np.ndarray:
        if not self.v5:
            return legacy_normalized
        spec = self.normalization_log["ego"]
        return ((raw - np.asarray(spec["mean"], np.float32)) /
                np.asarray(spec["std"], np.float32)).astype(np.float32)


def live_ego_features(state: dict) -> tuple[np.ndarray, np.ndarray]:
    """Read the exact local signals exported into EGO_INFO during preparation."""
    names = ("speed", "local_velocity_x", "local_velocity_y",
             "angular_velocity_y", "accelY", "accelX")
    values = tuple(_finite(state.get(name)) for name in names)
    missing = [name for name, value in zip(names, values) if value is None]
    if missing:
        raise ValueError(f"telemetry missing/non-finite v2 ego inputs: {missing}")
    speed, longitudinal, lateral, yaw_rate, accel_y_g, accel_x_g = values
    gravity = 9.80665
    raw = np.asarray([
        abs(speed), longitudinal, lateral, yaw_rate,
        accel_y_g * gravity, accel_x_g * gravity,
    ], dtype=np.float32)
    return raw, normalize_ego_features(raw)


def _required_telemetry(state: dict) -> tuple[float, float, float, float, float]:
    names = ("world_position_x", "world_position_y", "yaw", "speed", "NormalizedSplinePosition")
    values = tuple(_finite(state.get(name)) for name in names)
    missing = [name for name, value in zip(names, values) if value is None]
    if missing:
        raise ValueError(f"telemetry missing/non-finite: {missing}")
    return values  # type: ignore[return-value]


def standing_lap_from_state(state: dict) -> bool:
    """Map live AC LapCount=0 to prepared dataset lap 1 (standing start)."""
    raw = state.get("LapCount", state.get("completedLaps"))
    value = _finite(raw)
    if value is None:
        raise ValueError("telemetry missing/non-finite: ['LapCount']")
    return value < 1.0


def run(args) -> Path:
    from ac_dataset.recorder import ACTelemetrySource
    from ac_dataset.recorder.screen_capture import ScreenCapture

    racing_line_path = args.racing_line or discover_racing_line(args.prepared)
    target_provider = RacingLineTarget.from_prepared(args.prepared, racing_line_path)
    predictor = ShadowPredictor(args.checkpoint, args.device, args.prepared)
    run_dir = args.output / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "mode": "shadow_read_only", "actuation_enabled": False,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": predictor.epoch,
        "checkpoint_val_loss": predictor.val_loss, "prepared": str(args.prepared.resolve()),
        "racing_line": str(racing_line_path.resolve()),
        "racing_line_lookaheads_m": list(RACING_LINE_LOOKAHEADS_M),
        "capture_region": list(args.region), "capture_backend": args.backend,
        "window_title": args.window_title, "hz": args.hz, "device": str(predictor.device),
        "preprocess_backend": predictor.preprocess_backend,
        "lap_phase_enabled": predictor.lap_phase_enabled,
        "start_delay_s": args.start_delay,
        "wait_for_motion": args.wait_for_motion,
        "motion_threshold_mps": args.motion_threshold_mps,
        "motion_hold_s": args.motion_hold,
        "save_frame_every": args.save_frame_every,
        "normalization": predictor.normalization_log,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    capture = None
    telemetry = None
    count = misses = repeated = 0
    duration_started = time.perf_counter()
    last_signature = None
    log_path = run_dir / "shadow.jsonl"
    try:
        telemetry = ACTelemetrySource(args.host, args.port).start()
        if not telemetry.wait_first(args.telemetry_timeout):
            raise RuntimeError("connected but no telemetry state arrived")
        capture = ScreenCapture(args.region, args.backend, args.monitor, args.window_title)
        # Desktop Duplication can spend over a second creating resources on
        # its first grab. Discard that frame before the timed recording.
        capture.grab()
        print(f"[SAFE] shadow/read-only mode; NO controls will be sent to AC")
        print(f"[ready] device={predictor.device} checkpoint_epoch={predictor.epoch} capture={capture.backend}")
        if args.start_delay > 0:
            print(f"[delay] recording starts in {args.start_delay:g}s; switch to the AC window NOW")
            time.sleep(args.start_delay)
        if args.wait_for_motion:
            print(f"[armed] waiting for speed >= {args.motion_threshold_mps:g}m/s "
                  f"for {args.motion_hold:g}s")
            moving_since = None
            while True:
                state = telemetry.latest_state()
                speed = None if state is None else _finite(state.get("speed"))
                now = time.perf_counter()
                if speed is not None and abs(speed) >= args.motion_threshold_mps:
                    moving_since = now if moving_since is None else moving_since
                    if now - moving_since >= args.motion_hold:
                        break
                else:
                    moving_since = None
                time.sleep(0.01)
        print("[recording] shadow logging started")
        duration_started = time.perf_counter()
        period = 1.0 / args.hz
        next_tick = time.perf_counter()
        with log_path.open("a", encoding="utf-8", buffering=1) as stream:
            while args.duration <= 0 or time.perf_counter() - duration_started < args.duration:
                tick_started = time.perf_counter()
                state = telemetry.latest_state()
                capture_started = time.perf_counter()
                frame = capture.grab()
                capture_ms = (time.perf_counter() - capture_started) * 1000.0
                if state is None or frame is None:
                    misses += 1
                else:
                    if tuple(frame.shape) != TRAINING_FRAME_SHAPE and not args.allow_size_mismatch:
                        raise RuntimeError(
                            f"capture shape {tuple(frame.shape)} != training {TRAINING_FRAME_SHAPE}; "
                            "fix --region or explicitly use --allow-size-mismatch")
                    x, y, yaw, speed, spline = _required_telemetry(state)
                    line_local_m, line_normalized = target_provider.local_points(
                        spline, (x, y), yaw)
                    ego_raw, ego_normalized = live_ego_features(state)
                    standing_lap = standing_lap_from_state(state)
                    target_speed_mps, target_speed_features = target_provider.target_speed_features(
                        spline, speed, "standing" if standing_lap else "flying")
                    signature = int(frame[::32, ::32].astype(np.uint64).sum())
                    is_repeated = signature == last_signature
                    repeated += int(is_repeated)
                    last_signature = signature
                    pred = predictor.predict(
                        frame, predictor.normalize_ego(ego_raw, ego_normalized), line_normalized,
                        standing_lap=standing_lap,
                        target_speed=target_speed_features if predictor.target_speed_width else None)
                    actual = observed_controls(state)
                    saved_frame = None
                    if count == 0:
                        from PIL import Image
                        saved_frame = run_dir / "first_frame.jpg"
                        Image.fromarray(frame).save(saved_frame, quality=90)
                    elif args.save_frame_every and count % args.save_frame_every == 0:
                        from PIL import Image
                        frame_dir = run_dir / "frames"
                        frame_dir.mkdir(exist_ok=True)
                        saved_frame = frame_dir / f"{count:06d}.jpg"
                        Image.fromarray(frame).save(saved_frame, quality=85)
                    row = {
                        "index": count, "wall_time": datetime.now().astimezone().isoformat(),
                        "elapsed_s": time.perf_counter() - duration_started,
                        "telemetry_received": telemetry.received,
                        "position_x": x, "position_y": y, "yaw_rad": yaw,
                        "speed_mps": abs(speed), "spline_position": spline,
                        "lap_phase": "standing" if standing_lap else "flying",
                        "ego_raw": ego_raw.tolist(),
                        "ego_normalized": ego_normalized.tolist(),
                        "racing_line_local_m": line_local_m.tolist(),
                        "racing_line_normalized": line_normalized.tolist(),
                        "target_speed_profile_mps": target_speed_mps.tolist(),
                        "target_speed_features": target_speed_features.tolist(),
                        "frame_shape": list(frame.shape), "frame_repeated": is_repeated,
                        "saved_frame": None if saved_frame is None else str(saved_frame.resolve()),
                        "capture_ms": capture_ms,
                        **actual, **pred,
                    }
                    if actual["actual_steer"] is not None:
                        row["steer_error"] = pred["pred_steer"] - actual["actual_steer"]
                    if actual["actual_throttle"] is not None:
                        row["throttle_error"] = pred["pred_throttle"] - actual["actual_throttle"]
                    if actual["actual_brake"] is not None:
                        row["brake_error"] = pred["pred_brake"] - actual["actual_brake"]
                    row["loop_ms"] = (time.perf_counter() - tick_started) * 1000.0
                    row["deadline_missed"] = row["loop_ms"] > period * 1000.0
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
                    if count == 1 or count % max(1, round(args.hz)) == 0:
                        print(f"[{count:06d}] {row['loop_ms']:.1f}ms steer={pred['pred_steer']:+.3f} "
                              f"gas={pred['pred_throttle']:.3f} brake={pred['pred_brake']:.3f}")
                    if args.max_frames and count >= args.max_frames:
                        break
                next_tick += period
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -period:
                    next_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("[stop] Ctrl+C received")
    finally:
        if capture is not None:
            capture.close()
        if telemetry is not None:
            telemetry.close()

    elapsed = time.perf_counter() - duration_started
    summary = {
        "mode": "shadow_read_only", "actuation_enabled": False,
        "frames": count, "missing_frame_or_state": misses, "repeated_frame_signatures": repeated,
        "elapsed_s": elapsed, "effective_hz": count / elapsed if elapsed else 0.0,
        "log": str(log_path.resolve()),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only TCP-AC live shadow inference (never actuates AC)")
    parser.add_argument("--checkpoint", type=Path, default=Path("training/tcp_lagunaseca/checkpoints/best.pt"))
    parser.add_argument("--prepared", type=Path, default=Path("training/tcp_lagunaseca"))
    parser.add_argument("--racing-line", type=Path)
    parser.add_argument("--output", type=Path, default=Path("training/tcp_lagunaseca/shadow"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--hz", type=float, default=40.0)
    parser.add_argument("--duration", type=float, default=60.0, help="seconds; <=0 runs until Ctrl+C")
    parser.add_argument("--start-delay", type=float, default=5.0,
                        help="seconds after ready before recording, to refocus AC")
    parser.add_argument("--wait-for-motion", action=argparse.BooleanOptionalAction, default=True,
                        help="arm first and start logging only after sustained vehicle motion")
    parser.add_argument("--motion-threshold-mps", type=float, default=1.0)
    parser.add_argument("--motion-hold", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--save-frame-every", type=int, default=0,
                        help="save one debug JPEG every N logged frames; 0 disables periodic saving")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2345)
    parser.add_argument("--telemetry-timeout", type=float, default=5.0)
    parser.add_argument("--backend", default="dxcam", choices=("auto", "wgc", "dxcam", "mss"))
    parser.add_argument("--region", type=parse_region, default=parse_region("0,80,1920,945"))
    parser.add_argument("--monitor", type=int, default=0)
    parser.add_argument("--window-title")
    parser.add_argument("--allow-size-mismatch", action="store_true")
    args = parser.parse_args()
    if args.hz <= 0:
        parser.error("--hz must be positive")
    if args.start_delay < 0:
        parser.error("--start-delay cannot be negative")
    if args.motion_threshold_mps < 0 or args.motion_hold < 0:
        parser.error("motion threshold/hold cannot be negative")
    if args.save_frame_every < 0:
        parser.error("--save-frame-every cannot be negative")
    run(args)


if __name__ == "__main__":
    main()
