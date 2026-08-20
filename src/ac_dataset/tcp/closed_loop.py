"""Guarded TCP-AC closed-loop runner using the independent vJoy bridge."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import time

import numpy as np

from ac_dataset.control import ControlClient, SteerMap
from .config import RACING_LINE_LOOKAHEADS_M, normalization_metadata
from .shadow import (
    TRAINING_FRAME_SHAPE, RacingLineTarget, ShadowPredictor, _required_telemetry,
    discover_racing_line, live_ego_features, observed_controls, parse_region,
    standing_lap_from_state,
)


def audio_cue(enabled: bool, event: str) -> None:
    """Play short Windows tones without making audio part of the control path."""
    if not enabled:
        return
    try:
        import winsound
        if event == "armed":
            winsound.Beep(1000, 180)
        elif event == "completed":
            winsound.Beep(1200, 140)
            winsound.Beep(1500, 180)
        else:
            winsound.Beep(500, 350)
    except (ImportError, RuntimeError):
        pass


def control_lap_phases(telemetry_standing_lap: bool,
                       launch_complete: bool) -> tuple[bool, bool]:
    """Separate the learned command phase from the speed-profile phase.

    The model must receive the same LapCount-derived command used in training.
    The speed controller may latch to the flying profile after launch so the
    standing grid-speed samples do not brake the car near the end of lap one.
    """
    model_standing = bool(telemetry_standing_lap)
    speed_profile_standing = model_standing and not bool(launch_complete)
    return model_standing, speed_profile_standing


def slip_limited_throttle(throttle: float, slip_deg: float,
                          cut_deg: float, zero_deg: float) -> tuple[float, dict]:
    """Linearly remove positive throttle between two slip-angle thresholds."""
    throttle = float(throttle)
    slip_deg = abs(float(slip_deg))
    if throttle <= 0 or slip_deg <= cut_deg:
        return throttle, {}
    span = float(zero_deg) - float(cut_deg)
    if span <= 0:
        raise ValueError("slip throttle zero angle must exceed cut angle")
    fraction = float(np.clip((slip_deg - cut_deg) / span, 0.0, 1.0))
    limited = throttle * (1.0 - fraction)
    return limited, {
        "slip_throttle_cut_active": True,
        "slip_throttle_before": throttle,
        "slip_throttle_after": limited,
        "slip_cut_angle_deg": slip_deg,
    }


class StallDetector:
    """Detect a persistent stop only after the car has reached driving speed."""

    def __init__(self, arm_speed_mps: float, stop_speed_mps: float, duration_s: float):
        self.arm_speed_mps = float(arm_speed_mps)
        self.stop_speed_mps = float(stop_speed_mps)
        self.duration_s = float(duration_s)
        self.armed = False
        self.stopped_since: float | None = None

    def update(self, speed_mps: float, now: float) -> bool:
        speed_mps = abs(float(speed_mps))
        if speed_mps >= self.arm_speed_mps:
            self.armed = True
        if not self.armed:
            return False
        if speed_mps >= self.stop_speed_mps:
            self.stopped_since = None
            return False
        if self.stopped_since is None:
            self.stopped_since = float(now)
            return False
        return float(now) - self.stopped_since >= self.duration_s


class ExpertSpeedProfile:
    """Phase-aware expert speed targets indexed by circular track position."""

    def __init__(self, profiles: dict[str, np.ndarray], quantile: float,
                 sample_counts: dict[str, int]):
        if not profiles:
            raise ValueError("speed profiles cannot be empty")
        widths = {len(np.asarray(values)) for values in profiles.values()}
        if len(widths) != 1 or next(iter(widths)) < 2:
            raise ValueError("speed profiles must share a width of at least two bins")
        self.profiles = {
            phase: np.asarray(values, dtype=float) for phase, values in profiles.items()
        }
        if any(not np.isfinite(values).all() for values in self.profiles.values()):
            raise ValueError("speed profiles must contain only finite values")
        self.bins = next(iter(widths))
        self.quantile = float(quantile)
        self.sample_counts = {key: int(value) for key, value in sample_counts.items()}

    @staticmethod
    def _fill_circular(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        valid = np.flatnonzero(np.isfinite(values))
        if not len(valid):
            raise ValueError("cannot build a speed profile without populated bins")
        if len(valid) == 1:
            return np.full_like(values, values[valid[0]], dtype=float)
        indices = np.arange(len(values), dtype=float)
        xp = np.concatenate((valid - len(values), valid, valid + len(values)))
        fp = np.concatenate((values[valid], values[valid], values[valid]))
        return np.interp(indices, xp, fp)

    @classmethod
    def from_prepared(cls, prepared: Path, bins: int = 500,
                      quantile: float = .75) -> "ExpertSpeedProfile":
        import pandas as pd

        manifest_path = prepared / "manifest.parquet"
        if not manifest_path.exists():
            raise FileNotFoundError(f"prepared manifest does not exist: {manifest_path}")
        table = pd.read_parquet(
            manifest_path,
            columns=["session", "lap_phase", "spline_position", "speed_mps", "eligible"],
        )
        split_path = prepared / "split.json"
        if split_path.exists():
            split = json.loads(split_path.read_text(encoding="utf-8"))
            train_sessions = set(split.get("train_sessions", ()))
            if train_sessions:
                table = table[table.session.isin(train_sessions)]
        table = table[
            table.eligible.astype(bool)
            & np.isfinite(table.spline_position)
            & np.isfinite(table.speed_mps)
            & (table.speed_mps >= 0)
        ].copy()
        if table.empty:
            raise ValueError(f"no eligible speed-profile samples in {manifest_path}")
        table["speed_bin"] = np.floor(
            (table.spline_position.astype(float) % 1.0) * bins
        ).astype(int).clip(0, bins - 1)
        profiles = {}
        counts = {}
        for phase in ("standing", "flying"):
            phase_table = table[table.lap_phase == phase]
            if phase_table.empty:
                continue
            raw = np.full(bins, np.nan, dtype=float)
            grouped = phase_table.groupby("speed_bin").speed_mps.quantile(quantile)
            raw[grouped.index.to_numpy(int)] = grouped.to_numpy(float)
            profiles[phase] = cls._fill_circular(raw)
            counts[phase] = len(phase_table)
        if not profiles:
            raise ValueError("manifest has no standing/flying speed-profile samples")
        if "standing" not in profiles:
            profiles["standing"] = profiles["flying"].copy()
            counts["standing"] = 0
        if "flying" not in profiles:
            profiles["flying"] = profiles["standing"].copy()
            counts["flying"] = 0
        return cls(profiles, quantile, counts)

    def target_mps(self, spline_position: float, standing_lap: bool) -> float:
        phase = "standing" if standing_lap else "flying"
        values = self.profiles[phase]
        coordinate = (float(spline_position) % 1.0) * self.bins - .5
        lower = int(np.floor(coordinate))
        fraction = coordinate - lower
        return float(
            values[lower % self.bins] * (1.0 - fraction)
            + values[(lower + 1) % self.bins] * fraction
        )

    def metadata(self) -> dict:
        return {
            "bins": self.bins,
            "quantile": self.quantile,
            "sample_counts": self.sample_counts,
            "phases": sorted(self.profiles),
        }

    def write_csv(self, path: Path) -> None:
        import pandas as pd

        centers = (np.arange(self.bins, dtype=float) + .5) / self.bins
        pd.DataFrame({
            "spline_position": centers,
            **{f"{phase}_target_mps": values
               for phase, values in sorted(self.profiles.items())},
        }).to_csv(path, index=False)


class SpeedProfileGuard:
    """Cut power and add braking only when faster than the expert envelope."""

    def __init__(self, profile: ExpertSpeedProfile, margin_mps: float = 1.5,
                 throttle_cut_range_mps: float = 3.0,
                 full_brake_range_mps: float = 8.0):
        self.profile = profile
        self.margin_mps = float(margin_mps)
        self.throttle_cut_range_mps = float(throttle_cut_range_mps)
        self.full_brake_range_mps = float(full_brake_range_mps)
        if self.margin_mps < 0:
            raise ValueError("speed-profile margin cannot be negative")
        if min(self.throttle_cut_range_mps, self.full_brake_range_mps) <= 0:
            raise ValueError("speed-profile control ranges must be positive")

    def apply(self, speed_mps: float, spline_position: float, standing_lap: bool,
              throttle: float, brake: float, brake_limit: float) -> tuple[float, float, dict]:
        target = self.profile.target_mps(spline_position, standing_lap)
        overspeed = float(speed_mps) - target
        excess = max(0.0, overspeed - self.margin_mps)
        throttle_scale = float(np.clip(
            1.0 - excess / self.throttle_cut_range_mps, 0.0, 1.0))
        brake_floor = float(np.clip(
            excess / self.full_brake_range_mps, 0.0, brake_limit))
        guarded_throttle = float(throttle) * throttle_scale
        guarded_brake = max(float(brake), brake_floor)
        # During an intervention, never let combined pedal demand exceed one.
        # This preserves mild expert overlap but prevents power fighting added braking.
        if excess > 0:
            guarded_throttle = min(guarded_throttle, max(0.0, 1.0 - guarded_brake))
        return guarded_throttle, guarded_brake, {
            "speed_profile_target_mps": target,
            "speed_profile_overspeed_mps": overspeed,
            "speed_guard_active": excess > 0,
            "speed_guard_excess_mps": excess,
            "speed_guard_throttle_scale": throttle_scale,
            "speed_guard_brake_floor": brake_floor,
        }

    def metadata(self) -> dict:
        return {
            "enabled": True,
            **self.profile.metadata(),
            "margin_mps": self.margin_mps,
            "throttle_cut_range_mps": self.throttle_cut_range_mps,
            "full_brake_range_mps": self.full_brake_range_mps,
            "pedal_policy": "during intervention throttle+brake<=1",
        }


class PreviewPIDLongitudinalController:
    """Track an expert speed profile with braking-distance preview and PID feedback.

    The preview converts each future speed target into the fastest speed that can
    still reach it with ``preview_decel_mps2``.  PID then tracks the most
    restrictive of those envelopes.  Derivative gain defaults to zero because
    target-profile steps and live telemetry noise otherwise create pedal kicks.
    """

    def __init__(self, profile: ExpertSpeedProfile, racing_line: RacingLineTarget,
                 lookaheads_m=RACING_LINE_LOOKAHEADS_M,
                 preview_decel_mps2: float = 14.0,
                 kp: float = .12, ki: float = .015, kd: float = 0.0,
                 deadband_mps: float = .25, integral_limit: float = 10.0,
                 target_scale: float = 1.0):
        self.profile = profile
        self.lookaheads_m = np.asarray(lookaheads_m, dtype=float)
        self.preview_decel_mps2 = float(preview_decel_mps2)
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.deadband_mps = float(deadband_mps)
        self.integral_limit = float(integral_limit)
        self.target_scale = float(target_scale)
        if len(self.lookaheads_m) < 1 or self.lookaheads_m[0] != 0:
            raise ValueError("longitudinal lookaheads must start at zero")
        if np.any(np.diff(self.lookaheads_m) < 0):
            raise ValueError("longitudinal lookaheads must be nondecreasing")
        if self.preview_decel_mps2 <= 0:
            raise ValueError("preview deceleration must be positive")
        if min(self.kp, self.ki, self.kd, self.deadband_mps) < 0:
            raise ValueError("PID gains and deadband cannot be negative")
        if self.integral_limit <= 0:
            raise ValueError("PID integral limit must be positive")
        if not 0 < self.target_scale <= 1:
            raise ValueError("longitudinal target scale must be in (0, 1]")

        # RacingLineTarget.dist does not include the final closure segment.
        # Append its known total length so distance lookahead wraps cleanly.
        self.line_dist = np.concatenate((
            np.asarray(racing_line.dist, dtype=float),
            np.asarray([racing_line.length], dtype=float),
        ))
        self.line_norm = np.concatenate((
            np.asarray(racing_line.norm, dtype=float),
            np.asarray([1.0], dtype=float),
        ))
        self.track_length_m = float(racing_line.length)
        self.integral = 0.0
        self.previous_error: float | None = None
        self.previous_speed_mps: float | None = None

    def _targets_mps(self, spline_position: float,
                     standing_lap: bool) -> np.ndarray:
        spline = float(spline_position) % 1.0
        current_distance = float(np.interp(spline, self.line_norm, self.line_dist))
        distances = (current_distance + self.lookaheads_m) % self.track_length_m
        splines = np.interp(distances, self.line_dist, self.line_norm) % 1.0
        return np.asarray([
            self.profile.target_mps(value, standing_lap) for value in splines
        ], dtype=float)

    def update(self, speed_mps: float, spline_position: float, standing_lap: bool,
               dt: float, throttle_limit: float = 1.0,
               brake_limit: float = 1.0) -> tuple[float, float, dict]:
        speed = abs(float(speed_mps))
        dt = float(np.clip(dt, 1e-3, .1))
        raw_targets = self._targets_mps(spline_position, standing_lap)
        targets = raw_targets * self.target_scale
        allowed_now = np.sqrt(np.maximum(
            0.0, targets * targets
            + 2.0 * self.preview_decel_mps2 * self.lookaheads_m))
        limiting_index = int(np.argmin(allowed_now))
        desired_speed = float(allowed_now[limiting_index])
        error = desired_speed - speed

        if self.previous_error is not None and error * self.previous_error < 0:
            self.integral = 0.0
        if abs(error) <= self.deadband_mps:
            effective_error = 0.0
            self.integral *= max(0.0, 1.0 - 2.0 * dt)
        else:
            effective_error = float(np.sign(error) * (abs(error) - self.deadband_mps))
            self.integral = float(np.clip(
                self.integral + effective_error * dt,
                -self.integral_limit, self.integral_limit))

        measured_accel = (0.0 if self.previous_speed_mps is None else
                          (speed - self.previous_speed_mps) / dt)
        effort = (self.kp * effective_error + self.ki * self.integral
                  - self.kd * measured_accel)
        throttle = float(np.clip(effort, 0.0, throttle_limit))
        brake = float(np.clip(-effort, 0.0, brake_limit))
        self.previous_error = error
        self.previous_speed_mps = speed
        return throttle, brake, {
            "longitudinal_controller": "profile_pid",
            "longitudinal_target_mps": desired_speed,
            "longitudinal_profile_now_mps": float(raw_targets[0]),
            "longitudinal_scaled_profile_now_mps": float(targets[0]),
            "longitudinal_error_mps": error,
            "longitudinal_effort": float(effort),
            "longitudinal_integral": self.integral,
            "longitudinal_measured_accel_mps2": float(measured_accel),
            "longitudinal_limiting_lookahead_m": float(
                self.lookaheads_m[limiting_index]),
            "longitudinal_limiting_profile_mps": float(targets[limiting_index]),
        }

    def metadata(self) -> dict:
        return {
            "mode": "profile_pid",
            "profile": self.profile.metadata(),
            "lookaheads_m": self.lookaheads_m.tolist(),
            "preview_decel_mps2": self.preview_decel_mps2,
            "kp": self.kp, "ki": self.ki, "kd": self.kd,
            "deadband_mps": self.deadband_mps,
            "integral_limit": self.integral_limit,
            "target_scale": self.target_scale,
            "pedal_policy": "mutually_exclusive",
        }

    def reset_integral(self) -> None:
        """Prevent recovery-mode speed caps from leaving positive PID windup."""
        self.integral = 0.0
        self.previous_error = None


class WaypointPurePursuitController:
    """Convert learned local waypoints into a smooth, speed-aware steer demand.

    Model waypoints use the shared ``to_local`` lateral/forward metre frame.
    AC's calibrated steer convention is the opposite sign of the stored
    lateral coordinate, hence the minus sign on curvature below.
    """

    def __init__(self, wheelbase_m: float = 3.5,
                 lookahead_base_m: float = 8.0,
                 lookahead_speed_gain_s: float = .35,
                 lookahead_min_m: float = 10.0,
                 lookahead_max_m: float = 35.0,
                 lookahead_curvature_gain: float = 0.0,
                 lookahead_curvature_probe_m: float = 15.0,
                 lookahead_curvature_min_scale: float = 0.5,
                 steer_gain: float = 4.5,
                 steer_rate_per_s: float = 2.5,
                 high_speed_steer_limit: float = .45,
                 cross_track_gain: float = 0.0,
                 cross_track_softening_mps: float = 5.0,
                 cross_track_highspeed_gain: float | None = None,
                 cross_track_lowspeed_mps: float = 28.0,
                 cross_track_highspeed_mps: float = 42.0,
                 recovery_enabled: bool = False,
                 recovery_enter_deviation_m: float = 3.0,
                 recovery_exit_deviation_m: float = 1.5,
                 recovery_slip_angle_rad: float = .12,
                 recovery_exit_slip_angle_rad: float = .06,
                 recovery_stable_s: float = .75,
                 recovery_lookahead_scale: float = 1.25,
                 recovery_speed_scale: float = .90,
                 recovery_max_speed_mps: float = 45.0,
                 recovery_throttle_limit: float = .20,
                 recovery_brake_gain: float = .04,
                 recovery_brake_limit: float = .25,
                 recovery_slip_brake_limit: float = .15):
        self.wheelbase_m = float(wheelbase_m)
        self.lookahead_base_m = float(lookahead_base_m)
        self.lookahead_speed_gain_s = float(lookahead_speed_gain_s)
        self.lookahead_min_m = float(lookahead_min_m)
        self.lookahead_max_m = float(lookahead_max_m)
        # Curvature-adaptive lookahead: shorten the aim distance when the path ahead
        # is tight so pure pursuit stops chord-cutting slow apexes, while leaving
        # straights/fast sweepers long. Tightness is sensed from the learned path at
        # a fixed probe distance. gain=0 disables it (flat, speed-only lookahead).
        self.lookahead_curvature_gain = float(lookahead_curvature_gain)
        self.lookahead_curvature_probe_m = float(lookahead_curvature_probe_m)
        self.lookahead_curvature_min_scale = float(lookahead_curvature_min_scale)
        self.steer_gain = float(steer_gain)
        self.steer_rate_per_s = float(steer_rate_per_s)
        self.high_speed_steer_limit = float(high_speed_steer_limit)
        self.cross_track_gain = float(cross_track_gain)
        self.cross_track_softening_mps = float(cross_track_softening_mps)
        # Speed-scheduled cross-track: strong at low speed (stops pure pursuit from
        # chord-cutting slow-corner apexes over the inside kerb) and weak at high
        # speed (avoids the fast-esses steering oscillation). Falls back to a flat
        # gain when the high-speed value is unset.
        self.cross_track_highspeed_gain = (
            float(cross_track_gain) if cross_track_highspeed_gain is None
            else float(cross_track_highspeed_gain))
        self.cross_track_lowspeed_mps = float(cross_track_lowspeed_mps)
        self.cross_track_highspeed_mps = float(cross_track_highspeed_mps)
        if self.cross_track_highspeed_mps <= self.cross_track_lowspeed_mps:
            raise ValueError("cross-track highspeed mps must exceed lowspeed mps")
        self.recovery_enabled = bool(recovery_enabled)
        self.recovery_enter_deviation_m = float(recovery_enter_deviation_m)
        self.recovery_exit_deviation_m = float(recovery_exit_deviation_m)
        self.recovery_slip_angle_rad = float(recovery_slip_angle_rad)
        self.recovery_exit_slip_angle_rad = float(recovery_exit_slip_angle_rad)
        self.recovery_stable_s = float(recovery_stable_s)
        self.recovery_lookahead_scale = float(recovery_lookahead_scale)
        self.recovery_speed_scale = float(recovery_speed_scale)
        self.recovery_max_speed_mps = float(recovery_max_speed_mps)
        self.recovery_throttle_limit = float(recovery_throttle_limit)
        self.recovery_brake_gain = float(recovery_brake_gain)
        self.recovery_brake_limit = float(recovery_brake_limit)
        self.recovery_slip_brake_limit = float(recovery_slip_brake_limit)
        if min(self.wheelbase_m, self.lookahead_min_m,
               self.lookahead_max_m, self.steer_gain,
               self.steer_rate_per_s, self.high_speed_steer_limit) <= 0:
            raise ValueError("pure-pursuit geometry, gain, rate and limits must be positive")
        if self.lookahead_base_m < 0 or self.lookahead_speed_gain_s < 0:
            raise ValueError("pure-pursuit lookahead terms cannot be negative")
        if self.lookahead_min_m > self.lookahead_max_m:
            raise ValueError("pure-pursuit minimum lookahead exceeds maximum")
        if (self.lookahead_curvature_gain < 0
                or self.lookahead_curvature_probe_m <= 0
                or not 0 < self.lookahead_curvature_min_scale <= 1):
            raise ValueError(
                "curvature lookahead requires gain >= 0, probe > 0, and scale in (0, 1]")
        if self.high_speed_steer_limit > 1:
            raise ValueError("high-speed steer limit must be at most one")
        if self.cross_track_gain < 0 or self.cross_track_softening_mps <= 0:
            raise ValueError("cross-track gain must be nonnegative and softening positive")
        if (self.cross_track_highspeed_gain < 0
                or self.cross_track_lowspeed_mps < 0):
            raise ValueError("scheduled cross-track gains and speeds must be nonnegative")
        if not 0 < self.recovery_exit_deviation_m < self.recovery_enter_deviation_m:
            raise ValueError("recovery deviation thresholds must satisfy 0 < exit < enter")
        if not 0 <= self.recovery_exit_slip_angle_rad < self.recovery_slip_angle_rad:
            raise ValueError("recovery slip thresholds must satisfy 0 <= exit < enter")
        if min(self.recovery_stable_s, self.recovery_lookahead_scale,
               self.recovery_speed_scale, self.recovery_max_speed_mps,
               self.recovery_brake_gain) <= 0:
            raise ValueError("recovery timing, scales, speed and brake gain must be positive")
        if self.recovery_speed_scale > 1:
            raise ValueError("recovery speed scale must be at most one")
        if not 0 <= self.recovery_throttle_limit <= 1:
            raise ValueError("recovery throttle limit must be in [0, 1]")
        if not 0 < self.recovery_brake_limit <= 1:
            raise ValueError("recovery brake limit must be in (0, 1]")
        if not 0 <= self.recovery_slip_brake_limit <= self.recovery_brake_limit:
            raise ValueError("slip brake limit must be within recovery brake limit")
        self.previous_steer = 0.0
        self.recovery_active = False
        self.recovery_stable_time_s = 0.0
        self.slip_active = False
        self.recovery_trigger = "none"

    @staticmethod
    def _target_at_distance(waypoints_m: np.ndarray,
                            lookahead_m: float) -> tuple[np.ndarray, float]:
        points = np.asarray(waypoints_m, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 1:
            raise ValueError("predicted waypoints must have shape [N, 2]")
        if not np.all(np.isfinite(points)):
            raise ValueError("predicted waypoints contain non-finite values")
        path = np.vstack((np.zeros((1, 2), dtype=float), points))
        cumulative = np.concatenate(([0.0], np.cumsum(
            np.linalg.norm(np.diff(path, axis=0), axis=1))))
        distance = min(float(lookahead_m), float(cumulative[-1]))
        upper = int(np.searchsorted(cumulative, distance, side="right"))
        if upper >= len(path):
            return path[-1], distance
        lower = max(0, upper - 1)
        segment = cumulative[upper] - cumulative[lower]
        fraction = 0.0 if segment <= 1e-9 else (
            distance - cumulative[lower]) / segment
        return path[lower] + fraction * (path[upper] - path[lower]), distance

    def update(self, waypoints_m: np.ndarray, speed_mps: float, dt: float,
               steer_limit: float = 1.0,
               racing_line_local_m: np.ndarray | None = None,
               racing_line_deviation_m: float = 0.0,
               longitudinal_speed_mps: float | None = None,
               lateral_speed_mps: float = 0.0) -> tuple[float, dict]:
        speed = abs(float(speed_mps))
        dt = float(np.clip(dt, 1e-3, .1))
        longitudinal_speed = abs(float(
            speed if longitudinal_speed_mps is None else longitudinal_speed_mps))
        lateral_speed = abs(float(lateral_speed_mps))
        slip_angle = float(np.arctan2(lateral_speed, max(longitudinal_speed, 1.0)))
        deviation = abs(float(racing_line_deviation_m))
        self.slip_active = speed >= 10.0 and slip_angle >= self.recovery_slip_angle_rad
        deviation_triggered = speed >= 10.0 and deviation >= self.recovery_enter_deviation_m
        if self.recovery_enabled and not self.recovery_active:
            if self.slip_active or deviation_triggered:
                self.recovery_active = True
                self.recovery_stable_time_s = 0.0
                self.recovery_trigger = "slip" if self.slip_active else "deviation"
        elif self.recovery_active:
            stable = (deviation <= self.recovery_exit_deviation_m
                      and slip_angle <= self.recovery_exit_slip_angle_rad)
            self.recovery_stable_time_s = (
                self.recovery_stable_time_s + dt if stable else 0.0)
            if self.recovery_stable_time_s >= self.recovery_stable_s:
                self.recovery_active = False
                self.recovery_stable_time_s = 0.0
                self.recovery_trigger = "none"

        base_lookahead = self.lookahead_base_m + self.lookahead_speed_gain_s * speed
        curvature_scale = 1.0
        if self.lookahead_curvature_gain > 0:
            probe, _ = self._target_at_distance(
                waypoints_m, self.lookahead_curvature_probe_m)
            probe_r2 = float(probe[0] * probe[0] + probe[1] * probe[1])
            probe_curvature = (0.0 if probe_r2 <= 1e-9
                               else 2.0 * abs(float(probe[0])) / probe_r2)
            curvature_scale = max(
                self.lookahead_curvature_min_scale,
                1.0 / (1.0 + self.lookahead_curvature_gain * probe_curvature))
            base_lookahead *= curvature_scale
        desired_lookahead = float(np.clip(
            base_lookahead, self.lookahead_min_m, self.lookahead_max_m))
        source_points = waypoints_m
        source = "learned_waypoints"
        active_lookahead = desired_lookahead
        if self.recovery_active:
            if racing_line_local_m is None:
                raise ValueError("racing-line points are required during recovery")
            source_points = racing_line_local_m
            source = "racing_line_recovery"
            active_lookahead = desired_lookahead * self.recovery_lookahead_scale
        target, used_lookahead = self._target_at_distance(
            source_points, active_lookahead)
        x, y = float(target[0]), float(target[1])
        radius_squared = x * x + y * y
        curvature = 0.0 if radius_squared <= 1e-9 else 2.0 * x / radius_squared
        road_wheel_angle = float(np.arctan(self.wheelbase_m * curvature))
        pursuit_steer = -self.steer_gain * road_wheel_angle
        cross_track_x = 0.0
        cross_track_correction = 0.0
        if (not self.recovery_active and racing_line_local_m is not None
                and self.cross_track_gain > 0):
            line = np.asarray(racing_line_local_m, dtype=float)
            if line.ndim != 2 or line.shape[1] != 2 or len(line) < 1:
                raise ValueError("racing-line points must have shape [N, 2]")
            cross_track_x = float(line[0, 0])
            scheduled_gain = float(np.interp(
                speed, (self.cross_track_lowspeed_mps, self.cross_track_highspeed_mps),
                (self.cross_track_gain, self.cross_track_highspeed_gain)))
            cross_track_correction = float(
                -scheduled_gain * np.arctan2(
                    cross_track_x, speed + self.cross_track_softening_mps))
        raw_steer = pursuit_steer + cross_track_correction
        speed_limit = float(np.interp(speed, (0.0, 30.0, 80.0),
                                      (1.0, .75, self.high_speed_steer_limit)))
        active_limit = min(float(steer_limit), speed_limit)
        limited_steer = float(np.clip(raw_steer, -active_limit, active_limit))
        command = _approach(
            self.previous_steer, limited_steer, self.steer_rate_per_s * dt)
        self.previous_steer = float(command)
        return self.previous_steer, {
            "lateral_controller": "waypoint_pure_pursuit",
            "lateral_source": source,
            "recovery_active": self.recovery_active,
            "recovery_trigger": self.recovery_trigger,
            "recovery_stable_time_s": self.recovery_stable_time_s,
            "recovery_slip_angle_rad": slip_angle,
            "recovery_slip_active": self.slip_active,
            "recovery_lateral_speed_mps": lateral_speed,
            "lateral_desired_lookahead_m": desired_lookahead,
            "lateral_curvature_scale": curvature_scale,
            "lateral_used_lookahead_m": used_lookahead,
            "lateral_target_x_m": x,
            "lateral_target_y_m": y,
            "lateral_curvature_per_m": curvature,
            "lateral_pursuit_steer": pursuit_steer,
            "lateral_cross_track_x_m": cross_track_x,
            "lateral_cross_track_correction": cross_track_correction,
            "lateral_raw_steer": raw_steer,
            "lateral_active_steer_limit": active_limit,
            "lateral_limited_steer": limited_steer,
            "lateral_command_steer": self.previous_steer,
        }

    def apply_recovery_longitudinal(
            self, speed_mps: float, nominal_target_mps: float,
            throttle: float, brake: float,
            throttle_limit: float = 1.0,
            brake_limit: float = 1.0) -> tuple[float, float, dict]:
        if not self.recovery_active:
            return float(throttle), float(brake), {
                "recovery_speed_control_active": False,
                "recovery_speed_target_mps": None,
            }
        target = min(float(nominal_target_mps) * self.recovery_speed_scale,
                     self.recovery_max_speed_mps)
        overspeed = float(speed_mps) - target
        throttle_cap = 0.0 if self.slip_active else min(
            self.recovery_throttle_limit, float(throttle_limit))
        guarded_throttle = min(float(throttle), throttle_cap)
        brake_floor = float(np.clip(
            max(0.0, overspeed) * self.recovery_brake_gain,
            0.0, min(self.recovery_brake_limit, float(brake_limit))))
        if overspeed > 0:
            guarded_throttle = 0.0
        if self.slip_active:
            # Large combined braking and lateral slip can provoke another spin.
            # Lift power immediately, but cap braking until the car realigns.
            brake_floor = 0.0
            guarded_brake = min(
                float(brake), self.recovery_slip_brake_limit, float(brake_limit))
        else:
            guarded_brake = max(float(brake), brake_floor)
        return guarded_throttle, guarded_brake, {
            "recovery_speed_control_active": True,
            "recovery_speed_target_mps": target,
            "recovery_speed_overspeed_mps": overspeed,
            "recovery_throttle_cap": throttle_cap,
            "recovery_brake_floor": brake_floor,
        }

    def metadata(self) -> dict:
        return {
            "mode": "waypoint_pure_pursuit",
            "waypoint_source": "learned_checkpoint",
            "wheelbase_m": self.wheelbase_m,
            "lookahead_base_m": self.lookahead_base_m,
            "lookahead_speed_gain_s": self.lookahead_speed_gain_s,
            "lookahead_min_m": self.lookahead_min_m,
            "lookahead_max_m": self.lookahead_max_m,
            "lookahead_curvature_gain": self.lookahead_curvature_gain,
            "lookahead_curvature_probe_m": self.lookahead_curvature_probe_m,
            "lookahead_curvature_min_scale": self.lookahead_curvature_min_scale,
            "steer_gain": self.steer_gain,
            "steer_rate_per_s": self.steer_rate_per_s,
            "high_speed_steer_limit": self.high_speed_steer_limit,
            "cross_track_gain": self.cross_track_gain,
            "cross_track_softening_mps": self.cross_track_softening_mps,
            "cross_track_highspeed_gain": self.cross_track_highspeed_gain,
            "cross_track_lowspeed_mps": self.cross_track_lowspeed_mps,
            "cross_track_highspeed_mps": self.cross_track_highspeed_mps,
            "recovery_enabled": self.recovery_enabled,
            "recovery_enter_deviation_m": self.recovery_enter_deviation_m,
            "recovery_exit_deviation_m": self.recovery_exit_deviation_m,
            "recovery_slip_angle_rad": self.recovery_slip_angle_rad,
            "recovery_exit_slip_angle_rad": self.recovery_exit_slip_angle_rad,
            "recovery_stable_s": self.recovery_stable_s,
            "recovery_lookahead_scale": self.recovery_lookahead_scale,
            "recovery_speed_scale": self.recovery_speed_scale,
            "recovery_max_speed_mps": self.recovery_max_speed_mps,
            "recovery_throttle_limit": self.recovery_throttle_limit,
            "recovery_brake_gain": self.recovery_brake_gain,
            "recovery_brake_limit": self.recovery_brake_limit,
            "recovery_slip_brake_limit": self.recovery_slip_brake_limit,
        }


class FrameWriter:
    """Write diagnostic JPEGs off the control thread with bounded memory use."""

    def __init__(self, directory: Path, width: int = 960, max_pending: int = 4):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.pending: queue.Queue = queue.Queue(maxsize=max_pending)
        self.saved = 0
        self.dropped = 0
        self.errors: list[str] = []
        self.thread = threading.Thread(
            target=self._run, name="closed-loop-frame-writer", daemon=True)
        self.thread.start()

    def submit(self, index: int, elapsed_s: float, frame: np.ndarray) -> str | None:
        name = f"{index:06d}_{elapsed_s:07.3f}s.jpg"
        try:
            self.pending.put_nowait((name, frame.copy()))
        except queue.Full:
            self.dropped += 1
            return None
        return f"frames/{name}"

    def _run(self) -> None:
        from PIL import Image

        while True:
            item = self.pending.get()
            try:
                if item is None:
                    return
                name, frame = item
                image = Image.fromarray(frame)
                height = max(1, round(image.height * self.width / image.width))
                image = image.resize((self.width, height), Image.Resampling.BILINEAR)
                image.save(self.directory / name, format="JPEG", quality=80)
                self.saved += 1
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
            finally:
                self.pending.task_done()

    def close(self) -> None:
        self.pending.put(None)
        self.thread.join()


VK_CONTROL = 0x11
VK_SHIFT = 0x10
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_A = 0x41
VK_D = 0x44
VK_S = 0x53
VK_U = 0x55
VK_W = 0x57


def key_down(vk_code: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)


def handoff_key_down() -> bool:
    return key_down(VK_CONTROL) and key_down(VK_SHIFT) and key_down(VK_U)


def _approach(value: float, target: float, maximum_delta: float) -> float:
    return float(value + np.clip(target - value, -maximum_delta, maximum_delta))


class KeyboardController:
    """Smooth keyboard commands in AC steerNorm/pedal units."""

    def __init__(self, steer_rate: float, steer_return_rate: float, pedal_rate: float):
        self.steer_rate = float(steer_rate)
        self.steer_return_rate = float(steer_return_rate)
        self.pedal_rate = float(pedal_rate)
        self.steer = self.throttle = self.brake = 0.0

    def update(self, left: bool, right: bool, gas: bool, brake: bool,
               speed_mps: float, dt: float) -> tuple[float, float, float]:
        # At racing speed, keyboard steering needs a much smaller authority.
        speed_limit = float(np.interp(abs(speed_mps), (0.0, 30.0, 80.0), (1.0, .45, .22)))
        steer_target = (float(right) - float(left)) * speed_limit
        steer_rate = self.steer_return_rate if steer_target == 0 else self.steer_rate
        self.steer = _approach(self.steer, steer_target, steer_rate * dt)
        brake_target = float(brake)
        throttle_target = 0.0 if brake else float(gas)
        self.throttle = _approach(self.throttle, throttle_target, self.pedal_rate * dt)
        self.brake = _approach(self.brake, brake_target, self.pedal_rate * dt)
        return self.steer, self.throttle, self.brake


def steering_action(desired_steer: float, steer_actuator_gain: float,
                    steer_map: SteerMap | None = None) -> float:
    return (steer_map.action_for_normalized(desired_steer)
            if steer_map is not None else
            float(desired_steer) * float(steer_actuator_gain))


def keyboard_manual_drive(client: ControlClient, telemetry, args, log_path: Path,
                          steer_map: SteerMap | None = None) -> dict:
    """Drive vJoy from the keyboard until the handoff chord's rising edge."""
    controller = KeyboardController(
        args.keyboard_steer_rate, args.keyboard_steer_return_rate,
        args.keyboard_pedal_rate)
    started = last_tick = next_tick = time.perf_counter()
    period = 1.0 / args.hz
    handoff_was_down = handoff_key_down()
    count = 0
    print("[MANUAL] keyboard -> vJoy; arrows/WASD drive, Ctrl+Shift+U hands off, F8 stops")
    with log_path.open("a", encoding="utf-8", buffering=1) as stream:
        while time.perf_counter() - started < args.manual_timeout:
            now = time.perf_counter()
            handoff_now = handoff_key_down()
            if handoff_now and not handoff_was_down:
                state = telemetry.latest_state()
                if state is None:
                    raise RuntimeError("telemetry unavailable at keyboard handoff")
                x, y, yaw, speed, spline = _required_telemetry(state)
                print(f"[HANDOFF] Ctrl+Shift+U at spline={spline:.5f} "
                      f"speed={abs(speed)*3.6:.1f}km/h")
                return {"frames": count, "spline": spline, "speed_mps": abs(speed),
                        "position_x": x, "position_y": y, "yaw_rad": yaw}
            handoff_was_down = handoff_now
            state = telemetry.latest_state()
            if state is None:
                raise RuntimeError("telemetry unavailable during keyboard driving")
            x, y, yaw, speed, spline = _required_telemetry(state)
            dt = min(max(now - last_tick, 0.0), 0.1)
            last_tick = now
            left = key_down(VK_LEFT) or key_down(VK_A)
            right = key_down(VK_RIGHT) or key_down(VK_D)
            gas = key_down(VK_UP) or key_down(VK_W)
            brake_key = key_down(VK_DOWN) or key_down(VK_S)
            steer, throttle, brake = controller.update(
                left, right, gas, brake_key, speed, dt)
            status = client.command(
                steering_action(steer, args.steer_actuator_gain, steer_map),
                throttle, brake)
            if not status.get("ok") or status.get("emergency"):
                raise RuntimeError(f"vJoy bridge stopped keyboard control: {status}")
            stream.write(json.dumps({
                "index": count, "elapsed_s": now - started,
                "position_x": x, "position_y": y, "yaw_rad": yaw,
                "speed_mps": abs(speed), "spline_position": spline,
                "manual_steer": steer, "manual_throttle": throttle,
                "manual_brake": brake,
            }) + "\n")
            count += 1
            if count == 1 or count % max(1, round(args.hz)) == 0:
                print(f"[manual {count:05d}] {abs(speed)*3.6:6.1f}km/h spline={spline:.4f} "
                      f"steer={steer:+.3f} gas={throttle:.3f} brake={brake:.3f}")
            next_tick += period
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -period:
                next_tick = time.perf_counter()
    raise RuntimeError(f"keyboard handoff timed out after {args.manual_timeout:g}s")


def limited_controls(prediction: dict, steer_sign: float, steer_limit: float,
                     steer_actuator_gain: float,
                     throttle_limit: float, brake_limit: float,
                     steer_map: SteerMap | None = None) -> tuple[float, float, float]:
    desired_steer = float(np.clip(
        steer_sign * prediction["pred_steer"], -steer_limit, steer_limit))
    steer = steering_action(desired_steer, steer_actuator_gain, steer_map)
    throttle = float(np.clip(prediction["pred_throttle"], 0.0, throttle_limit))
    brake = float(np.clip(prediction["pred_brake"], 0.0, brake_limit))
    return steer, throttle, brake


def run(args) -> Path:
    from PIL import Image
    from ac_dataset.recorder import ACTelemetrySource
    from ac_dataset.recorder.screen_capture import ScreenCapture

    racing_line_path = args.racing_line or discover_racing_line(args.prepared)
    target_provider = RacingLineTarget.from_prepared(args.prepared, racing_line_path)
    predictor = ShadowPredictor(args.checkpoint, args.device, args.prepared)
    steer_map = SteerMap.from_csv(args.steer_map) if args.steer_map else None
    longitudinal_controller = None
    if args.longitudinal_controller == "profile-pid":
        longitudinal_profile = ExpertSpeedProfile.from_prepared(
            args.longitudinal_profile_prepared or args.prepared,
            args.longitudinal_profile_bins,
            args.longitudinal_profile_quantile,
        )
        longitudinal_controller = PreviewPIDLongitudinalController(
            longitudinal_profile, target_provider,
            preview_decel_mps2=args.longitudinal_preview_decel_mps2,
            kp=args.longitudinal_kp, ki=args.longitudinal_ki,
            kd=args.longitudinal_kd,
            deadband_mps=args.longitudinal_deadband_mps,
            integral_limit=args.longitudinal_integral_limit,
            target_scale=args.longitudinal_target_scale,
        )
    lateral_controller = None
    if args.lateral_controller == "waypoint-pure-pursuit":
        lateral_controller = WaypointPurePursuitController(
            wheelbase_m=args.lateral_wheelbase_m,
            lookahead_base_m=args.lateral_lookahead_base_m,
            lookahead_speed_gain_s=args.lateral_lookahead_speed_gain_s,
            lookahead_min_m=args.lateral_lookahead_min_m,
            lookahead_max_m=args.lateral_lookahead_max_m,
            lookahead_curvature_gain=args.lateral_lookahead_curvature_gain,
            lookahead_curvature_probe_m=args.lateral_lookahead_curvature_probe_m,
            lookahead_curvature_min_scale=args.lateral_lookahead_curvature_min_scale,
            steer_gain=args.lateral_steer_gain,
            steer_rate_per_s=args.lateral_steer_rate,
            high_speed_steer_limit=args.lateral_high_speed_steer_limit,
            cross_track_gain=args.lateral_cross_track_gain,
            cross_track_softening_mps=args.lateral_cross_track_softening_mps,
            cross_track_highspeed_gain=args.lateral_cross_track_highspeed_gain,
            cross_track_lowspeed_mps=args.lateral_cross_track_lowspeed_mps,
            cross_track_highspeed_mps=args.lateral_cross_track_highspeed_mps,
            recovery_enabled=args.lateral_recovery,
            recovery_enter_deviation_m=args.lateral_recovery_enter_deviation_m,
            recovery_exit_deviation_m=args.lateral_recovery_exit_deviation_m,
            recovery_slip_angle_rad=np.deg2rad(args.lateral_recovery_slip_angle_deg),
            recovery_exit_slip_angle_rad=np.deg2rad(
                args.lateral_recovery_exit_slip_angle_deg),
            recovery_stable_s=args.lateral_recovery_stable_s,
            recovery_lookahead_scale=args.lateral_recovery_lookahead_scale,
            recovery_speed_scale=args.lateral_recovery_speed_scale,
            recovery_max_speed_mps=args.lateral_recovery_max_speed_mps,
            recovery_throttle_limit=args.lateral_recovery_throttle_limit,
            recovery_brake_gain=args.lateral_recovery_brake_gain,
            recovery_brake_limit=args.lateral_recovery_brake_limit,
            recovery_slip_brake_limit=args.lateral_recovery_slip_brake_limit,
        )
    speed_guard = None
    if args.speed_profile_guard:
        speed_profile = ExpertSpeedProfile.from_prepared(
            args.prepared, args.speed_profile_bins, args.speed_profile_quantile)
        speed_guard = SpeedProfileGuard(
            speed_profile,
            args.speed_profile_margin_mps,
            args.speed_profile_throttle_cut_range_mps,
            args.speed_profile_full_brake_range_mps,
        )
    run_dir = args.output / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    if speed_guard is not None:
        speed_guard.profile.write_csv(run_dir / "speed_profile.csv")
    config = {
        "mode": "guarded_closed_loop", "actuation_enabled": True,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": predictor.epoch,
        "racing_line": str(racing_line_path.resolve()), "capture_region": list(args.region),
        "hz": args.hz, "duration_s": args.duration, "device": str(predictor.device),
        "preprocess_backend": predictor.preprocess_backend, "steer_sign": args.steer_sign,
        "lap_phase_enabled": predictor.lap_phase_enabled,
        "lap_phase_policy": {
            "model_command": "telemetry_LapCount_training_phase",
            "speed_profile": "flying_latched_after_launch",
            "standing_exit_speed_mps": args.standing_exit_speed_mps,
        },
        "steer_actuator_gain": args.steer_actuator_gain,
        "steer_mapping": (steer_map.metadata() if steer_map is not None else {
            "mode": "linear_gain", "gain": args.steer_actuator_gain}),
        "racing_line_lookaheads_m": list(RACING_LINE_LOOKAHEADS_M),
        "limits": {"steer": args.steer_limit, "throttle": args.throttle_limit,
                   "brake": args.brake_limit, "max_speed_mps": args.max_speed_mps,
                   "max_racing_line_deviation_m": args.max_racing_line_deviation_m},
        "stop_on_track_out": args.stop_on_track_out, "bridge_port": args.bridge_port,
        "speed_profile_guard": (speed_guard.metadata() if speed_guard is not None
                                else {"enabled": False}),
        "longitudinal_control": (
            {
                **longitudinal_controller.metadata(),
                "profile_prepared": str(
                    (args.longitudinal_profile_prepared or args.prepared).resolve()),
            } if longitudinal_controller is not None
            else {"mode": "model"}),
        "lateral_control": (lateral_controller.metadata()
                            if lateral_controller is not None
                            else {"mode": "model"}),
        "slip_throttle_control": {
            "enabled": True,
            "minimum_active_speed_mps": 5.0,
            "cut_angle_deg": args.slip_throttle_cut_deg,
            "zero_angle_deg": args.slip_throttle_zero_deg,
            "policy": "linear_throttle_ramp",
        },
        "diagnostics": {
            "save_frames_hz": args.save_frames_hz,
            "saved_frame_width": args.saved_frame_width,
            "realtime_lite": args.realtime_lite,
            "stall_arm_speed_mps": args.stall_arm_speed_mps,
            "stall_speed_mps": args.stall_speed_mps,
            "stall_duration_s": args.stall_duration,
        },
        "keyboard_handoff": {
            "enabled": args.keyboard_handoff,
            "manual_timeout_s": args.manual_timeout,
            "key": "Ctrl+Shift+U",
        },
        "normalization": predictor.normalization_log,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    telemetry = capture = client = frame_writer = None
    handoff_frame_to_save = None
    armed = False
    count = deadline_misses = speed_guard_frames = 0
    recovery_frames = slip_frames = 0
    max_slip_angle_rad = 0.0
    max_speed_profile_overspeed_mps = 0.0
    frame_interval = max(
        1, round(args.hz / args.save_frames_hz)) if args.save_frames_hz > 0 else 0
    stall_detector = StallDetector(
        args.stall_arm_speed_mps, args.stall_speed_mps, args.stall_duration)
    started = time.perf_counter()
    control_ended = None
    stop_reason = "completed"
    log_path = run_dir / "closed_loop.jsonl"
    try:
        telemetry = ACTelemetrySource(args.host, args.telemetry_port).start()
        if not telemetry.wait_first(args.telemetry_timeout):
            raise RuntimeError("connected but no telemetry state arrived")
        capture = ScreenCapture(args.region, args.backend, args.monitor, args.window_title)
        if args.save_frames_hz > 0:
            frame_writer = FrameWriter(run_dir / "frames", args.saved_frame_width)
        warm_frame = capture.grab()
        if warm_frame is None:
            raise RuntimeError("screen capture returned no frame")
        if tuple(warm_frame.shape) != TRAINING_FRAME_SHAPE:
            raise RuntimeError(f"capture shape {tuple(warm_frame.shape)} != {TRAINING_FRAME_SHAPE}")
        client = ControlClient(port=args.bridge_port)
        print("[CONTROL] guarded closed-loop mode; F8 = latched emergency brake")
        print(f"[ready] epoch={predictor.epoch} device={predictor.device} capture={capture.backend}")
        if args.realtime_lite:
            print("[realtime] lite logging enabled; model inputs and controls are unchanged")
        print(f"[limits] steer=+/-{args.steer_limit:g} throttle<={args.throttle_limit:g} "
              f"brake<={args.brake_limit:g} max_speed={args.max_speed_mps:g}m/s")
        if speed_guard is not None:
            print(
                f"[speed guard] expert q={args.speed_profile_quantile:g} "
                f"margin={args.speed_profile_margin_mps:g}m/s "
                f"throttle_cut={args.speed_profile_throttle_cut_range_mps:g}m/s "
                f"full_brake={args.speed_profile_full_brake_range_mps:g}m/s"
            )
        if longitudinal_controller is not None:
            print(
                f"[longitudinal] profile-pid preview_decel="
                f"{args.longitudinal_preview_decel_mps2:g}m/s^2 "
                f"kp={args.longitudinal_kp:g} ki={args.longitudinal_ki:g} "
                f"kd={args.longitudinal_kd:g} "
                f"target_scale={args.longitudinal_target_scale:g}; model pedals ignored"
            )
        if lateral_controller is not None:
            print(
                f"[lateral] waypoint-pure-pursuit wheelbase="
                f"{args.lateral_wheelbase_m:g}m lookahead="
                f"{args.lateral_lookahead_base_m:g}+"
                f"{args.lateral_lookahead_speed_gain_s:g}*speed "
                f"steer_gain={args.lateral_steer_gain:g} "
                f"cross_track_gain={args.lateral_cross_track_gain:g}; "
                f"model direct steer ignored"
            )
            if args.lateral_recovery:
                print(
                    f"[recovery] enter_dev={args.lateral_recovery_enter_deviation_m:g}m "
                    f"exit_dev={args.lateral_recovery_exit_deviation_m:g}m "
                    f"slip={args.lateral_recovery_slip_angle_deg:g}deg "
                    f"speed_cap={args.lateral_recovery_max_speed_mps*3.6:g}km/h"
                )
        print(f"[delay] arming in {args.arm_delay:g}s; switch to AC NOW")
        time.sleep(args.arm_delay)
        warmup_frame = capture.grab()
        if warmup_frame is None:
            raise RuntimeError("pre-arm screen capture returned no frame")
        if tuple(warmup_frame.shape) != TRAINING_FRAME_SHAPE:
            raise RuntimeError(
                f"pre-arm capture shape {tuple(warmup_frame.shape)} != {TRAINING_FRAME_SHAPE}")
        warmup_state = telemetry.latest_state()
        if warmup_state is None:
            raise RuntimeError("pre-arm telemetry state became unavailable")
        warmup_x, warmup_y, warmup_yaw, warmup_speed, warmup_spline = _required_telemetry(
            warmup_state)
        _, warmup_line = target_provider.local_points(
            warmup_spline, (warmup_x, warmup_y), warmup_yaw)
        warmup_standing = standing_lap_from_state(warmup_state)
        _, warmup_target_speed = target_provider.target_speed_features(
            warmup_spline, warmup_speed, "standing" if warmup_standing else "flying")
        warmup_ego_raw, warmup_ego_legacy = live_ego_features(warmup_state)
        warmup_ego = predictor.normalize_ego(warmup_ego_raw, warmup_ego_legacy)
        predictor.predict(
            warmup_frame, warmup_ego, warmup_line,
            standing_lap=warmup_standing,
            target_speed=warmup_target_speed if predictor.target_speed_width else None)
        print("[warmup] first model inference completed before arming")
        pre_arm_frame = capture.grab()
        if pre_arm_frame is None:
            raise RuntimeError("post-warmup screen capture returned no frame")
        initial_frame_name = "manual_start_frame.jpg" if args.keyboard_handoff else "pre_arm_frame.jpg"
        Image.fromarray(pre_arm_frame).save(run_dir / initial_frame_name, quality=90)
        # winsound.Beep is synchronous. Play it before arming so its 180 ms
        # duration cannot trip the bridge's 150 ms command watchdog.
        audio_cue(args.audio_cues, "armed")
        # The beep and pre-arm JPEG leave a short idle gap after the ordinary
        # warmup. Re-warm the exact lite actuation path immediately before arm
        # so the first controlled frame does not pay a 50-60 ms cold spike.
        if args.realtime_lite:
            predictor.predict(
                warmup_frame, warmup_ego, warmup_line,
                standing_lap=warmup_standing,
                target_speed=(warmup_target_speed
                              if predictor.target_speed_width else None),
                full_diagnostics=False)
        status = client.arm()
        if not status.get("ok") or not status.get("armed"):
            raise RuntimeError(f"vJoy bridge refused arm: {status}")
        armed = True
        if args.keyboard_handoff:
            handoff = keyboard_manual_drive(
                client, telemetry, args, run_dir / "manual.jsonl", steer_map)
            handoff_frame = capture.grab()
            handoff_state = telemetry.latest_state()
            if handoff_frame is None or handoff_state is None:
                raise RuntimeError("frame or telemetry unavailable at keyboard handoff")
            hx, hy, hyaw, hspeed, hspline = _required_telemetry(handoff_state)
            _, handoff_line = target_provider.local_points(
                hspline, (hx, hy), hyaw)
            handoff_standing = standing_lap_from_state(handoff_state)
            _, handoff_target_speed = target_provider.target_speed_features(
                hspline, hspeed, "standing" if handoff_standing else "flying")
            handoff_ego_raw, handoff_ego_legacy = live_ego_features(handoff_state)
            handoff_ego = predictor.normalize_ego(handoff_ego_raw, handoff_ego_legacy)
            predictor.predict(
                handoff_frame, handoff_ego, handoff_line,
                standing_lap=handoff_standing,
                target_speed=handoff_target_speed if predictor.target_speed_width else None)
            # Keep disk I/O out of the armed watchdog window; save after disarm.
            handoff_frame_to_save = handoff_frame.copy()
            (run_dir / "handoff.json").write_text(
                json.dumps(handoff, indent=2), encoding="utf-8")
            print("[HANDOFF READY] keyboard input released; model control starts now")
        print("[ARMED] model controls are now being sent; press F8 to stop")
        started = time.perf_counter()
        next_tick = started
        previous_control_tick = started
        period = 1.0 / args.hz
        # AC reports LapCount<1 for the whole first lap, so the standing (launch)
        # profile stays selected until the line is crossed. That profile dips to
        # ~launch speed near the grid spline, which brakes the car hard when it
        # re-approaches the start/finish line to complete lap 1. Latch to flying
        # once the launch is clearly done.
        launch_complete = False
        # Per-line flushing and diagnostic GPU copies consume meaningful time
        # in a 25 ms control budget. Normal shutdown still flushes this buffer.
        with log_path.open("a", encoding="utf-8", buffering=64 * 1024) as stream:
            while time.perf_counter() - started < args.duration:
                tick_started_ns = time.perf_counter_ns()
                tick_started = tick_started_ns / 1e9
                state, telemetry_received_ns, telemetry_count = (
                    telemetry.latest_state_with_meta())
                capture_started_ns = time.perf_counter_ns()
                frame = capture.grab()
                capture_finished_ns = time.perf_counter_ns()
                if state is None or frame is None:
                    raise RuntimeError("live frame or telemetry became unavailable")
                x, y, yaw, speed, spline = _required_telemetry(state)
                speed = abs(speed)
                if args.max_speed_mps > 0 and speed > args.max_speed_mps:
                    stop_reason = f"max_speed:{speed:.3f}"
                    client.emergency(stop_reason)
                    armed = False
                    break
                if args.stop_on_track_out and bool(state.get("out_of_track", False)):
                    stop_reason = "out_of_track"
                    client.emergency(stop_reason)
                    armed = False
                    break
                if stall_detector.update(speed, tick_started):
                    stop_reason = f"stalled:{speed:.3f}"
                    client.emergency(stop_reason)
                    armed = False
                    break
                indexed_racing_line_deviation_m = target_provider.indexed_deviation_m(
                    spline, (x, y))
                control_line_local_m, line_projection_distance_m, (
                    racing_line_deviation_m) = target_provider.control_local_points(
                        (x, y), yaw)
                if (args.max_racing_line_deviation_m > 0
                        and racing_line_deviation_m > args.max_racing_line_deviation_m):
                    stop_reason = f"racing_line_deviation:{racing_line_deviation_m:.3f}"
                    client.emergency(stop_reason)
                    armed = False
                    break
                line_local_m, line_normalized = target_provider.local_points(
                    spline, (x, y), yaw)
                ego_raw, ego_normalized = live_ego_features(state)
                telemetry_standing_lap = standing_lap_from_state(state)
                if not launch_complete and speed >= args.standing_exit_speed_mps:
                    launch_complete = True
                    print(f"[lap] launch complete at {speed*3.6:.0f}km/h; "
                          "flying speed profile latched")
                model_standing_lap, speed_profile_standing_lap = control_lap_phases(
                    telemetry_standing_lap, launch_complete)
                target_speed_mps, target_speed_features = target_provider.target_speed_features(
                    spline, speed,
                    "standing" if speed_profile_standing_lap else "flying")
                pred = predictor.predict(
                    frame, predictor.normalize_ego(ego_raw, ego_normalized), line_normalized,
                    standing_lap=model_standing_lap,
                    target_speed=target_speed_features if predictor.target_speed_width else None,
                    full_diagnostics=not args.realtime_lite,
                    actuation_waypoints=lateral_controller is not None)
                prediction_finished_ns = time.perf_counter_ns()
                steer, throttle, brake = limited_controls(
                    pred, args.steer_sign, args.steer_limit,
                    args.steer_actuator_gain,
                    args.throttle_limit, args.brake_limit,
                    steer_map,
                )
                control_dt = tick_started - previous_control_tick
                lateral_state = {}
                if lateral_controller is not None:
                    desired_steer, lateral_state = lateral_controller.update(
                        np.asarray(pred["pred_waypoints_m"], dtype=float),
                        speed, control_dt, args.steer_limit, control_line_local_m,
                        racing_line_deviation_m,
                        longitudinal_speed_mps=float(ego_raw[1]),
                        lateral_speed_mps=float(ego_raw[2]))
                    recovery_frames += int(lateral_state["recovery_active"])
                    slip_frames += int(lateral_state["recovery_slip_active"])
                    max_slip_angle_rad = max(
                        max_slip_angle_rad,
                        lateral_state["recovery_slip_angle_rad"])
                    desired_steer = float(np.clip(
                        args.steer_sign * desired_steer,
                        -args.steer_limit, args.steer_limit))
                    steer = steering_action(
                        desired_steer, args.steer_actuator_gain, steer_map)
                longitudinal_state = {}
                if longitudinal_controller is not None:
                    throttle, brake, longitudinal_state = longitudinal_controller.update(
                        speed, spline, speed_profile_standing_lap, control_dt,
                        args.throttle_limit, args.brake_limit)
                previous_control_tick = tick_started
                limited_throttle, limited_brake = throttle, brake
                guard_state = {}
                if speed_guard is not None:
                    throttle, brake, guard_state = speed_guard.apply(
                        speed, spline, speed_profile_standing_lap, throttle, brake,
                        args.brake_limit)
                    speed_guard_frames += int(guard_state["speed_guard_active"])
                    max_speed_profile_overspeed_mps = max(
                        max_speed_profile_overspeed_mps,
                        guard_state["speed_profile_overspeed_mps"],
                    )
                recovery_speed_state = {}
                if lateral_controller is not None and lateral_controller.recovery_active:
                    nominal_target = float(
                        longitudinal_state.get("longitudinal_target_mps",
                                               target_speed_mps[0]))
                    throttle, brake, recovery_speed_state = (
                        lateral_controller.apply_recovery_longitudinal(
                            speed, nominal_target, throttle, brake,
                            args.throttle_limit, args.brake_limit))
                    if longitudinal_controller is not None:
                        longitudinal_controller.reset_integral()
                # Slip-aware throttle cut: the speed PID has no grip awareness, so
                # during a slide the car is below target and the PID floors the
                # throttle, which sustains the power-on slide instead of recovering.
                # Ease power off as the slip angle grows past normal cornering.
                slip_cut_state = {}
                if lateral_state and speed >= 5.0:
                    slip_deg = float(np.degrees(lateral_state["recovery_slip_angle_rad"]))
                    throttle, slip_cut_state = slip_limited_throttle(
                        throttle, slip_deg, args.slip_throttle_cut_deg,
                        args.slip_throttle_zero_deg)
                    if slip_cut_state and longitudinal_controller is not None:
                        longitudinal_controller.reset_integral()
                status = client.command(steer, throttle, brake)
                command_returned_ns = time.perf_counter_ns()
                if not status.get("ok") or status.get("emergency"):
                    stop_reason = f"bridge:{status.get('reason')}"
                    armed = False
                    print(f"[stop] vJoy bridge stopped control: {status}")
                    break
                actual = observed_controls(state)
                saved_frame = None
                if frame_writer is not None and count % frame_interval == 0:
                    saved_frame = frame_writer.submit(
                        count, time.perf_counter() - started, frame)
                loop_ms = (time.perf_counter() - tick_started) * 1000.0
                missed = loop_ms > period * 1000.0
                deadline_misses += int(missed)
                row = {
                    "index": count, "elapsed_s": time.perf_counter() - started,
                    "position_x": x, "position_y": y, "yaw_rad": yaw,
                    "speed_mps": speed, "spline_position": spline,
                    "racing_line_deviation_m": racing_line_deviation_m,
                    "spline_indexed_racing_line_deviation_m": (
                        indexed_racing_line_deviation_m),
                    "racing_line_projection_distance_m": line_projection_distance_m,
                    "lap_phase": ("standing" if model_standing_lap else "flying"),
                    "model_lap_phase": (
                        "standing" if model_standing_lap else "flying"),
                    "speed_profile_lap_phase": (
                        "standing" if speed_profile_standing_lap else "flying"),
                    "launch_complete": launch_complete,
                    **actual, **pred,
                    "limited_throttle_before_speed_guard": limited_throttle,
                    "limited_brake_before_speed_guard": limited_brake,
                    **lateral_state,
                    **longitudinal_state,
                    **guard_state,
                    **recovery_speed_state,
                    **slip_cut_state,
                    "sent_steer": steer, "sent_throttle": throttle, "sent_brake": brake,
                    "timing": {
                        "tick_started_perf_ns": tick_started_ns,
                        "telemetry_received_perf_ns": telemetry_received_ns,
                        "telemetry_count": telemetry_count,
                        "capture_started_perf_ns": capture_started_ns,
                        "capture_finished_perf_ns": capture_finished_ns,
                        "prediction_finished_perf_ns": prediction_finished_ns,
                        "client_sent_perf_ns": status.get("client_sent_perf_ns"),
                        "bridge_received_perf_ns": status.get("bridge_received_perf_ns"),
                        "bridge_applied_perf_ns": status.get("bridge_applied_perf_ns"),
                        "client_received_perf_ns": status.get("client_received_perf_ns"),
                        "command_returned_perf_ns": command_returned_ns,
                    },
                    "telemetry_age_at_capture_ms": (
                        (capture_started_ns - telemetry_received_ns) / 1e6
                        if telemetry_received_ns is not None else None),
                    "capture_call_ms": (capture_finished_ns - capture_started_ns) / 1e6,
                    "capture_to_command_send_ms": (
                        (status["client_sent_perf_ns"] - capture_started_ns) / 1e6
                        if status.get("client_sent_perf_ns") is not None else None),
                    "capture_to_bridge_apply_ms": (
                        (status["bridge_applied_perf_ns"] - capture_started_ns) / 1e6
                        if status.get("bridge_applied_perf_ns") is not None else None),
                    "telemetry_receive_to_bridge_apply_ms": (
                        (status["bridge_applied_perf_ns"] - telemetry_received_ns) / 1e6
                        if (status.get("bridge_applied_perf_ns") is not None
                            and telemetry_received_ns is not None) else None),
                    "out_of_track": bool(state.get("out_of_track", False)),
                    "saved_frame": saved_frame,
                    "loop_ms": loop_ms, "deadline_missed": missed,
                }
                if not args.realtime_lite:
                    row.update({
                        "ego_raw": ego_raw.tolist(),
                        "ego_normalized": ego_normalized.tolist(),
                        "racing_line_local_m": line_local_m.tolist(),
                        "racing_line_normalized": line_normalized.tolist(),
                        "target_speed_profile_mps": target_speed_mps.tolist(),
                        "target_speed_features": target_speed_features.tolist(),
                    })
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
                if count == 1 or count % max(1, round(args.hz)) == 0:
                    stream.flush()
                    guard_text = ""
                    if speed_guard is not None:
                        guard_text = (
                            f" target={guard_state['speed_profile_target_mps']*3.6:5.1f}km/h"
                            f" guard={'ON' if guard_state['speed_guard_active'] else 'off'}"
                        )
                    if longitudinal_state:
                        guard_text += (
                            f" pid_target={longitudinal_state['longitudinal_target_mps']*3.6:5.1f}km/h"
                            f" preview={longitudinal_state['longitudinal_limiting_lookahead_m']:g}m"
                        )
                    if slip_cut_state:
                        guard_text += (
                            f" SLIPCUT:{slip_cut_state['slip_cut_angle_deg']:.0f}deg"
                            f" gas{slip_cut_state['slip_throttle_before']:.2f}->{slip_cut_state['slip_throttle_after']:.2f}"
                        )
                    if lateral_state:
                        guard_text += (
                            f" pp={lateral_state['lateral_command_steer']:+.2f}"
                            f" lookahead={lateral_state['lateral_used_lookahead_m']:.1f}m"
                        )
                        if lateral_state["recovery_active"]:
                            guard_text += (
                                f" RECOVERY:{lateral_state['recovery_trigger']}"
                                f" slip={np.degrees(lateral_state['recovery_slip_angle_rad']):.1f}deg"
                            )
                    pipeline_ms = row["capture_to_bridge_apply_ms"]
                    pipeline_text = (f" pipeline={pipeline_ms:.1f}ms"
                                     if pipeline_ms is not None else " pipeline=n/a")
                    print(f"[{count:05d}] {speed*3.6:6.1f}km/h {loop_ms:4.1f}ms "
                          f"steer={steer:+.3f} gas={throttle:.3f} brake={brake:.3f}"
                          f"{pipeline_text}"
                          f"{guard_text}")
                next_tick += period
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -period and not args.realtime_lite:
                    next_tick = time.perf_counter()
    except KeyboardInterrupt:
        stop_reason = "ctrl_c"
        print("[stop] Ctrl+C")
    finally:
        control_ended = time.perf_counter()
        if client is not None:
            if armed:
                client.disarm()
            client.close()
        if capture is not None:
            capture.close()
        if telemetry is not None:
            telemetry.close()
        if frame_writer is not None:
            frame_writer.close()
        if handoff_frame_to_save is not None:
            Image.fromarray(handoff_frame_to_save).save(
                run_dir / "pre_arm_frame.jpg", quality=90)
        if count > 0:
            audio_cue(args.audio_cues, "completed" if stop_reason == "completed" else "stopped")

    elapsed = control_ended - started
    summary = {
        "mode": "guarded_closed_loop", "frames": count, "elapsed_s": elapsed,
        "effective_hz": count / elapsed if elapsed > 0 else 0.0,
        "deadline_misses": deadline_misses, "stop_reason": stop_reason,
        "actuation_was_enabled": True,
        "speed_guard_frames": speed_guard_frames,
        "max_speed_profile_overspeed_mps": max_speed_profile_overspeed_mps,
        "recovery_frames": recovery_frames,
        "slip_frames": slip_frames,
        "max_slip_angle_deg": float(np.degrees(max_slip_angle_rad)),
        "saved_frames": frame_writer.saved if frame_writer is not None else 0,
        "dropped_frames": frame_writer.dropped if frame_writer is not None else 0,
        "frame_write_errors": frame_writer.errors if frame_writer is not None else [],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Guarded TCP-AC vJoy closed-loop test")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("training/tcp_lagunaseca/checkpoints/best.pt"))
    parser.add_argument("--prepared", type=Path, default=Path("training/tcp_lagunaseca"))
    parser.add_argument("--racing-line", type=Path)
    parser.add_argument("--output", type=Path, default=Path("training/tcp_lagunaseca/closed_loop"))
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--hz", type=float, default=40.0)
    parser.add_argument("--arm-delay", type=float, default=5.0)
    parser.add_argument("--steer-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--steer-limit", type=float, default=0.20)
    parser.add_argument("--steer-actuator-gain", type=float, default=0.40,
                        help="vJoy steering command per desired AC steerNorm (calibrated 2026-08-05)")
    parser.add_argument("--steer-map", type=Path,
                        default=Path("configs/ferrari_sf15t_steer_map.csv"),
                        help="CSV mapping physical steerAngle degrees to vJoy action_steerAngle")
    parser.add_argument("--confirm-steer-map-calibrated", action="store_true",
                        help="required with --steer-map; confirms this exact AC/vJoy setup was axis-tested")
    parser.add_argument(
        "--lateral-controller", choices=("model", "waypoint-pure-pursuit"),
        default="model",
        help=("model sends checkpoint steering directly; waypoint-pure-pursuit "
              "uses the checkpoint's predicted waypoints and classical steering"))
    parser.add_argument("--lateral-wheelbase-m", type=float, default=3.5)
    parser.add_argument("--lateral-lookahead-base-m", type=float, default=8.0)
    parser.add_argument("--lateral-lookahead-speed-gain-s", type=float, default=.35)
    parser.add_argument("--lateral-lookahead-min-m", type=float, default=10.0)
    parser.add_argument("--lateral-lookahead-max-m", type=float, default=35.0)
    parser.add_argument(
        "--lateral-lookahead-curvature-gain", type=float, default=0.0,
        help="shorten lookahead on tight upcoming path (per 1/m of probed curvature); "
             "stops pure pursuit from chord-cutting slow apexes. 0 disables.")
    parser.add_argument("--lateral-lookahead-curvature-probe-m", type=float, default=15.0)
    parser.add_argument("--lateral-lookahead-curvature-min-scale", type=float, default=0.5)
    parser.add_argument("--lateral-steer-gain", type=float, default=4.5)
    parser.add_argument("--lateral-steer-rate", type=float, default=2.5)
    parser.add_argument("--lateral-high-speed-steer-limit", type=float, default=.45)
    parser.add_argument(
        "--lateral-cross-track-gain", type=float, default=0.0,
        help="Stanley-style correction toward the supplied racing line; 0 disables")
    parser.add_argument("--lateral-cross-track-softening-mps", type=float, default=5.0)
    parser.add_argument(
        "--lateral-cross-track-highspeed-gain", type=float, default=None,
        help="cross-track gain at/above --lateral-cross-track-highspeed-mps; the "
             "gain ramps from --lateral-cross-track-gain (low speed, fixes slow-corner "
             "apex cuts) down to this (high speed, avoids fast-section oscillation). "
             "Unset = flat gain.")
    parser.add_argument("--lateral-cross-track-lowspeed-mps", type=float, default=28.0)
    parser.add_argument("--lateral-cross-track-highspeed-mps", type=float, default=42.0)
    parser.add_argument(
        "--lateral-recovery", action=argparse.BooleanOptionalAction, default=False,
        help="use racing-line steering and conservative pedals after slip/deviation")
    parser.add_argument("--lateral-recovery-enter-deviation-m", type=float, default=3.0)
    parser.add_argument("--lateral-recovery-exit-deviation-m", type=float, default=1.5)
    parser.add_argument("--lateral-recovery-slip-angle-deg", type=float, default=7.0)
    parser.add_argument("--lateral-recovery-exit-slip-angle-deg", type=float, default=3.5)
    parser.add_argument("--lateral-recovery-stable-s", type=float, default=.75)
    parser.add_argument("--lateral-recovery-lookahead-scale", type=float, default=1.25)
    parser.add_argument("--lateral-recovery-speed-scale", type=float, default=.90)
    parser.add_argument("--lateral-recovery-max-speed-mps", type=float, default=45.0)
    parser.add_argument("--lateral-recovery-throttle-limit", type=float, default=.20)
    parser.add_argument("--lateral-recovery-brake-gain", type=float, default=.04)
    parser.add_argument("--lateral-recovery-brake-limit", type=float, default=.25)
    parser.add_argument("--lateral-recovery-slip-brake-limit", type=float, default=.15)
    parser.add_argument("--throttle-limit", type=float, default=0.30)
    parser.add_argument("--brake-limit", type=float, default=1.0)
    parser.add_argument("--max-speed-mps", type=float, default=20.0)
    parser.add_argument(
        "--longitudinal-controller", choices=("model", "profile-pid"), default="model",
        help=("model uses checkpoint throttle/brake; profile-pid uses only model steering "
              "and controls pedals from an expert speed profile"))
    parser.add_argument(
        "--longitudinal-profile-prepared", type=Path,
        help="prepared dataset used to build the profile-pid speed reference; defaults to --prepared")
    parser.add_argument("--longitudinal-profile-bins", type=int, default=500)
    parser.add_argument("--longitudinal-profile-quantile", type=float, default=.50)
    parser.add_argument("--longitudinal-preview-decel-mps2", type=float, default=14.0)
    parser.add_argument("--longitudinal-kp", type=float, default=.12)
    parser.add_argument("--longitudinal-ki", type=float, default=.015)
    parser.add_argument("--longitudinal-kd", type=float, default=0.0)
    parser.add_argument("--longitudinal-deadband-mps", type=float, default=.25)
    parser.add_argument("--longitudinal-integral-limit", type=float, default=10.0)
    parser.add_argument(
        "--longitudinal-target-scale", type=float, default=1.0,
        help="multiply the complete expert speed profile; e.g. 0.90 adds corner margin")
    parser.add_argument(
        "--slip-throttle-cut-deg", type=float, default=12.0,
        help="slip angle where throttle starts being eased off (normal cornering "
             "for this car reaches ~10 deg, so keep this above that)")
    parser.add_argument(
        "--slip-throttle-zero-deg", type=float, default=18.0,
        help="slip angle at/above which throttle is fully cut; power ramps down "
             "linearly between --slip-throttle-cut-deg and this value")
    parser.add_argument(
        "--standing-exit-speed-mps", type=float, default=20.0,
        help="once the car first exceeds this speed the launch is treated as complete "
             "and the flying speed profile is latched on; prevents the standing "
             "launch-zone target from braking the car when it re-approaches the "
             "start/finish line at the end of the first lap")
    parser.add_argument("--speed-profile-guard", action="store_true",
                        help="limit power and add braking above the expert speed envelope")
    parser.add_argument("--speed-profile-bins", type=int, default=500)
    parser.add_argument("--speed-profile-quantile", type=float, default=.75)
    parser.add_argument("--speed-profile-margin-mps", type=float, default=1.5)
    parser.add_argument("--speed-profile-throttle-cut-range-mps", type=float, default=3.0)
    parser.add_argument("--speed-profile-full-brake-range-mps", type=float, default=8.0)
    parser.add_argument("--max-racing-line-deviation-m", type=float, default=15.0,
                        help="emergency stop beyond this reference-line distance; 0 disables")
    parser.add_argument("--stop-on-track-out", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--telemetry-port", type=int, default=2345)
    parser.add_argument("--telemetry-timeout", type=float, default=5.0)
    parser.add_argument("--bridge-port", type=int, default=2350)
    parser.add_argument("--backend", default="dxcam", choices=("auto", "dxcam", "mss", "wgc"))
    parser.add_argument("--region", type=parse_region, default=parse_region("0,80,1920,945"))
    parser.add_argument("--monitor", type=int, default=0)
    parser.add_argument("--window-title")
    parser.add_argument("--save-frames-hz", type=float, default=0.0,
                        help="save diagnostic live frames at this rate; 0 disables")
    parser.add_argument(
        "--realtime-lite", action=argparse.BooleanOptionalAction, default=False,
        help=("skip diagnostic-only GPU copies and bulky per-frame arrays to "
              "maximize the live control rate"))
    parser.add_argument("--saved-frame-width", type=int, default=960)
    parser.add_argument("--stall-arm-speed-mps", type=float, default=10.0)
    parser.add_argument("--stall-speed-mps", type=float, default=1.0)
    parser.add_argument("--stall-duration", type=float, default=1.0)
    parser.add_argument("--keyboard-handoff", action="store_true",
                        help="drive vJoy with arrows/WASD until Ctrl+Shift+U hands off")
    parser.add_argument("--manual-timeout", type=float, default=300.0)
    parser.add_argument("--keyboard-steer-rate", type=float, default=2.0)
    parser.add_argument("--keyboard-steer-return-rate", type=float, default=3.5)
    parser.add_argument("--keyboard-pedal-rate", type=float, default=4.0)
    parser.add_argument("--audio-cues", action="store_true",
                        help="beep once when armed and twice after a normal completion")
    parser.add_argument("--confirm-vjoy-mapped", action="store_true", required=True,
                        help="required acknowledgement that X/Y/Z are mapped in AC")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.hz <= 0 or args.arm_delay < 0:
        parser.error("hz must be positive and arm delay nonnegative")
    if args.max_racing_line_deviation_m < 0:
        parser.error("--max-racing-line-deviation-m cannot be negative")
    if args.speed_profile_bins < 20:
        parser.error("--speed-profile-bins must be at least 20")
    if args.longitudinal_profile_bins < 20:
        parser.error("--longitudinal-profile-bins must be at least 20")
    if not 0 < args.speed_profile_quantile < 1:
        parser.error("--speed-profile-quantile must be in (0,1)")
    if not 0 < args.longitudinal_profile_quantile < 1:
        parser.error("--longitudinal-profile-quantile must be in (0,1)")
    if args.longitudinal_preview_decel_mps2 <= 0:
        parser.error("--longitudinal-preview-decel-mps2 must be positive")
    if min(args.longitudinal_kp, args.longitudinal_ki, args.longitudinal_kd,
           args.longitudinal_deadband_mps) < 0:
        parser.error("longitudinal PID gains and deadband cannot be negative")
    if args.longitudinal_integral_limit <= 0:
        parser.error("--longitudinal-integral-limit must be positive")
    if not 0 < args.longitudinal_target_scale <= 1:
        parser.error("--longitudinal-target-scale must be in (0,1]")
    if min(args.lateral_wheelbase_m, args.lateral_lookahead_min_m,
           args.lateral_lookahead_max_m, args.lateral_steer_gain,
           args.lateral_steer_rate, args.lateral_high_speed_steer_limit) <= 0:
        parser.error("lateral geometry, gain, rate and limits must be positive")
    if min(args.lateral_lookahead_base_m,
           args.lateral_lookahead_speed_gain_s) < 0:
        parser.error("lateral lookahead terms cannot be negative")
    if args.lateral_lookahead_min_m > args.lateral_lookahead_max_m:
        parser.error("lateral minimum lookahead cannot exceed maximum")
    if (args.lateral_lookahead_curvature_gain < 0
            or args.lateral_lookahead_curvature_probe_m <= 0
            or not 0 < args.lateral_lookahead_curvature_min_scale <= 1):
        parser.error(
            "curvature lookahead requires gain >= 0, probe > 0, and min scale in (0,1]")
    if args.lateral_high_speed_steer_limit > 1:
        parser.error("--lateral-high-speed-steer-limit must be at most 1")
    if (args.lateral_cross_track_gain < 0
            or args.lateral_cross_track_softening_mps <= 0):
        parser.error("lateral cross-track gain must be nonnegative and softening positive")
    if (args.lateral_cross_track_highspeed_gain is not None
            and args.lateral_cross_track_highspeed_gain < 0):
        parser.error("--lateral-cross-track-highspeed-gain cannot be negative")
    if (args.lateral_cross_track_lowspeed_mps < 0
            or args.lateral_cross_track_highspeed_mps
            <= args.lateral_cross_track_lowspeed_mps):
        parser.error("cross-track speed schedule requires 0 <= low speed < high speed")
    if (args.slip_throttle_cut_deg < 0
            or args.slip_throttle_zero_deg <= args.slip_throttle_cut_deg):
        parser.error("slip throttle angles must satisfy 0 <= cut < zero")
    if args.standing_exit_speed_mps <= 0:
        parser.error("--standing-exit-speed-mps must be positive")
    if not (0 < args.lateral_recovery_exit_deviation_m
            < args.lateral_recovery_enter_deviation_m):
        parser.error("recovery deviation thresholds must satisfy 0 < exit < enter")
    if not (0 <= args.lateral_recovery_exit_slip_angle_deg
            < args.lateral_recovery_slip_angle_deg):
        parser.error("recovery slip thresholds must satisfy 0 <= exit < enter")
    if min(args.lateral_recovery_stable_s,
           args.lateral_recovery_lookahead_scale,
           args.lateral_recovery_speed_scale,
           args.lateral_recovery_max_speed_mps,
           args.lateral_recovery_brake_gain) <= 0:
        parser.error("recovery timing, scales, speed and brake gain must be positive")
    if args.lateral_recovery_speed_scale > 1:
        parser.error("--lateral-recovery-speed-scale must be at most 1")
    if not 0 <= args.lateral_recovery_throttle_limit <= 1:
        parser.error("--lateral-recovery-throttle-limit must be in [0,1]")
    if not 0 < args.lateral_recovery_brake_limit <= 1:
        parser.error("--lateral-recovery-brake-limit must be in (0,1]")
    if not (0 <= args.lateral_recovery_slip_brake_limit
            <= args.lateral_recovery_brake_limit):
        parser.error("slip brake limit must be within recovery brake limit")
    if args.speed_profile_margin_mps < 0:
        parser.error("--speed-profile-margin-mps cannot be negative")
    if min(args.speed_profile_throttle_cut_range_mps,
           args.speed_profile_full_brake_range_mps) <= 0:
        parser.error("--speed-profile control ranges must be positive")
    if args.save_frames_hz < 0 or args.save_frames_hz > args.hz:
        parser.error("--save-frames-hz must be in [0, hz]")
    if args.saved_frame_width <= 0:
        parser.error("--saved-frame-width must be positive")
    if min(args.stall_arm_speed_mps, args.stall_speed_mps, args.stall_duration) <= 0:
        parser.error("stall detection values must be positive")
    if min(args.manual_timeout, args.keyboard_steer_rate,
           args.keyboard_steer_return_rate, args.keyboard_pedal_rate) <= 0:
        parser.error("keyboard handoff timing/rate values must be positive")
    if args.steer_actuator_gain <= 0:
        parser.error("--steer-actuator-gain must be positive")
    if args.steer_map and not args.confirm_steer_map_calibrated:
        parser.error("--steer-map requires --confirm-steer-map-calibrated")
    for name in ("steer_limit", "throttle_limit", "brake_limit"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0,1]")
    run(args)


if __name__ == "__main__":
    main()
