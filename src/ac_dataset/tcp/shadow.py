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
    FLYING_LAP_COMMAND_INDEX, POSITION_SCALE_M, SPEED_SCALE_MPS,
    STANDING_LAP_COMMAND_INDEX, normalization_metadata,
)
from .data import preprocess_rgb, to_local


TRAINING_FRAME_SHAPE = (945, 1920, 3)


def parse_region(value: str) -> tuple[int, int, int, int]:
    fields = tuple(int(item.strip()) for item in value.split(","))
    if len(fields) != 4 or fields[2] <= 0 or fields[3] <= 0:
        raise argparse.ArgumentTypeError("region must be left,top,width,height")
    return fields


def decode_signed_accel(value: float) -> tuple[float, float]:
    """Convert TCP's one longitudinal output into non-overlapping pedals."""
    value = float(np.clip(value, -1.0, 1.0))
    return max(value, 0.0), max(-value, 0.0)


def observed_controls(state: dict) -> dict[str, float | str | None]:
    throttle = _finite(state.get("accStatus"))
    brake = _finite(state.get("brakeStatus"))
    steer = _finite(state.get("steerNorm"))
    source = "steerNorm"
    if steer is None:
        candidate = _finite(state.get("steerAngle"))
        if candidate is not None and abs(candidate) <= 1.05:
            steer, source = candidate, "steerAngle"
        else:
            source = "unavailable"
    signed = None if throttle is None or brake is None else throttle - brake
    return {
        "actual_steer": steer, "actual_steer_source": source,
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
    def __init__(self, table: pd.DataFrame, lookahead_m: float = 50.0):
        required = {"normalized_position", "x", "y", "dist"}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"racing line is missing columns: {sorted(missing)}")
        self.norm = table["normalized_position"].to_numpy(float)
        self.dist = table["dist"].to_numpy(float)
        self.xy = table[["x", "y"]].to_numpy(float)
        self.length = float(self.dist[-1] + np.linalg.norm(self.xy[0] - self.xy[-1]))
        self.lookahead_m = float(lookahead_m)

    @classmethod
    def from_csv(cls, path: Path, lookahead_m: float = 50.0):
        return cls(pd.read_csv(path), lookahead_m)

    def target_world(self, spline_position: float) -> np.ndarray:
        spline = float(spline_position) % 1.0
        current_dist = float(np.interp(spline, self.norm, self.dist))
        target_dist = (current_dist + self.lookahead_m) % self.length
        index = min(int(np.searchsorted(self.dist, target_dist)), len(self.xy) - 1)
        return self.xy[index].astype(np.float32)

    def target_local(self, spline_position: float, origin_xy, yaw: float) -> tuple[np.ndarray, np.ndarray]:
        world = self.target_world(spline_position)
        local_m = to_local(world[None, :], np.asarray(origin_xy, np.float32), float(yaw))[0]
        return local_m, local_m / POSITION_SCALE_M


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
    def __init__(self, checkpoint_path: Path, device_name: str = "auto"):
        import torch
        from .model import build_model

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        actual = checkpoint.get("normalization")
        expected = normalization_metadata()
        if actual != expected:
            raise RuntimeError(f"checkpoint normalization mismatch: {actual} != {expected}")
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.model = build_model(False).to(self.device)
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
        self.val_loss = float(checkpoint["val_loss"])
        self.lap_phase_enabled = checkpoint.get("lap_phase") is not None
        # Pay CUDA/cuDNN's one-time setup cost before the first logged frame.
        dummy_image = torch.zeros((1, 3, 256, 512), dtype=torch.float32, device=self.device)
        dummy_speed = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
        dummy_target = torch.zeros((1, 2), dtype=torch.float32, device=self.device)
        dummy_command = torch.zeros((1, 6), dtype=torch.float32, device=self.device)
        dummy_command[0, 3] = 1.0
        with torch.inference_mode():
            self.model(dummy_image, dummy_speed, dummy_target, dummy_command)
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

    def predict(self, rgb: np.ndarray, speed_mps: float, target_point: np.ndarray,
                standing_lap: bool | None = None) -> dict:
        import torch

        preprocess_started = time.perf_counter()
        image = self._preprocess(rgb)
        speed = torch.tensor([[speed_mps]], dtype=torch.float32, device=self.device)
        target = torch.as_tensor(target_point, dtype=torch.float32, device=self.device).reshape(1, 2)
        command = torch.zeros((1, 6), dtype=torch.float32, device=self.device)
        command[0, 3] = 1.0
        if self.lap_phase_enabled:
            if standing_lap is None:
                raise ValueError("phase-aware checkpoint requires standing_lap")
            command[0, STANDING_LAP_COMMAND_INDEX if standing_lap
                    else FLYING_LAP_COMMAND_INDEX] = 1.0
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
        started = time.perf_counter()
        with torch.inference_mode():
            pred = self.model(image, speed, target, command)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - started) * 1000.0
        action = pred["action"][0].detach().cpu().numpy()
        throttle, brake = decode_signed_accel(float(action[0]))
        return {
            "pred_signed_accel": float(action[0]), "pred_steer": float(action[1]),
            "pred_throttle": throttle, "pred_brake": brake,
            "pred_speed_mps": float(pred["speed"][0, 0].detach().cpu()) * SPEED_SCALE_MPS,
            "pred_waypoints_m": (pred["waypoints"][0].detach().cpu().numpy() * POSITION_SCALE_M).tolist(),
            "pred_future_actions": pred["future_actions"][0].detach().cpu().numpy().tolist(),
            "preprocess_ms": preprocess_ms, "inference_ms": inference_ms,
        }


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
    target_provider = RacingLineTarget.from_csv(racing_line_path, args.lookahead_m)
    predictor = ShadowPredictor(args.checkpoint, args.device)
    run_dir = args.output / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "mode": "shadow_read_only", "actuation_enabled": False,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": predictor.epoch,
        "checkpoint_val_loss": predictor.val_loss, "prepared": str(args.prepared.resolve()),
        "racing_line": str(racing_line_path.resolve()), "lookahead_m": args.lookahead_m,
        "capture_region": list(args.region), "capture_backend": args.backend,
        "window_title": args.window_title, "hz": args.hz, "device": str(predictor.device),
        "preprocess_backend": predictor.preprocess_backend,
        "lap_phase_enabled": predictor.lap_phase_enabled,
        "start_delay_s": args.start_delay,
        "wait_for_motion": args.wait_for_motion,
        "motion_threshold_mps": args.motion_threshold_mps,
        "motion_hold_s": args.motion_hold,
        "save_frame_every": args.save_frame_every,
        "normalization": normalization_metadata(),
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
                    local_m, target_norm = target_provider.target_local(spline, (x, y), yaw)
                    signature = int(frame[::32, ::32].astype(np.uint64).sum())
                    is_repeated = signature == last_signature
                    repeated += int(is_repeated)
                    last_signature = signature
                    standing_lap = standing_lap_from_state(state)
                    pred = predictor.predict(
                        frame, abs(speed), target_norm, standing_lap=standing_lap)
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
                        "target_local_m": local_m.tolist(), "target_normalized": target_norm.tolist(),
                        "frame_shape": list(frame.shape), "frame_repeated": is_repeated,
                        "saved_frame": None if saved_frame is None else str(saved_frame.resolve()),
                        "capture_ms": capture_ms,
                        **actual, **pred,
                    }
                    if actual["actual_steer"] is not None:
                        row["steer_error"] = pred["pred_steer"] - actual["actual_steer"]
                    if actual["actual_signed_accel"] is not None:
                        row["signed_accel_error"] = pred["pred_signed_accel"] - actual["actual_signed_accel"]
                    row["loop_ms"] = (time.perf_counter() - tick_started) * 1000.0
                    row["deadline_missed"] = row["loop_ms"] > period * 1000.0
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
                    if count == 1 or count % max(1, round(args.hz)) == 0:
                        print(f"[{count:06d}] {row['loop_ms']:.1f}ms steer={pred['pred_steer']:+.3f} "
                              f"accel={pred['pred_signed_accel']:+.3f} "
                              f"=> gas={pred['pred_throttle']:.3f} brake={pred['pred_brake']:.3f}")
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
    parser.add_argument("--lookahead-m", type=float, default=50.0)
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
