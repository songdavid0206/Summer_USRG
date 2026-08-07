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

from ac_dataset.control import ControlClient
from .config import normalization_metadata
from .shadow import (
    TRAINING_FRAME_SHAPE, RacingLineTarget, ShadowPredictor, _required_telemetry,
    discover_racing_line, observed_controls, parse_region, standing_lap_from_state,
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


def keyboard_manual_drive(client: ControlClient, telemetry, args, log_path: Path) -> dict:
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
                steer * args.steer_actuator_gain, throttle, brake)
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
                     throttle_limit: float, brake_limit: float) -> tuple[float, float, float]:
    desired_steer = float(np.clip(
        steer_sign * prediction["pred_steer"], -steer_limit, steer_limit))
    steer = desired_steer * steer_actuator_gain
    throttle = float(np.clip(prediction["pred_throttle"], 0.0, throttle_limit))
    brake = float(np.clip(prediction["pred_brake"], 0.0, brake_limit))
    return steer, throttle, brake


def run(args) -> Path:
    from PIL import Image
    from ac_dataset.recorder import ACTelemetrySource
    from ac_dataset.recorder.screen_capture import ScreenCapture

    racing_line_path = args.racing_line or discover_racing_line(args.prepared)
    target_provider = RacingLineTarget.from_csv(racing_line_path, args.lookahead_m)
    predictor = ShadowPredictor(args.checkpoint, args.device)
    run_dir = args.output / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "mode": "guarded_closed_loop", "actuation_enabled": True,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": predictor.epoch,
        "racing_line": str(racing_line_path.resolve()), "capture_region": list(args.region),
        "hz": args.hz, "duration_s": args.duration, "device": str(predictor.device),
        "preprocess_backend": predictor.preprocess_backend, "steer_sign": args.steer_sign,
        "lap_phase_enabled": predictor.lap_phase_enabled,
        "steer_actuator_gain": args.steer_actuator_gain,
        "limits": {"steer": args.steer_limit, "throttle": args.throttle_limit,
                   "brake": args.brake_limit, "max_speed_mps": args.max_speed_mps},
        "stop_on_track_out": args.stop_on_track_out, "bridge_port": args.bridge_port,
        "diagnostics": {
            "save_frames_hz": args.save_frames_hz,
            "saved_frame_width": args.saved_frame_width,
            "stall_arm_speed_mps": args.stall_arm_speed_mps,
            "stall_speed_mps": args.stall_speed_mps,
            "stall_duration_s": args.stall_duration,
        },
        "keyboard_handoff": {
            "enabled": args.keyboard_handoff,
            "manual_timeout_s": args.manual_timeout,
            "key": "Ctrl+Shift+U",
        },
        "normalization": normalization_metadata(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    telemetry = capture = client = frame_writer = None
    handoff_frame_to_save = None
    armed = False
    count = deadline_misses = 0
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
        print(f"[limits] steer=+/-{args.steer_limit:g} throttle<={args.throttle_limit:g} "
              f"brake<={args.brake_limit:g} max_speed={args.max_speed_mps:g}m/s")
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
        _, warmup_target = target_provider.target_local(
            warmup_spline, (warmup_x, warmup_y), warmup_yaw)
        predictor.predict(
            warmup_frame, abs(warmup_speed), warmup_target,
            standing_lap=standing_lap_from_state(warmup_state))
        print("[warmup] first model inference completed before arming")
        pre_arm_frame = capture.grab()
        if pre_arm_frame is None:
            raise RuntimeError("post-warmup screen capture returned no frame")
        initial_frame_name = "manual_start_frame.jpg" if args.keyboard_handoff else "pre_arm_frame.jpg"
        Image.fromarray(pre_arm_frame).save(run_dir / initial_frame_name, quality=90)
        # winsound.Beep is synchronous. Play it before arming so its 180 ms
        # duration cannot trip the bridge's 150 ms command watchdog.
        audio_cue(args.audio_cues, "armed")
        status = client.arm()
        if not status.get("ok") or not status.get("armed"):
            raise RuntimeError(f"vJoy bridge refused arm: {status}")
        armed = True
        if args.keyboard_handoff:
            handoff = keyboard_manual_drive(
                client, telemetry, args, run_dir / "manual.jsonl")
            handoff_frame = capture.grab()
            handoff_state = telemetry.latest_state()
            if handoff_frame is None or handoff_state is None:
                raise RuntimeError("frame or telemetry unavailable at keyboard handoff")
            hx, hy, hyaw, hspeed, hspline = _required_telemetry(handoff_state)
            _, handoff_target = target_provider.target_local(hspline, (hx, hy), hyaw)
            predictor.predict(
                handoff_frame, abs(hspeed), handoff_target,
                standing_lap=standing_lap_from_state(handoff_state))
            # Keep disk I/O out of the armed watchdog window; save after disarm.
            handoff_frame_to_save = handoff_frame.copy()
            (run_dir / "handoff.json").write_text(
                json.dumps(handoff, indent=2), encoding="utf-8")
            print("[HANDOFF READY] keyboard input released; model control starts now")
        print("[ARMED] model controls are now being sent; press F8 to stop")
        started = time.perf_counter()
        next_tick = started
        period = 1.0 / args.hz
        with log_path.open("a", encoding="utf-8", buffering=1) as stream:
            while time.perf_counter() - started < args.duration:
                tick_started = time.perf_counter()
                state = telemetry.latest_state()
                frame = capture.grab()
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
                local_m, target_norm = target_provider.target_local(spline, (x, y), yaw)
                standing_lap = standing_lap_from_state(state)
                pred = predictor.predict(
                    frame, speed, target_norm, standing_lap=standing_lap)
                steer, throttle, brake = limited_controls(
                    pred, args.steer_sign, args.steer_limit,
                    args.steer_actuator_gain,
                    args.throttle_limit, args.brake_limit,
                )
                status = client.command(steer, throttle, brake)
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
                    "lap_phase": "standing" if standing_lap else "flying",
                    "target_local_m": local_m.tolist(), **actual, **pred,
                    "sent_steer": steer, "sent_throttle": throttle, "sent_brake": brake,
                    "out_of_track": bool(state.get("out_of_track", False)),
                    "saved_frame": saved_frame,
                    "loop_ms": loop_ms, "deadline_missed": missed,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
                if count == 1 or count % max(1, round(args.hz)) == 0:
                    print(f"[{count:05d}] {speed*3.6:6.1f}km/h {loop_ms:4.1f}ms "
                          f"steer={steer:+.3f} gas={throttle:.3f} brake={brake:.3f}")
                next_tick += period
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -period:
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
    parser.add_argument("--throttle-limit", type=float, default=0.30)
    parser.add_argument("--brake-limit", type=float, default=1.0)
    parser.add_argument("--max-speed-mps", type=float, default=20.0)
    parser.add_argument("--stop-on-track-out", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lookahead-m", type=float, default=50.0)
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
    if args.duration <= 0 or args.duration > 60:
        parser.error("initial guarded test duration must be in (0, 60] seconds")
    if args.hz <= 0 or args.arm_delay < 0:
        parser.error("hz must be positive and arm delay nonnegative")
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
    for name in ("steer_limit", "throttle_limit", "brake_limit"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0,1]")
    run(args)


if __name__ == "__main__":
    main()
