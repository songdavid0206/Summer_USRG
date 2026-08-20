"""Measure local vJoy bridge and AC control-observation latency while stationary."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from .client import ControlClient
from .steer_map import SteerMap


def summarize(rows: list[dict]) -> dict:
    def stats(key: str) -> dict:
        values = np.asarray([row[key] for row in rows], dtype=float)
        return {
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.quantile(values, .95)),
            "min_ms": float(values.min()),
            "max_ms": float(values.max()),
        }

    return {
        "transitions": len(rows),
        "client_to_bridge_receive": stats("client_to_bridge_receive_ms"),
        "bridge_update_vjd": stats("bridge_update_vjd_ms"),
        "client_roundtrip": stats("client_roundtrip_ms"),
        "bridge_apply_to_ac_telemetry": stats("bridge_apply_to_ac_telemetry_ms"),
        "client_send_to_ac_telemetry": stats("client_send_to_ac_telemetry_ms"),
    }


def _speed_mps(state: dict) -> float:
    for key in ("speed", "speed_mps"):
        try:
            value = abs(float(state[key]))
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    raise RuntimeError("telemetry has no finite speed/speed_mps")


def _actual_steer(state: dict) -> float:
    try:
        value = float(state["steerNorm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("telemetry has no usable steerNorm") from exc
    if not np.isfinite(value):
        raise RuntimeError("telemetry steerNorm is non-finite")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stationary steering-step latency measurement for vJoy -> AC telemetry")
    parser.add_argument("--steer-map", type=Path,
                        default=Path("configs/ferrari_sf15t_steer_map.csv"))
    parser.add_argument("--amplitude", type=float, default=.25,
                        help="AC normalized steering step magnitude")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=.04)
    parser.add_argument("--transition-timeout", type=float, default=.5)
    parser.add_argument("--hold", type=float, default=.2)
    parser.add_argument("--command-hz", type=float, default=100.0)
    parser.add_argument("--max-speed-mps", type=float, default=.5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--telemetry-port", type=int, default=2345)
    parser.add_argument("--bridge-port", type=int, default=2350)
    parser.add_argument("--output", type=Path, default=Path("training/latency_probe"))
    parser.add_argument("--confirm-car-stationary", action="store_true", required=True)
    parser.add_argument("--confirm-vjoy-mapped", action="store_true", required=True)
    args = parser.parse_args()
    if not 0 < args.amplitude <= .5:
        parser.error("--amplitude must be in (0, 0.5]")
    if args.cycles < 1 or args.command_hz < 20:
        parser.error("--cycles must be positive and --command-hz at least 20")
    if min(args.tolerance, args.transition_timeout, args.hold, args.max_speed_mps) <= 0:
        parser.error("tolerance, timing, and max-speed values must be positive")

    from ac_dataset.recorder import ACTelemetrySource

    mapping = SteerMap.from_csv(args.steer_map)
    telemetry = ACTelemetrySource(args.host, args.telemetry_port).start()
    client = ControlClient(port=args.bridge_port)
    armed = False
    rows: list[dict] = []
    run_dir = args.output / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        if not telemetry.wait_first(5.0):
            raise RuntimeError("no AC telemetry")
        state, _, _ = telemetry.latest_state_with_meta()
        if state is None or _speed_mps(state) > args.max_speed_mps:
            raise RuntimeError("car must be stationary before latency probe")
        status = client.arm()
        if not status.get("ok") or not status.get("armed"):
            raise RuntimeError(f"bridge refused arm: {status}")
        armed = True
        print("[SAFE] brake=1 throughout; steering-only steps; F8 = emergency")
        print(f"[probe] amplitude=+/-{args.amplitude:g} cycles={args.cycles} "
              f"command_hz={args.command_hz:g}")

        targets = [value for _ in range(args.cycles)
                   for value in (args.amplitude, 0.0, -args.amplitude, 0.0)]
        period = 1.0 / args.command_hz
        # Establish a stationary, braked neutral baseline.
        baseline_end = time.perf_counter() + .5
        while time.perf_counter() < baseline_end:
            client.command(mapping.action_for_normalized(0.0), 0.0, 1.0)
            time.sleep(period)

        for index, target in enumerate(targets):
            action = mapping.action_for_normalized(target)
            first = client.command(action, 0.0, 1.0)
            bridge_received_ns = first.get("bridge_received_perf_ns")
            bridge_applied_ns = first.get("bridge_applied_perf_ns")
            if bridge_received_ns is None or bridge_applied_ns is None:
                raise RuntimeError("restart the updated vjoy_bridge.py; timing fields are missing")
            deadline = time.perf_counter() + args.transition_timeout
            last_keepalive = time.perf_counter()
            observed = None
            while time.perf_counter() < deadline:
                state, telemetry_received_ns, telemetry_count = telemetry.latest_state_with_meta()
                if state is None or telemetry_received_ns is None:
                    continue
                speed = _speed_mps(state)
                if speed > args.max_speed_mps:
                    client.emergency(f"latency_probe_speed:{speed:.3f}")
                    armed = False
                    raise RuntimeError(f"car moved during probe: {speed:.3f}m/s")
                actual = _actual_steer(state)
                if (telemetry_received_ns >= bridge_applied_ns
                        and abs(actual - target) <= args.tolerance):
                    observed = (telemetry_received_ns, telemetry_count, actual)
                    break
                now = time.perf_counter()
                if now - last_keepalive >= period:
                    client.command(action, 0.0, 1.0)
                    last_keepalive = now
                time.sleep(.001)
            if observed is None:
                raise RuntimeError(
                    f"steer target {target:+.3f} not observed within "
                    f"{args.transition_timeout:g}s")
            telemetry_received_ns, telemetry_count, actual = observed
            row = {
                "index": index, "target_steer_norm": target,
                "vjoy_action": action, "observed_steer_norm": actual,
                "telemetry_count": telemetry_count,
                "client_sent_perf_ns": first["client_sent_perf_ns"],
                "bridge_received_perf_ns": bridge_received_ns,
                "bridge_applied_perf_ns": bridge_applied_ns,
                "telemetry_received_perf_ns": telemetry_received_ns,
                "client_to_bridge_receive_ms":
                    (bridge_received_ns - first["client_sent_perf_ns"]) / 1e6,
                "bridge_update_vjd_ms":
                    (bridge_applied_ns - bridge_received_ns) / 1e6,
                "client_roundtrip_ms": first["client_roundtrip_ms"],
                "bridge_apply_to_ac_telemetry_ms":
                    (telemetry_received_ns - bridge_applied_ns) / 1e6,
                "client_send_to_ac_telemetry_ms":
                    (telemetry_received_ns - first["client_sent_perf_ns"]) / 1e6,
            }
            rows.append(row)
            print(f"[{index+1:02d}/{len(targets):02d}] target={target:+.2f} "
                  f"actual={actual:+.3f} bridge={row['bridge_update_vjd_ms']:.3f}ms "
                  f"AC={row['bridge_apply_to_ac_telemetry_ms']:.2f}ms "
                  f"end_to_end={row['client_send_to_ac_telemetry_ms']:.2f}ms")
            hold_end = time.perf_counter() + args.hold
            while time.perf_counter() < hold_end:
                client.command(action, 0.0, 1.0)
                time.sleep(period)
    except KeyboardInterrupt:
        print("[stop] Ctrl+C")
    finally:
        if armed:
            client.disarm()
        client.close()
        telemetry.close()

    if not rows:
        raise RuntimeError("latency probe produced no completed transitions")
    report = {"config": vars(args), "rows": rows, "summary": summarize(rows)}
    # Path objects are only present in config.
    report["config"] = {key: str(value) if isinstance(value, Path) else value
                        for key, value in report["config"].items()}
    path = run_dir / "latency.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
