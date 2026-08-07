"""Send a small fixed command through the bridge to verify AC axis mapping."""

from __future__ import annotations

import argparse
import time

from .client import ControlClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed vJoy axis/sign test; start bridge first")
    parser.add_argument("--steer", type=float, default=0.0)
    parser.add_argument("--throttle", type=float, default=0.0)
    parser.add_argument("--brake", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--port", type=int, default=2350)
    parser.add_argument("--verify-telemetry", action="store_true",
                        help="also print live AC steer/throttle/brake before and during the test")
    parser.add_argument("--confirm-axis-test", action="store_true", required=True)
    args = parser.parse_args()
    telemetry = None
    if args.verify_telemetry:
        from ac_dataset.recorder import ACTelemetrySource
        from ac_dataset.tcp.shadow import observed_controls
        telemetry = ACTelemetrySource().start()
        if not telemetry.wait_first(5.0):
            telemetry.close()
            raise RuntimeError("no AC telemetry for axis verification")
        print("[telemetry-before]", observed_controls(telemetry.latest_state()))
    client = ControlClient(port=args.port)
    status = client.arm()
    if not status.get("ok") or status.get("emergency"):
        raise RuntimeError(f"bridge refused arm: {status}")
    print(f"[axis-test] steer={args.steer:+.3f} throttle={args.throttle:.3f} "
          f"brake={args.brake:.3f} for {args.duration:g}s; F8 = emergency")
    started = time.perf_counter()
    try:
        while time.perf_counter() - started < args.duration:
            status = client.command(args.steer, args.throttle, args.brake)
            if not status.get("ok") or status.get("emergency"):
                raise RuntimeError(f"bridge stopped command: {status}")
            time.sleep(1.0 / args.hz)
        if telemetry is not None:
            print("[telemetry-during]", observed_controls(telemetry.latest_state()))
    finally:
        client.disarm()
        client.close()
        if telemetry is not None:
            telemetry.close()
    print("[axis-test] complete; output returned to neutral")


if __name__ == "__main__":
    main()
