"""Independent vJoy owner with F8 emergency latch and packet watchdog."""

from __future__ import annotations

import argparse
import ctypes
import json
import socket
import time

from .vjoy import VJoyDevice


VK_F8 = 0x77


def _clamp(value, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def emergency_key_down() -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(VK_F8) & 0x8000)


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent local watchdog and vJoy output bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2350)
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=0.15,
                        help="command silence before latched emergency brake")
    parser.add_argument("--emergency-brake", type=float, default=1.0)
    parser.add_argument("--arm-token", default="ARM_TCP_AC")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.settimeout(0.01)
    device = VJoyDevice(args.device_id)
    armed = emergency = False
    reason = "disarmed"
    session = None
    last_command = None
    f8_was_down = False

    def apply_emergency(why: str) -> None:
        nonlocal armed, emergency, reason
        armed, emergency, reason = False, True, why
        device.set_controls(0.0, 0.0, args.emergency_brake)
        print(f"[EMERGENCY] {why}: steer=0 throttle=0 brake={args.emergency_brake:g}")

    print(f"[bridge] vJoy device={args.device_id} version=0x{device.version:04x}")
    print(f"[bridge] listening on udp://{args.host}:{args.port}; F8 = LATCHED EMERGENCY")
    print(f"[bridge] watchdog={args.timeout:g}s; restart bridge to clear an emergency latch")
    try:
        while True:
            now = time.perf_counter()
            f8_down = emergency_key_down()
            if f8_down and not f8_was_down and not emergency:
                apply_emergency("F8")
            f8_was_down = f8_down
            if armed and last_command is not None and now - last_command > args.timeout:
                apply_emergency("watchdog_timeout")
            try:
                payload, address = sock.recvfrom(4096)
            except socket.timeout:
                continue
            try:
                message = json.loads(payload.decode())
                if message.get("token") != args.arm_token:
                    raise ValueError("invalid arm token")
                kind = message.get("kind")
                incoming_session = message.get("session")
                if kind == "arm":
                    if emergency:
                        raise RuntimeError("emergency is latched; restart bridge")
                    session = incoming_session
                    armed, reason, last_command = True, "armed", now
                    device.set_controls(0.0, 0.0, 0.0)
                    print(f"[armed] session={session}")
                elif kind == "command":
                    if emergency or not armed or incoming_session != session:
                        raise RuntimeError("bridge is not armed for this session")
                    steer = _clamp(message["steer"], -1.0, 1.0)
                    throttle = _clamp(message["throttle"], 0.0, 1.0)
                    brake = _clamp(message["brake"], 0.0, 1.0)
                    device.set_controls(steer, throttle, brake)
                    last_command = now
                elif kind == "disarm":
                    if incoming_session == session and not emergency:
                        armed, reason, session = False, "disarmed", None
                        device.set_controls(0.0, 0.0, 0.0)
                        print("[disarmed]")
                elif kind == "emergency":
                    apply_emergency(str(message.get("reason", "client_emergency")))
                else:
                    raise ValueError(f"unknown message kind: {kind}")
                response = {"ok": True, "armed": armed, "emergency": emergency, "reason": reason}
            except Exception as exc:
                response = {"ok": False, "armed": armed, "emergency": emergency,
                            "reason": reason, "error": str(exc)}
            sock.sendto(json.dumps(response).encode(), address)
    except KeyboardInterrupt:
        print("[bridge] Ctrl+C; neutralizing and closing")
    finally:
        device.close()
        sock.close()


if __name__ == "__main__":
    main()
