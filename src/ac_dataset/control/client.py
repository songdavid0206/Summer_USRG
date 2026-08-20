"""UDP client for the independent local vJoy watchdog bridge."""

from __future__ import annotations

import json
import socket
import time
import uuid


class ControlClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 2350,
                 arm_token: str = "ARM_TCP_AC", timeout: float = 0.1):
        self.address = (host, int(port))
        self.arm_token = arm_token
        self.session = uuid.uuid4().hex
        self.sequence = 0
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(timeout)

    def _exchange(self, message: dict) -> dict:
        message.update({"token": self.arm_token, "session": self.session})
        sent_ns = time.perf_counter_ns()
        self.socket.sendto(json.dumps(message).encode(), self.address)
        data, _ = self.socket.recvfrom(4096)
        received_ns = time.perf_counter_ns()
        status = json.loads(data.decode())
        if not isinstance(status, dict):
            raise RuntimeError("invalid vJoy bridge response")
        # Local diagnostic timestamps are backward-compatible extra fields.
        status["client_sent_perf_ns"] = sent_ns
        status["client_received_perf_ns"] = received_ns
        status["client_roundtrip_ms"] = (received_ns - sent_ns) / 1e6
        return status

    def arm(self) -> dict:
        return self._exchange({"kind": "arm"})

    def command(self, steer: float, throttle: float, brake: float) -> dict:
        self.sequence += 1
        return self._exchange({
            "kind": "command", "sequence": self.sequence,
            "steer": float(steer), "throttle": float(throttle), "brake": float(brake),
        })

    def emergency(self, reason: str) -> dict:
        return self._exchange({"kind": "emergency", "reason": reason})

    def disarm(self) -> dict | None:
        try:
            return self._exchange({"kind": "disarm"})
        except (OSError, TimeoutError, socket.timeout):
            return None

    def close(self) -> None:
        self.socket.close()
