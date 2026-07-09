"""
ACTelemetrySource - sensors_par 플러그인 ego_server에서 실시간 state를 받는 UDP 클라이언트
==========================================================================

레포(assetto_corsa_gym)의 무거운 make_ac_env 를 쓰지 않고, 플러그인이 여는
ego_server(UDP, 기본 127.0.0.1:2345)에 직접 붙어 telemetry(state dict)를 받는다.

프로토콜 (ego_server.py 기준)
----------------------------
- 클라이언트가 "connect" 전송 → 서버가 "identified" 응답 후, 매 tick 차 상태(JSON)를
  현재 클라이언트에게 push 하기 시작한다(약 수십~수백 Hz).
- 클라이언트가 이후 아무 데이터도 보내지 않으면 서버는 제어(steer/acc/brake)를
  적용하지 않는다 → 사람 주행에 개입하지 않는 '읽기 전용' 수집이 된다.
- 종료 시 "disconnect" 전송.

사용법
------
    src = ACTelemetrySource().start()
    src.wait_first()                 # 첫 state 올 때까지 대기
    state = src.latest_state()       # 최근 state dict(복사본) 또는 None
    ...
    src.close()
"""

from __future__ import annotations

import json
import socket
import threading
import time

MAX_MSG_SIZE = 2 ** 17


class ACTelemetrySource:
    def __init__(self, host: str = "127.0.0.1", port: int = 2345,
                 timeout: float = 2.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._count = 0

    def start(self) -> "ACTelemetrySource":
        """서버에 connect 하고 백그라운드 수신 스레드를 띄운다."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(self.timeout)
        self._sock.sendto(b"connect", (self.host, self.port))
        try:
            self._sock.recvfrom(MAX_MSG_SIZE)  # "identified"
        except socket.timeout:
            self._sock.close()
            self._sock = None
            raise RuntimeError(
                "ego_server 응답 없음 ({}:{}). AC 실행 + sensors_par 플러그인 ON "
                "+ 세션(트랙) 진입 상태인지 확인.".format(self.host, self.port))
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        return self

    def _recv_loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(MAX_MSG_SIZE)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                d = json.loads(data.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue  # "identified" 등 비-JSON 무시
            if isinstance(d, dict):
                with self._lock:
                    self._latest = d
                    self._count += 1

    def latest_state(self) -> dict | None:
        """가장 최근에 받은 state dict의 복사본. 아직 없으면 None."""
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def wait_first(self, timeout: float = 5.0) -> bool:
        """첫 state를 받을 때까지 대기. 받으면 True."""
        t_end = time.time() + timeout
        while time.time() < t_end:
            if self.latest_state() is not None:
                return True
            time.sleep(0.05)
        return False

    @property
    def received(self) -> int:
        with self._lock:
            return self._count

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._sock is not None:
            try:
                self._sock.sendto(b"disconnect", (self.host, self.port))
            except OSError:
                pass
            self._sock.close()
            self._sock = None
