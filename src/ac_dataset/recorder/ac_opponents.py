"""
ACOpponentsSource - sensors_par 플러그인 opponents_server에서 상대차 목록을 받는 TCP 클라이언트
==========================================================================

상대차 정보는 ego_server(2345)가 아니라 별도 opponents_server(TCP, 기본 2346)로 나온다.
이 서버는 상대차가 있는 세션에서만 열린다(혼자 주행이면 안 열림).

프로토콜 (sensors_par.py opponents_server_task 기준)
--------------------------------------------------
- 접속하면 서버가 먼저 상대차 수를 문자열로 보냄(예: "2").
- 이후 계속 `<길이>HEADER-END<payload>` 형식으로 보냄.
  payload = str(list[Opponent])  = 파이썬 리스트 리터럴, 예:
    [{'id': 1, 'world_position': [x, y, z], 'speedKMH': 120.5, 'yaw': 0.53}, {...}]

파싱은 길이 프리픽스 대신 '균형 잡힌 [...] 추출'로 한다(수 프리픽스/HEADER-END에 영향 안 받게).
ast.literal_eval 로 안전하게 dict 리스트로 변환한다.

사용법
------
    opp = ACOpponentsSource().start()   # 상대차 없으면 ConnectionError
    lst = opp.latest()                  # [{'id':.., 'world_position':[..], ...}, ...] 또는 None
    opp.close()
"""

from __future__ import annotations

import ast
import socket
import threading


class ACOpponentsSource:
    def __init__(self, host: str = "127.0.0.1", port: int = 2346,
                 timeout: float = 2.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.n_opponents: int | None = None
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._latest: list[dict] | None = None
        self._count = 0

    def start(self) -> "ACOpponentsSource":
        """opponents_server에 접속. 서버가 없으면(=상대차 없음) ConnectionError."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        try:
            self._sock.connect((self.host, self.port))
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            self._sock.close()
            self._sock = None
            raise ConnectionError(
                "opponents_server 접속 실패 ({}:{}). 상대차 없는 세션이면 정상."
                .format(self.host, self.port)) from e
        # n_opponents 는 첫 프레임을 파싱하면 리스트 길이로 확정된다(아래 _recv_loop).
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        return self

    def _recv_loop(self) -> None:
        buf = ""
        while self._running:
            try:
                data = self._sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                break
            buf += data.decode(errors="ignore")

            # 버퍼에서 균형 잡힌 [...] 를 전부 뽑아 마지막 것만 보관
            latest = None
            while True:
                i = buf.find("[")
                if i == -1:
                    buf = ""            # 쓸 데이터 없음
                    break
                depth = 0
                end = -1
                for j in range(i, len(buf)):
                    c = buf[j]
                    if c == "[":
                        depth += 1
                    elif c == "]":
                        depth -= 1
                        if depth == 0:
                            end = j
                            break
                if end == -1:
                    buf = buf[i:]       # 아직 안 닫힘 → 더 받기
                    break
                payload = buf[i:end + 1]
                buf = buf[end + 1:]
                try:
                    lst = ast.literal_eval(payload)
                    if isinstance(lst, list):
                        latest = lst
                except (ValueError, SyntaxError):
                    pass

            if latest is not None:
                with self._lock:
                    self._latest = latest
                    self._count += 1
                    if self.n_opponents is None:
                        self.n_opponents = len(latest)

    def wait_first(self, timeout: float = 2.0) -> bool:
        """첫 상대차 리스트를 파싱할 때까지 대기. 받으면 True."""
        import time as _time
        t_end = _time.time() + timeout
        while _time.time() < t_end:
            if self.latest() is not None:
                return True
            _time.sleep(0.05)
        return False

    def latest(self) -> list[dict] | None:
        """가장 최근 상대차 리스트(복사본). 아직 없으면 None."""
        with self._lock:
            return None if self._latest is None else [dict(o) for o in self._latest]

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
            self._sock.close()
            self._sock = None
