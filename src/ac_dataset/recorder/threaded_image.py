"""ThreadedImageRecorder - 이미지 캡처/인코딩을 메인 루프에서 분리
==================================================================

문제: collect 가 단일 루프라 이미지 처리(grab ~6ms + JPG 인코딩·저장 ~17ms = ~23ms)가
루프를 잡아 telemetry 까지 20ms(50Hz) 안에 못 들어간다. 게임은 telemetry 를 ~70Hz 로
주는데도 기록은 39~44Hz 로 떨어졌다.

해법: 이미지 경로를 두 스레드로 분리한다.
  - **캡처 스레드**: image_hz 주기로 screen.grab() 만(6ms) 하고 (t, frame) 을 큐에 넣는다.
  - **인코더 워커**: 큐에서 꺼내 JPG 인코딩+저장(17ms). 프레임 순서 유지(단일 워커).
dxcam.grab 과 PIL 인코딩은 네이티브 호출 중 GIL 을 놓으므로 실제로 병렬 실행된다.
→ 메인 루프는 telemetry 만 → 50Hz 유지. 이미지는 grab/인코딩이 파이프라인으로 겹쳐 ~57Hz 상한.

큐는 bounded. 인코더가 밀리면 캡처가 잠깐 블록돼 이미지 Hz 만 내려가고(메모리 안전),
**telemetry(메인 루프)는 영향 없다.** 프레임을 버리지 않아 데이터 유실도 없다.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np

from ..ac_shmem import ACCameraPoseShm, ACGraphicsShm
from .image_logger import ImageLogger


class ThreadedImageRecorder:
    def __init__(self, run_dir: str | Path, clock, screen,
                 image_hz: float = 50.0, image_ext: str = "jpg",
                 jpg_quality: int = 85, queue_max: int = 30):
        """
        clock  : collect 의 Clock (telemetry 와 같은 시계여야 timestamp 축이 일치).
        screen : ScreenCapture. 캡처 스레드가 단독 소유(dxcam grab 은 한 스레드에서만).
        queue_max : 대기 프레임 상한. 인코더가 밀리면 여기서 캡처가 블록된다(메모리 보호).
        """
        self.clock = clock
        self.screen = screen
        self.image_hz = image_hz
        self.period = 1.0 / image_hz
        self.logger = ImageLogger(run_dir, image_ext=image_ext, jpg_quality=jpg_quality)
        self._q: queue.Queue = queue.Queue(maxsize=queue_max)
        self._stop = threading.Event()
        self._cap_thread: threading.Thread | None = None
        self._enc_thread: threading.Thread | None = None
        self.captured = 0            # grab 성공 수
        self.max_queue_seen = 0      # 인코더가 얼마나 밀렸는지 진단용
        # AC Graphics 공유메모리: grab 순간의 '화면에 그려진 프레임 게임시간'을 이미지에 같이 기록해
        # 이미지↔telemetry 를 게임시계로 페어링(시간동기, 방법 a). 게임 미실행이면 available=False.
        self.gshm = ACGraphicsShm()
        self.camera_shm = ACCameraPoseShm()

    @property
    def image_ext(self) -> str:
        return self.logger.image_ext

    def start(self) -> "ThreadedImageRecorder":
        print(f"[capture] AC Graphics 공유메모리 = {'연결(게임시간 기록 ON)' if self.gshm.available else '없음(게임시간 미기록 — 게임 실행/세션 진입 확인)'}")
        print(f"[capture] CSP 실제 카메라 pose = "
              f"{'연결(프레임별 3D 투영 ON)' if self.camera_shm.available else '없음 — ac_camera_pose_bridge 설치/활성화 필요'}")
        self._cap_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._enc_thread = threading.Thread(target=self._encode_loop, daemon=True)
        self._enc_thread.start()
        self._cap_thread.start()
        return self

    def _capture_loop(self) -> None:
        next_img = self.clock.now()
        while not self._stop.is_set():
            now = self.clock.now()
            if now >= next_img:
                frame = self.screen.grab()
                if frame is not None:
                    # grab 직후 즉시 '지금 그려진 프레임의 게임시간'을 읽는다(마이크로초, 큐잉 전).
                    g_pid, g_time, g_laps = self.gshm.read_full()
                    # dxcam 은 내부 버퍼를 재사용할 수 있어 copy 로 스냅샷을 뜬다(인코더가 늦게 읽어도 안전).
                    # CSP Lua update가 현재 present 직후 아직 안 끝났을 수 있으므로 pose는 여기서
                    # 읽지 않는다. 인코더가 몇 ms 뒤 링 버퍼의 과거 game time을 조회한다.
                    self._q.put((now, g_pid, g_time, g_laps,
                                 np.array(frame, copy=True)))  # 큐 가득이면 블록
                    self.captured += 1
                    self.max_queue_seen = max(self.max_queue_seen, self._q.qsize())
                next_img += self.period
            sleep = next_img - self.clock.now()
            if sleep > 0:
                time.sleep(sleep)

    def _encode_loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:        # 종료 sentinel
                break
            t, g_pid, g_time, g_laps, frame = item
            camera_pose = self.camera_shm.read(g_time, g_laps, g_pid)
            self.logger.log(t, frame, graphics_packet=g_pid, game_time_ms=g_time,
                            completed_laps=g_laps, camera_pose=camera_pose)

    def stop(self) -> dict:
        """캡처 중단 → 큐 소진(남은 프레임 전부 저장) → index flush. 요약 반환."""
        self._stop.set()
        if self._cap_thread:
            self._cap_thread.join()
        self._q.put(None)           # 인코더에게 '남은 것 다 처리하고 종료'
        if self._enc_thread:
            self._enc_thread.join()
        self.gshm.close()
        self.camera_shm.close()
        summary = self.logger.flush()
        summary["image_captured"] = self.captured
        summary["image_max_queue"] = self.max_queue_seen
        return summary
