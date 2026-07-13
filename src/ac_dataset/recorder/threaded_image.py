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

from .image_logger import ImageLogger


class ThreadedImageRecorder:
    def __init__(self, run_dir: str | Path, clock, screen,
                 image_hz: float = 50.0, image_ext: str = "jpg",
                 jpg_quality: int = 95, queue_max: int = 30):
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

    @property
    def image_ext(self) -> str:
        return self.logger.image_ext

    def start(self) -> "ThreadedImageRecorder":
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
                    # dxcam 은 내부 버퍼를 재사용할 수 있어 copy 로 스냅샷을 뜬다(인코더가 늦게 읽어도 안전).
                    self._q.put((now, np.array(frame, copy=True)))  # 큐 가득이면 여기서 블록
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
            t, frame = item
            self.logger.log(t, frame)

    def stop(self) -> dict:
        """캡처 중단 → 큐 소진(남은 프레임 전부 저장) → index flush. 요약 반환."""
        self._stop.set()
        if self._cap_thread:
            self._cap_thread.join()
        self._q.put(None)           # 인코더에게 '남은 것 다 처리하고 종료'
        if self._enc_thread:
            self._enc_thread.join()
        summary = self.logger.flush()
        summary["image_captured"] = self.captured
        summary["image_max_queue"] = self.max_queue_seen
        return summary
