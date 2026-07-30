"""
ScreenCapture - 게임 화면을 직접 캡처
=====================================

참조 레포는 (Python 3.3 제약 때문에) 별도 프로세스 + shared memory로 캡처했지만,
우리 collect.py는 Python 3.9+에서 돌기 때문에 그럴 필요가 없다.
여기서 직접 화면을 캡처해 collect 루프의 같은 Clock으로 timestamp를 찍는다.
-> 프로세스가 하나라 image와 telemetry 동기화가 더 정확.

백엔드
-----
- wgc   : Windows Graphics Capture로 특정 창만 캡처. 다른 창이 앞에 와도 대상 창을 유지.
- dxcam : Windows Desktop Duplication 기반. mss보다 훨씬 빠름(게임 FPS 덜 깎임). 우선 사용.
- mss   : dxcam 미설치/실패 시 폴백. 느릴 수 있음(레포도 640x480/25Hz로 제한).

반환
----
grab() -> numpy (H, W, 3) uint8, RGB 순서. ImageLogger가 그대로 PNG로 저장.

주의
----
- AC를 '전체화면 독점(fullscreen exclusive)'으로 두면 캡처가 검게 나올 수 있음
  -> '테두리 없는 창(borderless windowed)' 권장.
- HUD(속도계/맵)가 화면에 찍히면 학습이 편법으로 흐를 수 있음 -> 게임에서 HUD 끄거나
  region/crop으로 잘라내는 걸 권장(아래 region 인자).
"""

from __future__ import annotations

import numpy as np


class ScreenCapture:
    def __init__(self, region: tuple[int, int, int, int] | None = None,
                 backend: str = "auto", monitor_idx: int = 0,
                 window_title: str | None = None):
        """
        region      : (left, top, width, height). wgc에서는 창 client 기준 crop,
                      나머지는 모니터 기준 좌표.
        backend     : "wgc" | "dxcam" | "mss" | "auto".
        monitor_idx : 다중 모니터에서 캡처할 모니터 인덱스(0 = 주 모니터).
        window_title: wgc로 캡처할 창 제목. 일부 문자열도 허용한다.
        """
        self.region = region
        self.monitor_idx = monitor_idx
        self.window_title = window_title
        self.resolved_window_title: str | None = None
        self._last_frame: np.ndarray | None = None
        self.backend = None
        self._impl = None

        if window_title:
            if backend not in {"auto", "wgc"}:
                raise ValueError("window_title은 wgc/auto backend에서만 사용할 수 있습니다")
            # 창 캡처 요청을 모니터 캡처로 조용히 폴백하면 다른 앱 화면이
            # 데이터에 들어가므로 실패를 명시적으로 알린다.
            order = ["wgc"]
        else:
            order = ["dxcam", "mss"] if backend == "auto" else [backend]
        errors = []
        for name in order:
            try:
                if name == "wgc":
                    self._init_wgc()
                elif name == "dxcam":
                    self._init_dxcam()
                elif name == "mss":
                    self._init_mss()
                self.backend = name
                break
            except Exception as e:  # 설치 안 됐거나 이 환경에서 실패
                errors.append(f"{name}: {e}")
        if self.backend is None:
            raise RuntimeError(
                "화면 캡처 백엔드 초기화 실패. wgc는 `pip install windows-cap`, "
                "모니터 캡처는 `pip install dxcam` 또는 `pip install mss` 필요.\n"
                + "\n".join(errors))

    # ---- 백엔드별 초기화 -----------------------------------------------------
    @staticmethod
    def _window_titles() -> list[str]:
        """현재 보이는 최상위 Windows 창 제목을 열거한다."""
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        titles: list[str] = []
        callback_type = ctypes.WINFUNCTYPE(
            wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        @callback_type
        def callback(hwnd, _lparam):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0 and user32.IsWindowVisible(hwnd):
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                if buf.value:
                    titles.append(buf.value)
            return True

        user32.EnumWindows(callback, 0)
        return titles

    def _init_wgc(self):
        if not self.window_title:
            raise ValueError("wgc backend에는 window_title이 필요합니다")
        from windows_cap import WindowCapture

        query = self.window_title.casefold()
        titles = self._window_titles()
        exact = [title for title in titles if title.casefold() == query]
        partial = [title for title in titles if query in title.casefold()]
        matches = exact or partial
        if not matches:
            raise RuntimeError(
                f"창 제목에 {self.window_title!r}을 포함한 창을 찾지 못했습니다. "
                f"현재 창: {titles}")
        if len(matches) > 1:
            raise RuntimeError(
                f"창 제목이 여러 개와 일치합니다: {matches}. 더 정확히 지정하세요")
        self.resolved_window_title = matches[0]
        self._impl = WindowCapture(self.resolved_window_title)
        self._impl.start_capture()

    def _init_dxcam(self):
        import dxcam
        # RGB로 바로 받기. output_idx = 모니터.
        self._impl = dxcam.create(output_idx=self.monitor_idx, output_color="RGB")
        if self._impl is None:
            raise RuntimeError("dxcam.create()가 None 반환(모니터 인덱스 확인)")
        # dxcam region은 (left, top, right, bottom) 형식으로 변환
        self._dxcam_region = None
        if self.region is not None:
            l, t, w, h = self.region
            self._dxcam_region = (l, t, l + w, t + h)

    def _init_mss(self):
        import mss
        self._mss = mss  # 모듈 보관
        # mss.MSS(신규) 우선, 없으면 mss.mss(구버전) 폴백
        self._sct = (mss.MSS() if hasattr(mss, "MSS") else mss.mss())
        if self.region is not None:
            l, t, w, h = self.region
            self._mss_monitor = {"left": l, "top": t, "width": w, "height": h}
        else:
            # monitors[0]=전체 가상화면, [1]=주 모니터. monitor_idx+1로 개별 모니터.
            self._mss_monitor = self._sct.monitors[self.monitor_idx + 1]

    # ---- 캡처 ----------------------------------------------------------------
    def grab(self) -> np.ndarray | None:
        """한 프레임 캡처해 (H,W,3) uint8 RGB 반환. 실패 시 마지막 프레임."""
        if self.backend == "wgc":
            raw = self._impl.next()
            width, height = self._impl.client_size
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
            arr = arr[:, :, :3][:, :, ::-1]  # BGRA -> RGB
            if self.region is not None:
                left, top, crop_width, crop_height = self.region
                right, bottom = left + crop_width, top + crop_height
                if (left < 0 or top < 0 or right > width or bottom > height):
                    raise ValueError(
                        f"wgc crop {self.region}이 창 client {width}x{height} 밖입니다")
                arr = arr[top:bottom, left:right]
            self._last_frame = np.ascontiguousarray(arr)
            return self._last_frame

        if self.backend == "dxcam":
            frame = self._impl.grab(region=self._dxcam_region)  # 변화 없으면 None 가능
            if frame is None:
                return self._last_frame
            self._last_frame = frame
            return frame

        # mss: BGRA -> RGB
        raw = self._sct.grab(self._mss_monitor)
        arr = np.asarray(raw)[:, :, :3][:, :, ::-1]  # BGRA에서 B,G,R만 -> RGB로 뒤집기
        arr = np.ascontiguousarray(arr)
        self._last_frame = arr
        return arr

    def close(self):
        try:
            if self.backend == "dxcam" and self._impl is not None:
                self._impl.release()
            elif self.backend == "wgc":
                self._impl = None
            elif self.backend == "mss":
                self._sct.close()
        except Exception:
            pass


# 단독 실행: 한 프레임 찍어 저장(백엔드/환경 점검용)
#   python recorder/screen_capture.py --save shot.png
if __name__ == "__main__":
    import argparse
    from PIL import Image

    ap = argparse.ArgumentParser(description="화면 캡처 스모크 테스트")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "wgc", "dxcam", "mss"])
    ap.add_argument("--window-title", default=None,
                    help="wgc 대상 창 제목 일부(예: Assetto Corsa)")
    ap.add_argument("--region", default=None,
                    help="left,top,width,height (예: 0,0,640,480). 생략 시 전체 모니터")
    ap.add_argument("--save", default="capture_test.png")
    args = ap.parse_args()

    region = tuple(int(x) for x in args.region.split(",")) if args.region else None
    cap = ScreenCapture(region=region, backend=args.backend,
                        window_title=args.window_title)
    frame = cap.grab()
    cap.close()

    if frame is None:
        print("[fail] 프레임을 못 받음")
    else:
        print(f"[ok] backend={cap.backend} shape={frame.shape} dtype={frame.dtype}")
        Image.fromarray(frame).save(args.save)
        print(f"[saved] {args.save}")
