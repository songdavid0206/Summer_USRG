"""
ImageLogger - 화면 캡처 raw 스트림 기록
======================================

telemetry와 마찬가지로 '자기 주기'(예: 20Hz)로 들어오는 프레임을 그대로 저장하고,
각 프레임의 timestamp `t`를 index 테이블에 남긴다. telemetry와의 매칭은 sync가 담당.

출력
----
  <run_dir>/camera/000000.png, 000001.png, ...
  <run_dir>/image_index.parquet     # 한 행 = (frame, t, wall_time, image_path)  (raw 소스)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


class ImageLogger:
    def __init__(self, run_dir: str | Path, image_ext: str = "jpg",
                 jpg_quality: int = 95):
        self.run_dir = Path(run_dir)
        self.img_dir = self.run_dir / "camera"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        # jpeg -> jpg 로 통일
        self.image_ext = "jpg" if image_ext.lower() in ("jpg", "jpeg") else image_ext.lower()
        self.jpg_quality = jpg_quality
        self._rows: list[dict] = []
        self.frame_idx = 0

    @staticmethod
    def _to_uint8(image: np.ndarray) -> np.ndarray:
        """float(0~1) 또는 (H,W,1) 등 다양한 입력을 저장 가능한 uint8 배열로 정리.

        get_current_image()의 실제 반환 shape/dtype은 게임에서 확정 필요(방어적 처리).
        """
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            if arr.size and arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255)
            arr = arr.astype(np.uint8)
        if arr.ndim == 3 and arr.shape[-1] == 1:   # (H,W,1) -> (H,W)
            arr = arr[..., 0]
        return arr

    def log(self, t: float, image: np.ndarray, wall_time: float | None = None) -> str:
        """프레임 한 장 저장 + index row 기록. 상대경로(camera/xxxxxx.png) 반환.

        t : 수집 시작 기준 상대 시간(초). telemetry 로거와 '같은 시계'.
        """
        arr = self._to_uint8(image)
        rel_path = f"camera/{self.frame_idx:06d}.{self.image_ext}"
        img = Image.fromarray(arr)
        if self.image_ext == "jpg":
            # JPG: PNG보다 인코딩 훨씬 빠르고 용량 작음(무손실은 아님). 품질 95로 손실 최소화.
            img.save(self.run_dir / rel_path, quality=self.jpg_quality)
        else:
            img.save(self.run_dir / rel_path)

        self._rows.append({
            "frame": self.frame_idx,
            "t": t,
            "wall_time": wall_time if wall_time is not None else time.time(),
            "image_path": rel_path,
        })
        self.frame_idx += 1
        return rel_path

    def flush(self) -> dict:
        df = pd.DataFrame(self._rows)
        path = self.run_dir / "image_index.parquet"
        df.to_parquet(path, index=False)
        print(f"[image] {len(df)} frames -> {path}")
        return {"image_frames": len(df)}
