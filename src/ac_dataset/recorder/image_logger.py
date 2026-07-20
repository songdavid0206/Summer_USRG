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
        self._render_car_rows: list[dict] = []
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

    def log(self, t: float, image: np.ndarray, wall_time: float | None = None,
            graphics_packet: int | None = None, game_time_ms: int | None = None,
            completed_laps: int | None = None,
            camera_pose: dict | None = None) -> str:
        """프레임 한 장 저장 + index row 기록. 상대경로(camera/xxxxxx.png) 반환.

        t : 수집 시작 기준 상대 시간(초). telemetry 로거와 '같은 시계'.
        graphics_packet / game_time_ms : grab 순간 AC Graphics 공유메모리에서 읽은
            '화면에 그려진 프레임'의 packetId 와 게임시간(ms). 이미지↔telemetry 를 게임시계로
            페어링(시간동기)하는 데 쓴다. 공유메모리 없으면 None.
        """
        arr = self._to_uint8(image)
        rel_path = f"camera/{self.frame_idx:06d}.{self.image_ext}"
        img = Image.fromarray(arr)
        if self.image_ext == "jpg":
            # JPG: PNG보다 인코딩 훨씬 빠르고 용량 작음(무손실은 아님). 품질 95로 손실 최소화.
            img.save(self.run_dir / rel_path, quality=self.jpg_quality)
        else:
            img.save(self.run_dir / rel_path)

        row = {
            "frame": self.frame_idx,
            "t": t,
            "wall_time": wall_time if wall_time is not None else time.time(),
            "image_path": rel_path,
            "graphics_packet": graphics_packet,
            "game_time_ms": game_time_ms,
            "completed_laps": completed_laps,
        }
        # CSP가 제공한 실제 렌더 카메라 값만 저장한다. 이미지 기반 보정값은 받지 않는다.
        if camera_pose:
            row.update({k: v for k, v in camera_pose.items() if not k.startswith("_")})
            for car in camera_pose.get("_render_cars", []):
                self._render_car_rows.append({"image_frame": self.frame_idx, "t": t, **car})
        self._rows.append(row)
        self.frame_idx += 1
        return rel_path

    def flush(self) -> dict:
        df = pd.DataFrame(self._rows)
        path = self.run_dir / "image_index.parquet"
        df.to_parquet(path, index=False)
        print(f"[image] {len(df)} frames -> {path}")
        pose_frames = (int(df["cam_pos_x"].notna().sum())
                       if "cam_pos_x" in df.columns else 0)
        if len(df):
            print(f"[image] actual camera pose: {pose_frames}/{len(df)} "
                  f"({100.0 * pose_frames / len(df):.1f}%)")
            if "cam_pose_game_delta_ms" in df.columns:
                delta = df["cam_pose_game_delta_ms"].dropna()
                if len(delta):
                    print(f"[image] camera pose game-time delta: "
                          f"median={delta.median():.1f}ms / p95={delta.quantile(.95):.1f}ms "
                          f"/ max={delta.max():.1f}ms")
            if "cam_packet_render_error" in df.columns:
                err = df["cam_packet_render_error"].dropna()
                if len(err):
                    print(f"[image] packet/render exact: {(err == 0).sum()}/{len(err)} "
                          f"({100.0 * (err == 0).mean():.1f}%)")
        cars = pd.DataFrame(self._render_car_rows)
        if len(cars):
            cars.to_parquet(self.run_dir / "render_car_state.parquet", index=False)
            print(f"[image] render car states: {len(cars)} rows / "
                  f"{cars['image_frame'].nunique()} frames")
        render_car_frames = int(cars["image_frame"].nunique()) if len(cars) else 0
        return {"image_frames": len(df), "camera_pose_frames": pose_frames,
                "camera_pose_coverage": pose_frames / len(df) if len(df) else 0.0,
                "render_car_frames": render_car_frames,
                "render_car_rows": len(cars),
                "render_car_coverage": render_car_frames / len(df) if len(df) else 0.0}
