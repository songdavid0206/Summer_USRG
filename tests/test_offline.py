"""
오프라인 테스트 (게임 불필요)
============================

telemetry(50Hz) 와 image(20Hz)를 '서로 다른 주기 + 약간의 지터'로 생성해서
raw 로 쌓은 뒤, sync_run 이 이미지마다 가장 가까운 telemetry를 제대로 붙이는지 검증한다.
상대차(opponents)도 몇 프레임에 넣어 opp_count가 반영되는지 확인.

실행:  python tests/test_offline.py   (ac_dataset/ 에서)
"""

import os
import sys
import shutil

import numpy as np

# ac_dataset/ 를 import 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recorder import TelemetryLogger, ImageLogger      # noqa: E402
from recorder.sync import sync_run                      # noqa: E402
from schema import ACTION_MAP                           # noqa: E402


def fake_state(i: int) -> dict:
    """레포 env.state 흉내. action 원본 키(steerAngle/accStatus/brakeStatus) 포함."""
    return {
        "speed": 100.0 + i,
        "velocity_x": 27.7, "velocity_y": 0.1, "velocity_z": 0.0,
        "local_velocity_x": 27.8, "local_velocity_y": 0.0, "local_velocity_z": 0.0,
        "world_position_x": float(i), "world_position_y": float(i * 2),
        "yaw": 0.01 * i, "actualGear": 4, "RPM": 8000, "LapDist": 12.5 * i,
        "out_of_track": 0,
        # --- action (사람 조작값 라벨) ---
        "steerAngle": 0.001 * i, "accStatus": 0.8, "brakeStatus": 0.0,
    }


def run(out_root: str) -> None:
    run_dir = os.path.join(out_root, "sync_check")
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)

    tel = TelemetryLogger(run_dir)
    img = ImageLogger(run_dir)

    duration = 2.0            # 초
    tel_dt, img_dt = 1 / 50.0, 1 / 20.0
    rng = np.random.default_rng(0)

    # telemetry 50Hz (±2ms 지터, opponents 일부 tick에 포함)
    t = 0.0
    i = 0
    while t < duration:
        jitter = float(rng.uniform(-0.002, 0.002))
        opponents = None
        if i % 10 == 0:  # 가끔 상대차 2대
            opponents = [
                {"opp_id": 1, "rel_x": 12.0, "rel_y": -1.5, "rel_vx": -2.0,
                 "rel_vy": 0.2, "distance": 12.1},
                {"opp_id": 2, "rel_x": -8.0, "rel_y": 2.0, "rel_vx": 1.4,
                 "rel_vy": -0.1, "distance": 8.2},
            ]
        tel.log(t + jitter, fake_state(i), opponents=opponents)
        t += tel_dt
        i += 1

    # image 20Hz (±5ms 지터)
    t = 0.0
    while t < duration:
        jitter = float(rng.uniform(-0.005, 0.005))
        frame = rng.integers(0, 255, size=(64, 96, 3), dtype=np.uint8)
        img.log(t + jitter, frame)
        t += img_dt

    tel.flush()
    img.flush()

    samples = sync_run(run_dir, tolerance=0.05)

    # ---- 검증 ----
    assert len(samples) == img.frame_idx, "모든 이미지가 매칭돼야 함(주기 겹침 충분)"
    assert samples["dt"].max() < 0.05, "매칭 시간차가 tolerance 이내여야 함"
    for canon in ACTION_MAP:  # steer/throttle/brake 라벨 컬럼 존재
        assert canon in samples.columns, f"action 컬럼 누락: {canon}"
    assert "speed" in samples.columns, "ego 컬럼(speed) 누락"
    assert (samples["opp_count"] > 0).any(), "상대차 opp_count 반영 안 됨"
    assert os.path.exists(os.path.join(run_dir, "opponents_raw.parquet"))

    print("\n--- samples.parquet head ---")
    print(samples[["frame", "image_path", "t_img", "t_tel", "dt",
                   "opp_count", "speed", "steer", "throttle", "brake"]].head())
    print(f"\nOK: {len(samples)} samples, opp_count>0 프레임 "
          f"{int((samples['opp_count'] > 0).sum())}개. 동기화/스키마 정상.")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    run(os.path.join(here, "_out"))
