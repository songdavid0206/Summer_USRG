"""
collect.py - 사람 주행 중 telemetry(state) / image 를 동시에 raw 기록
=========================================================
방법 C 구조에 맞춰, 두 로거(스크린샷 + 차량 state)를 '같은 시계(clock)'로 돌리되 각 스트림을
독립적으로(자기 주기로) 기록한다. 실시간 병합은 하지 않는다.
수집이 끝나면 recorder.dataset_builder 로 camera/ ego_state/ opponent/ 폴더를 만든다.

  [사람이 AC 주행]
        │  ego_server(UDP 2345) → ACTelemetrySource → TelemetryLogger.log(t, state, opponents)
        │  ScreenCapture.grab()  → ImageLogger.log(t, image)   # 우리가 직접 화면 캡처
        ▼
  telemetry_raw.parquet / image_index.parquet / (opponents_raw.parquet)   # raw 소스
        ▼  build_dataset()  (카메라 프레임 기준 정렬)
  camera/ + ego_state/ + opponent/opp_<id>/   (각 parquet+csv, 길이 동일)

telemetry 취득 방식 (B-2: 직접 UDP)
-----------------------------------
레포의 make_ac_env 대신, sensors_par 플러그인이 여는 ego_server(UDP)에 직접 붙어 state를 받는다.
- "connect"만 보내고 이후 아무것도 안 보내므로 서버가 제어를 적용하지 않음 → 사람 주행 비개입(읽기 전용).
- action(steer/throttle/brake)은 vjoy가 아니라 state의 steerAngle/accStatus/brakeStatus 를 라벨로 '읽는다'.
- 실측 확인 완료: state 90필드, 조작값 키 = steerAngle/accStatus/brakeStatus.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import math

from .recorder import TelemetryLogger, ImageLogger, ACTelemetrySource
from .recorder.ac_opponents import ACOpponentsSource
from .recorder.screen_capture import ScreenCapture
from .recorder.dataset_builder import build_dataset


class Clock:
    """수집 시작(perf_counter) 기준 상대 시간을 주는 단일 시계.
    두 로거가 반드시 이 시계를 공유해야 timestamp가 같은 축 위에 놓인다."""
    def __init__(self):
        self.t0 = time.perf_counter()

    def now(self) -> float:
        "수집 시작 후 지난 시간(초)"
        return time.perf_counter() - self.t0


def build_opponents(state: dict, opp_list: list[dict] | None) -> list[dict] | None:
    """opponents_server에서 받은 원본 상대차 리스트를 ego 기준 상대좌표로 변환.

    opp_list  : [{'id':.., 'world_position':[x,y,z], 'velocity':[x,y,z], 'speedKMH':.., 'yaw':..}]
    반환       : [{opp_id, rel_x, rel_y, rel_z, distance, opp_speedKMH, opp_yaw,
                  opp_vx, opp_vy, opp_vz}, ...] 또는 None

    ego world_position_x/y/z 와 상대차 world_position[0]/[1]/[2] 는 같은 좌표계(structures.py 확인).
    velocity 도 같은 축 순서. 구버전 플러그인(velocity 없음)이면 해당 컬럼은 None.
    """
    if not opp_list:
        return None
    ego_x = state.get("world_position_x")
    ego_y = state.get("world_position_y")
    ego_z = state.get("world_position_z")
    out: list[dict] = []
    for opp in opp_list:
        wp = opp.get("world_position") or [None, None, None]
        ox, oy, oz = wp[0], wp[1], wp[2]
        rel_x = (ox - ego_x) if (ox is not None and ego_x is not None) else None
        rel_y = (oy - ego_y) if (oy is not None and ego_y is not None) else None
        rel_z = (oz - ego_z) if (oz is not None and ego_z is not None) else None
        dist = (math.hypot(rel_x, rel_y)
                if (rel_x is not None and rel_y is not None) else None)
        vel = opp.get("velocity") or [None, None, None]
        row = {
            "opp_id": opp.get("id"),
            "rel_x": rel_x,
            "rel_y": rel_y,
            "rel_z": rel_z,
            "distance": dist,
            "opp_speedKMH": opp.get("speedKMH"),
            "opp_yaw": opp.get("yaw"),
            # 플러그인 실측 속도벡터(월드). 이전엔 speed+yaw로 재구성했었다.
            "opp_vx": vel[0],
            "opp_vy": vel[1],
            "opp_vz": vel[2],
        }
        # TyreContactPoint 실험 필드(플러그인이 보내는 경우에만). 상대차 차체 방향/치수.
        # tcp_center_* 는 검증용: world_position 과 일치해야 차량별로 동작하는 것.
        for k in ("tcp_ok", "yaw_body", "wheelbase", "track_width",
                  "tcp_center_x", "tcp_center_y"):
            if k in opp:
                row[k] = opp[k]
        if "tcp_center_x" in row and ox is not None:
            row["tcp_err"] = math.hypot(row["tcp_center_x"] - ox,
                                        row["tcp_center_y"] - oy)
        out.append(row)
    return out


def collect(out_dir: str, run_id: str | None = None,
            telemetry_hz: float = 50.0, image_hz: float = 50.0,
            max_seconds: float | None = None,
            host: str = "127.0.0.1", port: int = 2345, opp_port: int = 2346,
            capture_region: tuple[int, int, int, int] | None = None,
            capture_backend: str = "auto",
            image_format: str = "jpg") -> Path:
    """사람 주행 데이터 수집 메인 루프.

    telemetry_hz / image_hz : 각 스트림의 목표 기록 주파수(서로 달라도 됨).
    max_seconds             : 수집 시간 제한(테스트용). None이면 Ctrl+C 까지.
    host / port             : sensors_par ego_server 주소(기본 127.0.0.1:2345).
    capture_region          : (left, top, width, height). None이면 주 모니터 전체.
    capture_backend         : "auto"(dxcam→mss) | "dxcam" | "mss".
    image_format            : "jpg"(빠름/작음) | "png"(무손실/느림).
    """
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out_dir) / run_id

    # telemetry(ego+조작+상대차)는 AC 플러그인 UDP 서버에서, 이미지는 우리가 직접 화면 캡처.
    print(f"[telemetry] ego_server 연결 시도 {host}:{port} ...")
    tel_src = ACTelemetrySource(host=host, port=port).start()
    if not tel_src.wait_first(timeout=5.0):
        tel_src.close()
        raise RuntimeError("state 수신 실패: 세션(트랙)에 진입했는지, 플러그인이 켜져 있는지 확인.")
    print(f"[telemetry] 연결 OK (state 수신 중). 필드 수={len(tel_src.latest_state())}")

    # 상대차(opponents_server, TCP): 상대 있는 세션에서만 열림 → 실패해도 계속 진행
    opp_src = None
    try:
        opp_src = ACOpponentsSource(host=host, port=opp_port).start()
        if opp_src.wait_first(timeout=2.0):
            print(f"[opponents] 연결 OK (상대차 {opp_src.n_opponents}대)")
        else:
            print("[opponents] 접속은 됐지만 상대차 데이터 미수신 → 계속 진행")
    except ConnectionError:
        print("[opponents] 서버 없음 → 상대차 미기록(혼자 주행이면 정상)")

    screen = ScreenCapture(region=capture_region, backend=capture_backend)
    print(f"[capture] backend={screen.backend} region={capture_region or 'full monitor'}")
    tel_logger = TelemetryLogger(run_dir)
    img_logger = ImageLogger(run_dir, image_ext=image_format)
    print(f"[capture] image_format={img_logger.image_ext}")

    clock = Clock()
    tel_period = 1.0 / telemetry_hz
    img_period = 1.0 / image_hz
    next_tel = 0.0
    next_img = 0.0

    print(f"[collect] telemetry {telemetry_hz}Hz / image {image_hz}Hz. Ctrl+C 로 종료.")
    try:
        while True:
            now = clock.now()

            # telemetry: 자기 주기마다 최근 state를 읽어 기록 (제어는 보내지 않음)
            if now >= next_tel:
                state = tel_src.latest_state()
                if state is not None:
                    opps = build_opponents(state, opp_src.latest()) if opp_src else None
                    tel_logger.log(now, state, opponents=opps)
                next_tel += tel_period

            # image: 자기 주기마다 화면 캡처 기록 (우리가 직접 캡처)
            if now >= next_img:
                image = screen.grab()
                if image is not None:
                    img_logger.log(now, image)
                next_img += img_period

            if max_seconds is not None and now >= max_seconds:
                break

            # 다음 이벤트까지 살짝 쉬기 (busy-wait 방지)
            sleep = min(next_tel, next_img) - clock.now()
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\n[stop] 사용자 종료")
    finally:
        elapsed = clock.now()
        tel_summary = tel_logger.flush()
        img_summary = img_logger.flush()
        # 실제 달성 Hz = 기록된 개수 / 총 수집시간
        tel_hz_actual = tel_summary.get("telemetry_rows", 0) / elapsed if elapsed > 0 else 0
        img_hz_actual = img_summary.get("image_frames", 0) / elapsed if elapsed > 0 else 0
        opp_hz_actual = (opp_src.received / elapsed) if (opp_src and elapsed > 0) else 0
        print(f"[rate] 실측  telemetry {tel_hz_actual:.1f}Hz / image {img_hz_actual:.1f}Hz "
              f"/ opponents {opp_hz_actual:.1f}Hz  (목표 {telemetry_hz}/{image_hz}Hz, {elapsed:.1f}s)")
        meta = {
            "run_dir": str(run_dir),
            "source": "human_driving",
            "telemetry_hz": telemetry_hz,
            "image_hz": image_hz,
            "telemetry_hz_actual": round(tel_hz_actual, 2),
            "image_hz_actual": round(img_hz_actual, 2),
            "elapsed_seconds": round(elapsed, 2),
            "image_format": img_logger.image_ext,
            "ego_server": f"{host}:{port}",
            "opponents_server": f"{host}:{opp_port}" if opp_src else None,
            "telemetry_recv_total": tel_src.received,
            "opponents_recv_total": opp_src.received if opp_src else 0,
            "capture_backend": screen.backend,
            "capture_region": capture_region,
            **tel_summary,
            **img_summary,
        }
        (run_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        screen.close()
        tel_src.close()
        if opp_src:
            opp_src.close()

    # 수집 종료 후 오프라인 정렬 → camera/ ego_state/ opponent/ 폴더 구조로 저장
    build_dataset(run_dir)

    # 맵정보 + 레이싱 라인 추출 → track/ 폴더 (best-effort; 트랙 파일 없거나 식별 실패해도 계속)
    try:
        from .track import extract_track_for_run
        extract_track_for_run(run_dir)
    except Exception as e:
        print(f"[track] 맵/레이싱라인 추출 스킵: {e}")

    # 후처리 → processed/ 폴더 (E2E/IL 학습용 선별·가공 피처). raw는 그대로 보존.
    try:
        from .processed_builder import build_processed
        build_processed(run_dir)
    except Exception as e:
        print(f"[processed] 후처리 스킵: {e}")

    # 프레임별 txt 데이터셋 → dataset/<run_id>_<track>_<hz>hz/ (CAMERA/EGO_INFO/OBJECT_INFO)
    try:
        from .export_frames import export_run
        export_run(run_dir)
    except Exception as e:
        print(f"[export] 내보내기 스킵: {e}")

    return run_dir


def main():
    ap = argparse.ArgumentParser(description="AC human-driving dataset collector (UDP telemetry)")
    ap.add_argument("--out", default="./dataset", help="데이터셋 저장 폴더")
    ap.add_argument("--telemetry-hz", type=float, default=50.0)
    ap.add_argument("--image-hz", type=float, default=50.0)
    ap.add_argument("--max-seconds", type=float, default=None, help="수집 시간 제한(초)")
    ap.add_argument("--host", default="127.0.0.1", help="ego_server 호스트")
    ap.add_argument("--port", type=int, default=2345, help="ego_server 포트")
    ap.add_argument("--opp-port", type=int, default=2346, help="opponents_server 포트")
    ap.add_argument("--capture-backend", default="auto",
                    choices=["auto", "dxcam", "mss"], help="화면 캡처 백엔드")
    ap.add_argument("--capture-region", default=None,
                    help="left,top,width,height (예: 0,0,640,480). 생략 시 주 모니터 전체")
    ap.add_argument("--image-format", default="jpg", choices=["jpg", "png"],
                    help="이미지 저장 포맷: jpg(빠름/작음, 기본) | png(무손실/느림)")
    args = ap.parse_args()

    region = (tuple(int(x) for x in args.capture_region.split(","))
              if args.capture_region else None)

    collect(out_dir=args.out,
            telemetry_hz=args.telemetry_hz, image_hz=args.image_hz,
            max_seconds=args.max_seconds, host=args.host, port=args.port,
            opp_port=args.opp_port,
            capture_region=region, capture_backend=args.capture_backend,
            image_format=args.image_format)


if __name__ == "__main__":
    main()
