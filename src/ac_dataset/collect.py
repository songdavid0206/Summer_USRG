"""
collect.py - 사람 주행 중 telemetry(state) / image 를 동시에 raw 기록
=========================================================
방법 C 구조에 맞춰, 두 로거(스크린샷 + 차량 state)를 '같은 시계(clock)'로 돌리되 각 스트림을
독립적으로(자기 주기로) 기록한다. 실시간 병합은 하지 않는다.
수집이 끝나면 recorder.dataset_builder 로 camera/ ego_state/ opponent/ 폴더를 만든다.

  [사람이 AC 주행]
        │  ego_server(UDP 2345) → ACTelemetrySource → TelemetryLogger.log(t, state)   # 메인 루프
        │  ScreenCapture.grab() → ThreadedImageRecorder → 캡처/인코딩 별도 스레드      # 메인 루프 안 막음
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
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import math

from .recorder import TelemetryLogger, ACTelemetrySource
from .recorder.ac_opponents import ACOpponentsSource
from .recorder.screen_capture import ScreenCapture
from .recorder.threaded_image import ThreadedImageRecorder
from .recorder.dataset_builder import build_dataset


def _camera_cfg_snapshot() -> dict:
    """게임 카메라 설정을 meta 에 스냅샷 — 어떤 온보드 뷰·FOV·흔들림으로 캡처했는지 기록.

    검증툴(av-view)이 이 값으로 3D bbox 투영 카메라를 **자동 선택**한다:
      camera_mode(camera_manager.ini LAST_CAMERA MODE) / camera_drivable_mode(DRIVABLE_MODE
      0=Chase 1=Chase2 2=Bonnet 3=Bumper 4=Dash) / onboard_fov_deg(수직) /
      camera_gforce_mix·camera_shake_scale(0 이 아니면 카메라가 흔들려 박스 어긋남).
    주의: LAST_CAMERA 는 게임이 갱신하는 '마지막 사용' 값 — 수집 중 뷰를 바꾸면 어긋날 수 있음.
    """
    out = {}
    try:
        from .export_frames import _ac_cfg_dir, _read_ini
        cm = _read_ini(_ac_cfg_dir() / "camera_manager.ini")["LAST_CAMERA"]
        out["camera_mode"] = int(cm["MODE"])
        out["camera_drivable_mode"] = int(cm["DRIVABLE_MODE"])
    except Exception:
        pass
    try:
        from .export_frames import _ac_cfg_dir, _read_ini
        co = _read_ini(_ac_cfg_dir() / "camera_onboard.ini")
        sec = "MODE" if co.has_option("MODE", "FOV") else "HEADER"
        out["onboard_fov_deg"] = float(co[sec]["FOV"])
        # IS_WORLD_ALIGNED=1 (Lock onboard camera to horizon): 카메라가 차체 roll 을 안 따라감
        # (pitch 는 따라감). 검증툴이 roll 적용여부 결정에 씀.
        if co.has_option("MODE", "IS_WORLD_ALIGNED"):
            out["camera_world_aligned"] = int(co["MODE"]["IS_WORLD_ALIGNED"])
        if co.has_section("GFORCES"):
            out["camera_gforce_mix"] = co["GFORCES"].get("MIX", "")
        if co.has_section("SHAKE"):
            out["camera_shake_scale"] = co["SHAKE"].get("SCALE", "")
    except Exception:
        pass
    return out


def _race_cfg_snapshot() -> dict:
    """Capture the session configuration while it still matches this run."""
    try:
        from .export_frames import _ac_cfg_dir, _read_ini
        config = _read_ini(_ac_cfg_dir() / "race.ini")
    except Exception:
        return {}
    race = config["RACE"] if config.has_section("RACE") else {}
    session = config["SESSION_0"] if config.has_section("SESSION_0") else {}
    weather = config["WEATHER"] if config.has_section("WEATHER") else {}
    temperature = config["TEMPERATURE"] if config.has_section("TEMPERATURE") else {}
    wind = config["WIND"] if config.has_section("WIND") else {}
    lighting = config["LIGHTING"] if config.has_section("LIGHTING") else {}
    dynamic_track = (
        dict(config["DYNAMIC_TRACK"]) if config.has_section("DYNAMIC_TRACK") else {})
    groove = dict(config["GROOVE"]) if config.has_section("GROOVE") else {}
    cars = []
    for section in sorted((s for s in config.sections() if s.startswith("CAR_")),
                          key=lambda name: int(name.split("_")[1])):
        car_id = int(section.split("_")[1])
        model = config[section].get("MODEL")
        if car_id == 0 and (not model or model == "-"):
            model = race.get("MODEL")
        cars.append({
            "car_id": car_id,
            "model": model,
            "ai_level": config[section].get("AI_LEVEL"),
            "ai_aggression": config[section].get("AI_AGGRESSION"),
            "ballast": config[section].get("BALLAST"),
            "restrictor": config[section].get("RESTRICTOR"),
        })
    out = {
        "race_track": race.get("TRACK"),
        "race_track_config": race.get("CONFIG_TRACK"),
        "race_laps": int(race.get("RACE_LAPS", 0) or 0),
        "race_car_count": int(race.get("CARS", len(cars)) or len(cars)),
        "race_ai_level": race.get("AI_LEVEL"),
        "race_cars": cars,
        "race_session_name": session.get("NAME"),
        "race_session_type": session.get("TYPE"),
        "race_duration_minutes": int(session.get("DURATION_MINUTES", 0) or 0),
        "weather_name": weather.get("NAME"),
        "ambient_temperature_c": (
            float(temperature["AMBIENT"]) if temperature.get("AMBIENT") else None),
        "road_temperature_c": (
            float(temperature["ROAD"]) if temperature.get("ROAD") else None),
        "wind_speed_kmh_min": (
            float(wind["SPEED_KMH_MIN"]) if wind.get("SPEED_KMH_MIN") else None),
        "wind_speed_kmh_max": (
            float(wind["SPEED_KMH_MAX"]) if wind.get("SPEED_KMH_MAX") else None),
        "wind_direction_deg": (
            float(wind["DIRECTION_DEG"]) if wind.get("DIRECTION_DEG") else None),
        "sun_angle_deg": (
            float(lighting["SUN_ANGLE"]) if lighting.get("SUN_ANGLE") else None),
        "time_multiplier": (
            float(lighting["TIME_MULT"]) if lighting.get("TIME_MULT") else None),
        "weather_controller": lighting.get("__CM_WEATHER_CONTROLLER"),
        "weather_type": lighting.get("__CM_WEATHER_TYPE"),
        "dynamic_track": dynamic_track,
        "groove": groove,
    }
    try:
        from .track import DEFAULT_TRACKS_DIR
        track_name = race.get("TRACK")
        track_config = race.get("CONFIG_TRACK")
        track_root = DEFAULT_TRACKS_DIR / track_name
        if track_config:
            track_root /= track_config
        hints_path = track_root / "data" / "ai_hints.ini"
        hints = _read_ini(hints_path)
        out["ai_hints_source"] = str(hints_path)
        out["ai_hints"] = {
            section: dict(hints[section]) for section in hints.sections()}
    except Exception:
        out["ai_hints_source"] = None
        out["ai_hints"] = {}
    return out


class Clock:
    """수집 시작(perf_counter) 기준 상대 시간을 주는 단일 시계.
    두 로거가 반드시 이 시계를 공유해야 timestamp가 같은 축 위에 놓인다."""
    def __init__(self):
        self.t0 = time.perf_counter()

    def now(self) -> float:
        "수집 시작 후 지난 시간(초)"
        return time.perf_counter() - self.t0


def _lap_crossing_goal(target_laps: int, start_progress: float) -> int:
    """Return finish-line crossings needed for N guaranteed complete laps.

    The collector can be started anywhere in the current lap. Progress alone
    cannot prove that the first observed fragment began at the finish line, so
    that fragment is always discarded. Crossing 1 opens retained lap 1 and
    crossing ``target_laps + 1`` closes the final retained lap.
    """
    del start_progress  # Kept for call-site/API compatibility.
    return target_laps + 1


def _retained_complete_laps(completed_laps: int, lap_base: int,
                            target_laps: int,
                            standing_start: bool = False) -> int:
    """Count fully closed retained laps after discarding the first fragment."""
    discarded_fragments = 0 if standing_start else 1
    return max(0, min(
        target_laps, completed_laps - lap_base - discarded_fragments))


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
            capture_backend: str = "dxcam",
            capture_window: str | None = None,
            image_format: str = "jpg",
            jpg_quality: int = 85,
            onboard_camera: int = 0,
            session_mode: str = "auto",
            target_laps: int | None = None,
            structured_export: bool = True) -> Path:
    """사람 주행 데이터 수집 메인 루프.

    telemetry_hz / image_hz : 각 스트림의 목표 기록 주파수(서로 달라도 됨).
    max_seconds             : 수집 시간 제한(테스트용). None이면 Ctrl+C 까지.
    host / port             : sensors_par ego_server 주소(기본 127.0.0.1:2345).
    capture_region          : (left, top, width, height). wgc에서는 창 내부 기준.
    capture_backend         : "dxcam"(기본) | "wgc" | "mss" | "auto"(dxcam→mss).
    capture_window          : wgc로 캡처할 창 제목 일부. 지정하면 해당 창만 기록.
    image_format            : "jpg"(빠름/작음) | "png"(무손실/느림).
    jpg_quality             : JPG 품질(기본 85). PNG에서는 사용하지 않음.
    onboard_camera          : 이 수집에 쓴 F6 온보드캠 인덱스(cameras.ini CAMERA_N). 기본 0.
        게임은 활성 온보드캠 번호를 어디에도 노출하지 않으므로(camera_manager 의 DRIVABLE_MODE
        는 F1 뷰만 추적, telemetry 에도 없음), **사용자가 선언한 값을 meta 에 박아** 뷰어/검증이
        정확한 카메라로 투영하게 한다. 실제 F6 뷰와 이 값이 어긋나면 박스가 안 맞으니 주의.
    """
    if session_mode not in {"auto", "solo", "multi"}:
        raise ValueError("session_mode must be auto, solo, or multi")
    if target_laps is not None and target_laps < 1:
        raise ValueError("target_laps must be at least 1")
    if not 1 <= jpg_quality <= 100:
        raise ValueError("jpg_quality must be between 1 and 100")
    race_cfg = _race_cfg_snapshot()
    configured_laps = int(race_cfg.get("race_laps", 0) or 0)
    is_practice = str(race_cfg.get("race_session_name", "")).strip().lower() == "practice"
    required_session_laps = (
        _lap_crossing_goal(target_laps, 0.0) if target_laps is not None else 0)
    if (target_laps is not None and not is_practice
            and configured_laps and configured_laps < required_session_laps):
        print(f"[laps] WARNING: game race.ini has {configured_laps} lap(s), "
              f"but collector needs {required_session_laps} finish-line crossings "
              f"to retain {target_laps} complete lap(s); increase the game session length")

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

    screen = ScreenCapture(region=capture_region, backend=capture_backend,
                           window_title=capture_window)
    capture_target = (f"window={screen.resolved_window_title!r}"
                      if screen.resolved_window_title else
                      f"region={capture_region or 'full monitor'}")
    print(f"[capture] backend={screen.backend} {capture_target}")
    tel_logger = TelemetryLogger(run_dir)

    clock = Clock()
    tel_period = 1.0 / telemetry_hz
    next_tel = 0.0

    # 이미지 캡처/인코딩은 별도 스레드로 분리한다. 인코딩(~17ms)이 메인 루프를 잡으면
    # telemetry 까지 50Hz 를 못 지키기 때문(단일 루프 시 39~44Hz). 스레드 분리로 메인 루프는
    # telemetry 만 → 게임이 주는 만큼(~50Hz) 유지. clock 을 공유해 timestamp 축을 맞춘다.
    img_recorder = ThreadedImageRecorder(
        run_dir, clock, screen, image_hz=image_hz, image_ext=image_format,
        jpg_quality=jpg_quality).start()
    quality_text = f", quality={jpg_quality}" if img_recorder.image_ext == "jpg" else ""
    print(f"[capture] image_format={img_recorder.image_ext}{quality_text} "
          "(스레드 분리 캡처)")
    print(f"[camera] 온보드캠 = CAMERA_{onboard_camera} (cameras.ini)  "
          f"→ 게임 F6 뷰가 이 카메라인지 확인! (다르면 --onboard-camera 로 지정)")

    # 수집 중에는 수만 개의 telemetry/image dict가 의도적으로 계속 살아 있다.
    # CPython cyclic GC의 full scan이 누적 데이터 전체를 주기적으로 훑으면 모든
    # Python 스레드가 100~400ms 멈춰 이미지와 telemetry가 동시에 비는 현상이 생긴다.
    # 일반 참조 카운팅은 계속 동작하므로 cyclic GC만 캡처 종료 시점까지 미룬다.
    gc_was_enabled = gc.isenabled()
    gc_count_before = gc.get_count()
    if gc_was_enabled:
        gc.disable()
        print("[runtime] cyclic GC = OFF during capture (restored after capture stop)")

    print(f"[collect] telemetry {telemetry_hz}Hz / image {image_hz}Hz. Ctrl+C 로 종료.")
    lap_base = None
    lap_goal_increment = target_laps
    completed_laps_reported = 0
    start_mode = None
    stationary_at_collect_start = False
    motion_candidate_time = None
    standing_start_motion_time = None
    motion_threshold_mps = 1.0 / 3.6
    motion_hold_seconds = 0.5
    try:
        while True:
            now = clock.now()

            # telemetry: 자기 주기마다 최근 state를 읽어 기록 (제어는 보내지 않음)
            if now >= next_tel:
                state = tel_src.latest_state()
                if state is not None:
                    opps = build_opponents(state, opp_src.latest()) if opp_src else None
                    tel_logger.log(now, state, opponents=opps)
                    if target_laps is not None:
                        completed = int(state.get("completedLaps", state.get("LapCount", 0)) or 0)
                        progress = float(state.get("NormalizedSplinePosition", 0.0) or 0.0)
                        speed_mps = abs(float(state.get("speed", 0.0) or 0.0))
                        if lap_base is None:
                            lap_base = completed
                            stationary_at_collect_start = speed_mps < motion_threshold_mps
                            if stationary_at_collect_start:
                                print(f"[start] stationary start candidate "
                                      f"(speed={speed_mps * 3.6:.2f}km/h); "
                                      "waiting for sustained movement")
                            else:
                                start_mode = "flying"
                                lap_goal_increment = _lap_crossing_goal(
                                    target_laps, progress)
                                print(f"[laps] target={target_laps} complete laps "
                                      f"(start completedLaps={lap_base}, "
                                      f"progress={progress:.3f}; discarding first "
                                      "observed lap fragment)")
                        if stationary_at_collect_start and start_mode is None:
                            if speed_mps >= motion_threshold_mps:
                                if motion_candidate_time is None:
                                    motion_candidate_time = now
                                elif now - motion_candidate_time >= motion_hold_seconds:
                                    start_mode = "standing"
                                    standing_start_motion_time = motion_candidate_time
                                    lap_goal_increment = target_laps
                                    print(f"[start] standing start detected at "
                                          f"{standing_start_motion_time:.3f}s "
                                          "(keeping exactly one preceding frame)")
                                    print(f"[laps] target={target_laps} standing-start "
                                          "laps")
                            else:
                                motion_candidate_time = None
                        retained_complete = _retained_complete_laps(
                            completed, lap_base, target_laps,
                            standing_start=(start_mode == "standing"))
                        while completed_laps_reported < retained_complete:
                            completed_laps_reported += 1
                            print(f"[laps] {completed_laps_reported}/{target_laps} "
                                  "lap complete")
                        if completed - lap_base >= int(lap_goal_increment):
                            print(f"[laps] target reached: {target_laps} complete laps")
                            break
                next_tel += tel_period

            if max_seconds is not None and now >= max_seconds:
                break

            # 다음 telemetry tick 까지 살짝 쉬기 (busy-wait 방지). 이미지는 별도 스레드가 처리.
            sleep = next_tel - clock.now()
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\n[stop] 사용자 종료")
    finally:
        elapsed = clock.now()
        # 먼저 캡처를 멈춘다. telemetry parquet 저장 중에도 이미지 스레드가 계속
        # 실행되면 telemetry가 없는 꼬리 프레임이 생긴다.
        try:
            img_summary = img_recorder.stop()
        finally:
            gc_collect_objects = None
            gc_collect_ms = None
            if gc_was_enabled:
                gc.enable()
                gc_t0 = time.perf_counter()
                gc_collect_objects = gc.collect()
                gc_collect_ms = (time.perf_counter() - gc_t0) * 1000.0
                print(f"[runtime] cyclic GC = ON; post-capture collect "
                      f"{gc_collect_objects} objects / {gc_collect_ms:.1f}ms")
        tel_summary = tel_logger.flush()
        # 실제 달성 Hz = 기록된 개수 / 총 수집시간
        tel_hz_actual = tel_summary.get("telemetry_rows", 0) / elapsed if elapsed > 0 else 0
        img_hz_actual = img_summary.get("image_frames", 0) / elapsed if elapsed > 0 else 0
        opp_hz_actual = (opp_src.received / elapsed) if (opp_src and elapsed > 0) else 0
        print(f"[rate] 실측  telemetry {tel_hz_actual:.1f}Hz / image {img_hz_actual:.1f}Hz "
              f"/ opponents {opp_hz_actual:.1f}Hz  (목표 {telemetry_hz}/{image_hz}Hz, {elapsed:.1f}s)")
        meta = {
            "run_dir": str(run_dir),
            "source": "human_driving",
            "onboard_camera_index": onboard_camera,
            "onboard_camera_name": (f"CAMERA_{onboard_camera}"
                                    if onboard_camera >= 0 else "DASH"),
            "session_mode": session_mode,
            "target_laps": target_laps,
            "start_mode": start_mode,
            "stationary_at_collect_start": stationary_at_collect_start,
            "standing_start_motion_time_s": standing_start_motion_time,
            "standing_start_motion_threshold_kmh": motion_threshold_mps * 3.6,
            "standing_start_motion_hold_seconds": motion_hold_seconds,
            **_camera_cfg_snapshot(),
            **race_cfg,
            "telemetry_hz": telemetry_hz,
            "image_hz": image_hz,
            "telemetry_hz_actual": round(tel_hz_actual, 2),
            "image_hz_actual": round(img_hz_actual, 2),
            "elapsed_seconds": round(elapsed, 2),
            "image_format": img_recorder.image_ext,
            "jpg_quality": jpg_quality if img_recorder.image_ext == "jpg" else None,
            "cyclic_gc_disabled_during_capture": gc_was_enabled,
            "gc_count_before_disable": list(gc_count_before),
            "gc_collect_after_capture_objects": gc_collect_objects,
            "gc_collect_after_capture_ms": (
                round(gc_collect_ms, 2) if gc_collect_ms is not None else None),
            "ego_server": f"{host}:{port}",
            "opponents_server": f"{host}:{opp_port}" if opp_src else None,
            "telemetry_recv_total": tel_src.received,
            "opponents_recv_total": opp_src.received if opp_src else 0,
            "capture_backend": screen.backend,
            "capture_region": capture_region,
            "capture_window": screen.resolved_window_title,
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

    # 최종 학습용 TXT는 dataset_v2에만 생성한다. 예전
    # dataset/<run_id>_<track>_<hz>hz flat export는 중복하므로 만들지 않는다.
    if structured_export:
        try:
            from .structured_export import export_structured
            export_structured(run_dir, mode=session_mode)
        except Exception as e:
            print(f"[structured-export] 내보내기 스킵: {e}")

    # 검증(ac-validate)은 수집 파이프라인과 분리한다. 데이터가 저장된 뒤 필요할 때 직접 돌린다:
    #   ac-validate dataset/<run_id>
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
    ap.add_argument("--capture-backend", default="dxcam",
                    choices=["auto", "wgc", "dxcam", "mss"],
                    help="화면 캡처 백엔드(기본: dxcam 모니터 캡처)")
    ap.add_argument("--capture-window", default=None,
                    help="다른 앱에 가려져도 이 창만 캡처(wgc 창 제목 일부)")
    ap.add_argument("--capture-region", default=None,
                    help="left,top,width,height (예: 0,0,640,480). 생략 시 주 모니터 전체")
    ap.add_argument("--image-format", default="jpg", choices=["jpg", "png"],
                    help="이미지 저장 포맷: jpg(빠름/작음, 기본) | png(무손실/느림)")
    ap.add_argument("--jpg-quality", type=int, default=85,
                    help="JPG 저장 품질 1~100(기본: 85, PNG에는 영향 없음)")
    ap.add_argument("--onboard-camera", type=int, default=0,
                    help="이 수집에 쓴 F6 온보드캠 인덱스(cameras.ini CAMERA_N, 기본 0). "
                         "게임이 활성 온보드캠을 노출 안 하므로 사용자가 지정해 meta 에 기록한다.")
    ap.add_argument("--session-mode", choices=["auto", "solo", "multi"], default="auto",
                    help="dataset_v2 세션 종류(기본: 유효 상대차 데이터로 자동 판별)")
    ap.add_argument("--laps", type=int, default=None,
                    help="완전한 랩 N개를 수집한 뒤 자동 종료(예: --laps 5)")
    ap.add_argument("--no-structured-export", action="store_true",
                    help="수집 종료 후 dataset_v2 자동 생성을 건너뜀")
    args = ap.parse_args()

    region = (tuple(int(x) for x in args.capture_region.split(","))
              if args.capture_region else None)

    collect(out_dir=args.out,
            telemetry_hz=args.telemetry_hz, image_hz=args.image_hz,
            max_seconds=args.max_seconds, host=args.host, port=args.port,
            opp_port=args.opp_port,
            capture_region=region, capture_backend=args.capture_backend,
            capture_window=args.capture_window,
            image_format=args.image_format, jpg_quality=args.jpg_quality,
            onboard_camera=args.onboard_camera,
            session_mode=args.session_mode, target_laps=args.laps,
            structured_export=not args.no_structured_export)


if __name__ == "__main__":
    main()
