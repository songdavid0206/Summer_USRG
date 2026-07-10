"""export_frames - run 을 프레임별 txt 데이터셋으로 내보낸다 (조교 예시 포맷)
================================================================================

parquet 대신 **프레임 하나당 파일 하나**로 푸는 포맷. 파일명은 그 프레임의 시각(ms).

    <out>/<run_name>/
        CAMERA/<ms>.jpg          # 이미지
        EGO_INFO/<ms>.txt        # 자차 상태 (key: values)
        OBJECT_INFO/<ms>.txt     # 그 프레임 카메라에 '보이는' 상대차만 (헤더 + 차당 1행)

파일명 ms = 실제 캡처 timestamp(t_img)를 반올림한 값. 목표 Hz 의 균일 격자가 아니라
**실측 시각**이므로 간격이 20, 20, 21, 62... 처럼 흔들릴 수 있다(루프 지터가 그대로 보인다).

가시성 판정
-----------
AC 는 "이 차가 화면에 보이는가"를 알려주지 않는다(플러그인은 트랙의 모든 차를 항상 보낸다).
그래서 기하로 계산한다: ego 위치·yaw 기준으로 상대차가 **앞쪽(forward>0)** 에 있고
**|방위각| <= 수평FOV/2** 이면 프레임 안으로 본다.
- 수평FOV 는 게임 설정 `cfg/camera_onboard.ini` 의 FOV(수직) + 렌더 종횡비에서 유도.
- **차폐(occlusion)는 처리하지 않는다.** 앞차 뒤에 완전히 가려진 차도 '보임'으로 들어간다.
- 카메라가 차 원점에 있다고 근사한다(실제로는 약간의 오프셋. 먼 거리에선 무시 가능).

좌표/단위 규약 (예시 데이터셋에 맞춤)
------------------------------------
- position        : 월드 (x, y, z). z = 높이.
- orientation     : 쿼터니언 (x, y, z, w). ZYX(yaw-pitch-roll) 순.
- velocity        : **차체(body) 기준** (전진, 좌우, 상하) m/s.
- angular_velocity: (roll rate, pitch rate, yaw rate) **deg/s**.
- acceleration    : (종, 횡, 수직) m/s^2.
- steer_angle     : **[-1, 1] 정규화된 조작 커맨드** (= telemetry 의 steerNorm).
  주의: telemetry 의 `steer`(도) 는 AC 조향필터를 거친 '결과 각도'라 단일 상수로 환산되지
  않는다(소신호 이득 ~178.6, 큰 입력에서 ~173 으로 포화, 커맨드가 결과보다 앞섬).

원본 필드의 축 순서가 제각각이라 여기서 한 번에 정리한다(플러그인 structures.py 기준):
- local_velocity 는 재배열됨   -> (x=전진, y=좌우, z=상하)
- angular_velocity 는 재배열 안 됨 -> raw (x=pitch축, y=up=yaw축, z=forward=roll축)
- accelX 는 사실 **횡가속도**, accelY 가 **종가속도** (이름이 뒤바뀌어 있음). 단위는 G.
"""

from __future__ import annotations

import argparse
import configparser
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

G = 9.80665

# 차종 치수는 data.acd(colliders.ini)에서 직접 읽는다(acd.py). 읽기 실패 시에만 이 기본값.
DEFAULT_DIMENSION = (4.500, 1.800, 1.000)


def _load_dimensions(models: dict[int, str]) -> dict[int, tuple[float, float, float]]:
    """car_id -> (length, width, height). data.acd 에서 차종별로 읽고 캐시."""
    from .acd import read_car_specs_by_model
    cache: dict[str, tuple[float, float, float] | None] = {}
    out: dict[int, tuple[float, float, float]] = {}
    for cid, model in models.items():
        if model not in cache:
            specs = read_car_specs_by_model(model)
            cache[model] = tuple(specs["dimension"]) if (specs and specs.get("dimension")) else None
        out[cid] = cache[model] or DEFAULT_DIMENSION
    return out


def _load_steer_lock(models: dict[int, str]) -> float | None:
    """자차(car_id 0) STEER_LOCK(도)을 data.acd 에서. 없으면 None."""
    from .acd import read_car_specs_by_model
    model = models.get(0)
    if not model:
        return None
    specs = read_car_specs_by_model(model)
    return specs["steer_lock"] if specs else None


# --------------------------------------------------------------------------- 게임 설정
def _ac_cfg_dir() -> Path:
    return Path(os.path.expanduser("~")) / "Documents" / "Assetto Corsa" / "cfg"


def _read_ini(path: Path) -> configparser.ConfigParser:
    cp = configparser.ConfigParser(strict=False)
    cp.read(path, encoding="utf-8-sig")
    return cp


def read_camera_hfov() -> float:
    """수평 FOV(라디안). AC 의 FOV 설정은 '수직'이라 렌더 종횡비로 변환한다."""
    cfg = _ac_cfg_dir()
    vfov_deg = 56.0
    w, h = 1920.0, 1080.0
    try:
        vfov_deg = float(_read_ini(cfg / "camera_onboard.ini")["HEADER"]["FOV"])
    except Exception:
        pass
    try:
        vid = _read_ini(cfg / "video.ini")["VIDEO"]
        w, h = float(vid["WIDTH"]), float(vid["HEIGHT"])
    except Exception:
        pass
    vfov = math.radians(vfov_deg)
    return 2.0 * math.atan(math.tan(vfov / 2.0) * (w / h))


def read_car_models() -> dict[int, str]:
    """race.ini 에서 car_id -> 차종. [RACE] MODEL 이 자차(0), [CAR_n] 이 상대차."""
    out: dict[int, str] = {}
    try:
        cp = _read_ini(_ac_cfg_dir() / "race.ini")
    except Exception:
        return out
    if cp.has_option("RACE", "MODEL"):
        out[0] = cp.get("RACE", "MODEL")
    for sec in cp.sections():
        if sec.startswith("CAR_") and cp.has_option(sec, "MODEL"):
            model = cp.get(sec, "MODEL")
            try:
                cid = int(sec.split("_", 1)[1])
            except ValueError:
                continue
            if model and model != "-":
                out[cid] = model
    return out


# --------------------------------------------------------------------------- 수학
def quat_from_ypr(yaw: float, pitch: float, roll: float) -> tuple[float, float, float, float]:
    """ZYX(yaw-pitch-roll) 오일러각 -> 쿼터니언 (x, y, z, w)."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def _fmt(*vals: float) -> str:
    return " ".join("%.10f" % v for v in vals)


def _link_or_copy(src: Path, dst: Path) -> None:
    """이미지는 수 GB 라 하드링크로 잇는다(같은 볼륨이면 추가 용량 0). 안 되면 복사."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


# --------------------------------------------------------------------------- 본체
def export_run(run_dir: str | Path, out_root: str | Path | None = None,
               hfov: float | None = None, max_range: float | None = None,
               only_valid: bool = True) -> Path:
    """run 하나를 CAMERA/EGO_INFO/OBJECT_INFO 구조로 내보낸다.

    hfov      : 수평 화각(도). None 이면 게임 설정에서 읽는다.
    max_range : 이 거리(m)를 넘는 상대차는 제외. None 이면 거리 제한 없음.
    only_valid: 자차가 핏에 있는 프레임(ego_features.valid=False)은 건너뛴다.
    """
    run_dir = Path(run_dir)
    proc = run_dir / "processed"
    ego = pd.read_parquet(proc / "ego_features.parquet")
    ego_raw = pd.read_parquet(run_dir / "ego_state" / "ego_state.parquet")
    cam = pd.read_parquet(run_dir / "camera" / "camera_index.parquet")

    hfov_rad = math.radians(hfov) if hfov is not None else read_camera_hfov()
    half_fov = hfov_rad / 2.0
    models = read_car_models()
    dims = _load_dimensions(models)          # car_id -> (L, W, H), data.acd 에서
    steer_lock = _load_steer_lock(models)    # 도. steerNorm*steer_lock = 조향각(도)

    # steer 라벨: 정규화된 커맨드(steerNorm). 없으면(구버전 플러그인) 도 단위 그대로 두고 경고.
    if "steerNorm" in ego_raw.columns:
        steer_label = ego_raw["steerNorm"].to_numpy()
        steer_note = "steerNorm (normalized command, [-1,1])"
    else:
        steer_label = ego_raw["steer"].to_numpy()
        steer_note = "steer (DEGREES - not normalized; 구버전 플러그인)"
        print("[export] 경고: steerNorm 없음 -> steer_angle 이 도(degree) 단위로 나갑니다.")

    opps: dict[int, pd.DataFrame] = {}
    for pf in sorted((proc / "opponent").glob("opp_*_features.parquet")):
        oid = int(pf.stem.split("_")[1])
        opps[oid] = pd.read_parquet(pf)

    # 출력 폴더 이름: <run_id>_<track>_<hz>hz
    track = "unknown"
    tj = run_dir / "track" / "track.json"
    if tj.exists():
        track = json.loads(tj.read_text(encoding="utf-8")).get("track_name", "unknown")
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    hz = round(meta.get("image_hz_actual", 0))
    name = "%s_%s_%dhz" % (run_dir.name, track.replace("/", "-"), hz)

    # 기본 출력 위치는 run 폴더와 나란히 (dataset/<run_id>_<track>_<hz>hz).
    # run 폴더(dataset/<run_id>)와 이름이 겹치지 않는다 — 뒤에 트랙/Hz가 붙기 때문.
    out_root = Path(out_root) if out_root else run_dir.parent
    out = out_root / name
    for sub in ("CAMERA", "EGO_INFO", "OBJECT_INFO"):
        d = out / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    obj_header = ("position.x, position.y, position.z, "
                  "orientation.x, orientation.y, orientation.z, orientation.w, "
                  "dimension.length, dimension.width, dimension.height, "
                  "velocity.x, velocity.y, velocity.z, track_id")

    t_ms = np.rint(cam["t_img"].to_numpy() * 1000.0).astype(np.int64)
    if len(np.unique(t_ms)) != len(t_ms):
        print("[export] 경고: ms 반올림 후 중복 파일명 발생 -> 뒤 프레임이 덮어씁니다.")

    ego_valid = ego["valid"].to_numpy() if "valid" in ego.columns else np.ones(len(ego), bool)
    n_written = 0
    n_objects = 0
    for i in range(len(cam)):
        if only_valid and not ego_valid[i]:
            continue
        ms = int(t_ms[i])
        e = ego_raw.iloc[i]

        # CAMERA: 원본 프레임을 ms 이름으로 복사
        src = run_dir / cam["image_path"].iloc[i]
        if src.exists():
            _link_or_copy(src, out / "CAMERA" / ("%d%s" % (ms, src.suffix)))

        # EGO_INFO
        qx, qy, qz, qw = quat_from_ypr(float(e["yaw"]), float(e["pitch"]), float(e["roll"]))
        lines = [
            "position: " + _fmt(e["world_position_x"], e["world_position_y"], e["world_position_z"]),
            "orientation: " + _fmt(qx, qy, qz, qw),
            "velocity: " + _fmt(e["local_velocity_x"], e["local_velocity_y"], e["local_velocity_z"]),
            # 각속도는 raw 축 순서 -> (forward=roll, lateral=pitch, up=yaw) 로 재배열 + deg/s
            "angular_velocity: " + _fmt(math.degrees(e["angular_velocity_z"]),
                                        math.degrees(e["angular_velocity_x"]),
                                        math.degrees(e["angular_velocity_y"])),
            # accelX=횡, accelY=종 (이름이 뒤바뀜). G -> m/s^2. 수직 성분은 AC 가 안 준다.
            "acceleration: " + _fmt(e["accelY"] * G, e["accelX"] * G, 0.0),
            "throttle: " + _fmt(e["throttle"]),
            "brake: " + _fmt(e["brake"]),
            "steer_angle: " + _fmt(steer_label[i]),
        ]
        (out / "EGO_INFO" / ("%d.txt" % ms)).write_text("\n".join(lines) + "\n", encoding="utf-8")

        # OBJECT_INFO: 화각 안에 들어오는 상대차만
        rows = []
        for oid, o in opps.items():
            r = o.iloc[i]
            if bool(r.get("retired", False)) or not bool(r.get("present", True)):
                continue
            fwd, lat = float(r["opp_forward"]), float(r["opp_lateral"])
            if fwd <= 0:
                continue
            if abs(math.atan2(lat, fwd)) > half_fov:
                continue
            if max_range is not None and float(r["distance"]) > max_range:
                continue

            # 차체 방향: yaw_body(접지점 기반) 우선, 없으면 속도벡터 방향(정지 시 부정확)
            oyaw = float(r["yaw_body"]) if "yaw_body" in o.columns and pd.notna(r["yaw_body"]) \
                else float(r["opp_yaw"])
            ox, oy, oz, ow = quat_from_ypr(oyaw, 0.0, 0.0)
            L, W, H = dims.get(oid, DEFAULT_DIMENSION)

            # 월드 속도 -> 그 차의 body frame (전진, 좌우, 상하)
            vx, vy = float(r["opp_vx"]), float(r["opp_vy"])
            vz = float(r["opp_vz"]) if "opp_vz" in o.columns and pd.notna(r["opp_vz"]) else 0.0
            c, s = math.cos(oyaw), math.sin(oyaw)
            v_fwd, v_lat = vx * c + vy * s, -vx * s + vy * c

            opp_z = float(r["opp_z"]) if "opp_z" in o.columns and pd.notna(r["opp_z"]) else 0.0
            rows.append("%.10f, %.10f, %.10f, %.12f, %.12f, %.12f, %.12f, "
                        "%.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %d"
                        % (float(r["opp_x"]), float(r["opp_y"]), opp_z,
                           ox, oy, oz, ow, L, W, H, v_fwd, v_lat, vz, oid))
        n_objects += len(rows)
        (out / "OBJECT_INFO" / ("%d.txt" % ms)).write_text(
            obj_header + "\n" + "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        n_written += 1

    info = {
        "run_id": run_dir.name,
        "track": track,
        "frames": n_written,
        "frames_skipped_ego_invalid": int(len(cam) - n_written) if only_valid else 0,
        "objects_total": n_objects,
        "objects_per_frame": round(n_objects / n_written, 3) if n_written else 0,
        "image_hz_actual": meta.get("image_hz_actual"),
        "hfov_deg": round(math.degrees(hfov_rad), 3),
        "max_range_m": max_range,
        "steer_angle_source": steer_note,
        "steer_lock_deg": steer_lock,
        "car_models": {str(k): v for k, v in models.items()},
        "dimensions": {str(k): list(v) for k, v in dims.items()},
        "dimension_source": "data.acd colliders.ini (bounding box)",
        "note": ("파일명 ms = 실제 t_img 반올림(균일 격자 아님). OBJECT_INFO 는 화각 안 상대차만 "
                 "(차폐 미처리). dimension 은 data.acd(colliders.ini) 박스 합집합 = 게임 실제 값."),
    }
    (out / "export_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[export] %s" % out)
    print("[export] %d 프레임 (핏 등 제외 %d) | 화각 %.1f도 | 프레임당 평균 %.2f대"
          % (n_written, len(cam) - n_written, math.degrees(hfov_rad), info["objects_per_frame"]))
    return out


def main():
    ap = argparse.ArgumentParser(description="run -> CAMERA/EGO_INFO/OBJECT_INFO (프레임별 txt)")
    ap.add_argument("run_dir", help="dataset/<run_id>")
    ap.add_argument("--out", default=None, help="출력 루트 (기본: run 폴더와 같은 자리 = dataset/)")
    ap.add_argument("--hfov", type=float, default=None,
                    help="수평 화각(도). 생략 시 게임 설정에서 읽음")
    ap.add_argument("--max-range", type=float, default=None,
                    help="이 거리(m)보다 먼 상대차 제외")
    ap.add_argument("--all-frames", action="store_true",
                    help="자차가 핏에 있는 프레임도 포함")
    args = ap.parse_args()
    export_run(args.run_dir, args.out, hfov=args.hfov, max_range=args.max_range,
               only_valid=not args.all_frames)


if __name__ == "__main__":
    main()
