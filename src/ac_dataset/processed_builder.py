"""
processed_builder - raw 데이터 → E2E/IL 학습용 후처리 피처
==========================================================

raw(camera/ego_state/opponent/track)는 그대로 두고, `processed/` 폴더에 선별·가공된
피처를 만든다. E2E 자율주행/레이싱 IL에서 흔히 쓰는 입력 설계를 따른다:
  - ego: 타이어/서스펜션 같은 동역학 디테일은 빼고 운동상태+자세+위치+랩+라벨만
  - opponent: 절대좌표 + ego-frame(앞/옆) 상대좌표 + (시간차분) 속도/상대속도
  - track: 레이싱 라인 대비 횡오프셋 / heading error / 곡률 / 좌우 경계거리

좌표계 메모: ego world_position 수평면은 (x, y). 자차 진행방향 = yaw (위치평면 heading과
평균 0.8도로 일치 확인). 그래서 ego-frame 회전은 (cos yaw, sin yaw)를 forward로 사용.

유효 프레임 마스크
------------------
AC는 리타이어하거나 세션이 끝난 차를 핏 박스로 '순간이동'시킨 뒤 그 자리에 고정한다.
플러그인은 그 뒤로도 그 차를 계속 보내므로 present=True가 유지된다 → 카메라에는 보이지도
않는 차가 "내 뒤 20m"에 있는 것처럼 피처에 남아 학습을 오염시킨다. 자차도 세션 종료 시
같은 방식으로 개러지에 되돌아간다.
행을 지우지는 않는다(row i ↔ camera/{i:06d}.jpg 1:1 대응이 데이터셋 전체의 전제).
대신 bool 마스크 컬럼을 붙여 학습 시 걸러 쓴다:
  - ego_features.in_pit / .valid          (자차가 핏에 있는 프레임 제외)
  - opp_<id>_features.retired / .valid    (그 상대차가 이미 핏 박스에 있는 프레임 제외)
요약은 processed/quality.json 에 기록.

출력: <run>/processed/ego_features.{parquet,csv}, opponent/opp_<id>_features.{...},
      track_features.{...}, quality.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# E2E 핵심 ego 피처(동역학 디테일 제외). 없는 컬럼은 자동 스킵.
EGO_KEEP = [
    # 운동
    "speed", "velocity_x", "velocity_y", "velocity_z",
    "local_velocity_x", "local_velocity_y", "local_velocity_z",
    "accelX", "accelY",
    "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
    # 자세
    "yaw", "pitch", "roll",
    # 위치/트랙진행
    "world_position_x", "world_position_y", "world_position_z",
    "NormalizedSplinePosition",
    # 엔진/랩 상태
    "RPM", "actualGear", "LapCount", "numberOfTyresOut", "LapInvalidated", "isInPit",
    # 정답 라벨
    "steer", "throttle", "brake",
]


def _save(df: pd.DataFrame, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(base.with_suffix(".parquet"), index=False)
    df.to_csv(base.with_suffix(".csv"), index=False)


def _wrap(a):
    """각도를 [-pi, pi]로."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# 리타이어 = 핏 박스로 순간이동 후 영구 고정. 그 두 조건으로 시점을 찾는다.
# 45Hz에서 300km/h면 한 프레임에 1.9m, 상대차 위치가 1프레임 held되어도 4m 남짓.
# 10m를 넘는 이동은 물리적으로 불가능 → 순간이동.
TELEPORT_M = 10.0
STATIC_M = 1.0      # 순간이동 뒤 이 반경 안에 머물면 '고정됨'
STATIC_KMH = 1.0    # 그리고 사실상 정지
STATIC_CONFIRM_FRAMES = 20  # 40Hz 기준 0.5초. 이후 재이동/다단계 피트 이동 허용


def _detect_retirement(x: np.ndarray, y: np.ndarray, speed_kmh: np.ndarray) -> int | None:
    """핏 박스에 고정된 첫 프레임 인덱스. 리타이어가 없으면 None.

    트랙 위에 그냥 멈춰 선 차(사고 등)는 카메라에 실제로 보이므로 제외해선 안 된다.
    그래서 '정지'만으로는 판정하지 않고 반드시 순간이동을 동반해야 한다.
    """
    if len(x) < 2:
        return None
    step = np.hypot(np.diff(x), np.diff(y))
    for k in np.where(step > TELEPORT_M)[0] + 1:   # k = 점프해서 도착한 프레임
        # CSP는 리타이어 차량을 한 번 피트로 보낸 뒤 세션 종료 때 다른 슬롯으로
        # 다시 옮길 수 있다. 따라서 "run 끝까지 같은 좌표" 대신, 도착 직후 충분한
        # 정지 구간을 확인한다. 짧은 좌표 글리치는 곧 재이동하므로 통과하지 못한다.
        end = min(len(x), k + STATIC_CONFIRM_FRAMES)
        if end - k < STATIC_CONFIRM_FRAMES:
            continue
        if (np.hypot(x[k:end] - x[k], y[k:end] - y[k]).max() < STATIC_M
                and np.nanmedian(speed_kmh[k:end]) < STATIC_KMH):
            return int(k)
    return None


def _build_ego(ego: pd.DataFrame) -> pd.DataFrame:
    cols = ["frame", "t_img"] + [c for c in EGO_KEEP if c in ego.columns]
    out = ego[cols].copy()
    if "angular_velocity_y" in ego.columns:
        out["yaw_rate"] = ego["angular_velocity_y"]   # 요레이트(직관적 별칭)
    # 자차가 핏(개러지)에 있는 프레임 = 주행 데이터가 아님. 세션 종료 시 순간이동으로 들어간다.
    in_pit = (ego["isInPit"].to_numpy() > 0) if "isInPit" in ego.columns \
        else np.zeros(len(ego), dtype=bool)
    out["in_pit"] = in_pit
    out["valid"] = ~in_pit
    return out


def _build_opponents(run_dir: Path, ego: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """상대차마다 프레임 정렬된 피처 DataFrame을 만들어 {opp_id: df}로 반환."""
    opp_dirs = sorted((run_dir / "opponent").glob("opp_*"))
    if not opp_dirs:
        return {}
    ref = ego[["frame", "t_img", "world_position_x", "world_position_y",
               "world_position_z", "yaw", "speed"]].rename(
        columns={"world_position_x": "ego_x", "world_position_y": "ego_y",
                 "world_position_z": "ego_z", "speed": "ego_speed"})
    result: dict[int, pd.DataFrame] = {}
    for d in opp_dirs:
        oid = int(d.name.split("_")[1])
        pf = list(d.glob("*.parquet"))
        if not pf:
            continue
        o = pd.read_parquet(pf[0]).merge(ref, on=["frame", "t_img"], how="left")
        o = o.sort_values("frame").reset_index(drop=True)
        c, s = np.cos(o["yaw"]), np.sin(o["yaw"])
        # 절대좌표
        o["opp_x"] = o["ego_x"] + o["rel_x"]
        o["opp_y"] = o["ego_y"] + o["rel_y"]
        if "rel_z" in o.columns:
            o["opp_z"] = o["ego_z"] + o["rel_z"]
        # ego-frame (앞/옆). forward=+진행방향, lateral=+왼쪽
        o["opp_forward"] = o["rel_x"] * c + o["rel_y"] * s
        o["opp_lateral"] = -o["rel_x"] * s + o["rel_y"] * c
        # 속도. 위치 시간차분은 절대 쓰지 않는다(held 프레임에서 0↔폭주로 튄다).
        if "opp_vx" in o.columns and o["opp_vx"].notna().any():
            # 신형 플러그인: Opponent.velocity 실측 월드 속도벡터를 그대로 쓴다.
            o["opp_speed"] = np.hypot(o["opp_vx"], o["opp_vy"])
        else:
            # 구형 플러그인 폴백: 실측 속력 + 진행방향(opp_yaw)으로 재구성.
            # opp_yaw 는 속도벡터 방향이라 정지 시 무의미(관측: 정지 프레임 91%가 정확히 +45°).
            opp_spd = (o["opp_speedKMH"] / 3.6).to_numpy()
            oyaw = o["opp_yaw"].to_numpy()
            o["opp_speed"] = opp_spd
            o["opp_vx"] = opp_spd * np.cos(oyaw)
            o["opp_vy"] = opp_spd * np.sin(oyaw)
        # 자차 속도벡터도 speed+yaw로 (위치평면 heading=yaw 확인됨)
        ego_spd, eyaw = o["ego_speed"].to_numpy(), o["yaw"].to_numpy()
        ego_vx, ego_vy = ego_spd * np.cos(eyaw), ego_spd * np.sin(eyaw)
        o["rel_vx"] = o["opp_vx"] - ego_vx
        o["rel_vy"] = o["opp_vy"] - ego_vy
        # 접근속도(+면 가까워짐) = -d(distance)/dt = -(rel_pos·rel_vel)/distance
        with np.errstate(all="ignore"):
            o["closing_speed"] = -((o["rel_x"] * o["rel_vx"]
                                    + o["rel_y"] * o["rel_vy"]) / o["distance"])
        o["opp_id"] = oid
        # 리타이어 이후 프레임은 값은 그대로 두되(raw 보존) 마스크로 표시만 한다.
        ret_k = _detect_retirement(o["opp_x"].to_numpy(), o["opp_y"].to_numpy(),
                                   o["opp_speedKMH"].to_numpy())
        retired = np.zeros(len(o), dtype=bool)
        if ret_k is not None:
            retired[ret_k:] = True
        o["retired"] = retired
        o["valid"] = o["present"].to_numpy().astype(bool) & ~retired
        keep = ["frame", "t_img", "opp_id", "present", "retired", "valid",
                "opp_x", "opp_y", "opp_z", "opp_forward", "opp_lateral", "distance",
                "opp_vx", "opp_vy", "opp_vz", "opp_speed",
                "rel_vx", "rel_vy", "closing_speed",
                # yaw_body = 앞바퀴중점-뒷바퀴중점 벡터. 정지·스핀에도 유효한 '차체' 방향.
                # opp_yaw 는 속도벡터 방향이라 정지 시 못 쓴다.
                "opp_yaw", "yaw_body", "wheelbase", "track_width", "opp_speedKMH"]
        result[oid] = o[[c for c in keep if c in o.columns]].reset_index(drop=True)
    return result


def _nearest_indices(px, py, lx, ly):
    """각 (px,py)에 대해 라인 (lx,ly)의 최근접 점 인덱스. scipy 있으면 KDTree, 없으면 numpy."""
    pts = np.column_stack([px, py])
    line = np.column_stack([lx, ly])
    try:
        from scipy.spatial import cKDTree
        return cKDTree(line).query(pts)[1]
    except Exception:
        # numpy 폴백(청크로 메모리 절약)
        idx = np.empty(len(pts), dtype=int)
        for a in range(0, len(pts), 512):
            b = min(a + 512, len(pts))
            d = ((pts[a:b, None, :] - line[None, :, :]) ** 2).sum(-1)
            idx[a:b] = d.argmin(1)
        return idx


def _build_track(run_dir: Path, ego: pd.DataFrame) -> pd.DataFrame | None:
    line_path = run_dir / "track" / "racing_line.parquet"
    if not line_path.exists():
        return None
    line = pd.read_parquet(line_path)
    lx, ly = line["x"].to_numpy(), line["y"].to_numpy()
    # 라인 접선 방향(heading)
    th = np.arctan2(np.gradient(ly), np.gradient(lx))
    radius = line["radius"].to_numpy() if "radius" in line else np.full(len(lx), np.nan)
    sL = line["side_left"].to_numpy() if "side_left" in line else np.full(len(lx), np.nan)
    sR = line["side_right"].to_numpy() if "side_right" in line else np.full(len(lx), np.nan)

    px = ego["world_position_x"].to_numpy()
    py = ego["world_position_y"].to_numpy()
    yaw = ego["yaw"].to_numpy()
    idx = _nearest_indices(px, py, lx, ly)

    # 라인 왼쪽 법선으로 부호있는 횡오프셋 (+ = 라인 왼쪽)
    lnx, lny = -np.sin(th[idx]), np.cos(th[idx])
    lateral_offset = (px - lx[idx]) * lnx + (py - ly[idx]) * lny
    heading_error = _wrap(yaw - th[idx])
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = 1.0 / radius[idx]
    # 경계까지 거리(왼쪽으로 lateral_offset만큼 갔으면 왼쪽 엣지가 가까워짐)
    dist_left = sL[idx] - lateral_offset
    dist_right = sR[idx] + lateral_offset

    return pd.DataFrame({
        "frame": ego["frame"].to_numpy(),
        "t_img": ego["t_img"].to_numpy(),
        "lateral_offset": lateral_offset,
        "heading_error": heading_error,
        "curvature": curvature,
        "dist_left_edge": dist_left,
        "dist_right_edge": dist_right,
    })


def build_processed(run_dir: str | Path) -> dict:
    """raw → processed/ 후처리 피처 생성. 반환: 요약 dict."""
    run_dir = Path(run_dir)
    ego = pd.read_parquet(run_dir / "ego_state" / "ego_state.parquet")
    out = run_dir / "processed"

    ego_f = _build_ego(ego)
    _save(ego_f, out / "ego_features")

    # 상대차: opp별 파일로 저장(프레임 정렬). processed/opponent/opp_<id>_features
    opp_dict = _build_opponents(run_dir, ego)
    for oid, df in opp_dict.items():
        _save(df, out / "opponent" / f"opp_{oid}_features")

    trk_f = _build_track(run_dir, ego)
    if trk_f is not None:
        _save(trk_f, out / "track_features")

    quality = _build_quality(ego_f, opp_dict)
    (out / "quality.json").write_text(
        json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "ego_rows": len(ego_f), "ego_cols": len(ego_f.columns),
        "n_opponents": len(opp_dict),
        "opponent_ids": sorted(opp_dict),
        "track_features": trk_f is not None,
        "quality": quality,
    }
    print(f"[processed] ego {summary['ego_cols']}열×{summary['ego_rows']} / "
          f"opponent {summary['n_opponents']}대(opp별 파일) / "
          f"track {'O' if trk_f is not None else 'X'} -> {out}")
    eq = quality["ego"]
    print(f"[quality] 유효 프레임 {eq['valid_frames']}/{quality['n_frames']} "
          f"(핏 {eq['in_pit_frames']} 제외)")
    for oid in sorted(opp_dict):
        q = quality["opponents"][str(oid)]
        if q["retired_frame"] is None:
            print(f"[quality]   opp_{oid}: 리타이어 없음, 유효 {q['valid_frames']}프레임")
        else:
            print(f"[quality]   opp_{oid}: t={q['retired_t']:.1f}s 리타이어"
                  f"(순간이동 {q['teleport_m']:.0f}m) → 이후 {q['retired_frames']}프레임 제외, "
                  f"유효 {q['valid_frames']}")
    return summary


def _build_quality(ego_f: pd.DataFrame, opp_dict: dict[int, pd.DataFrame]) -> dict:
    """유효 프레임 마스크 요약. 학습 스크립트가 이 파일만 읽어도 상황 파악되게."""
    n = len(ego_f)
    in_pit = ego_f["in_pit"].to_numpy()
    first_pit = int(np.argmax(in_pit)) if in_pit.any() else None
    ego_q = {
        "in_pit_frames": int(in_pit.sum()),
        "first_in_pit_frame": first_pit,
        "first_in_pit_t": float(ego_f["t_img"].iloc[first_pit]) if first_pit is not None else None,
        "valid_frames": int((~in_pit).sum()),
    }
    opps = {}
    for oid, df in opp_dict.items():
        ret = df["retired"].to_numpy()
        k = int(np.argmax(ret)) if ret.any() else None
        jump = None
        if k is not None and k > 0:
            jump = float(np.hypot(df["opp_x"].iloc[k] - df["opp_x"].iloc[k - 1],
                                  df["opp_y"].iloc[k] - df["opp_y"].iloc[k - 1]))
        opps[str(oid)] = {
            "retired_frame": k,
            "retired_t": float(df["t_img"].iloc[k]) if k is not None else None,
            "teleport_m": jump,
            "pit_box_xy": [float(df["opp_x"].iloc[-1]), float(df["opp_y"].iloc[-1])]
            if k is not None else None,
            "retired_frames": int(ret.sum()),
            "valid_frames": int(df["valid"].to_numpy().sum()),
        }
    return {
        "n_frames": n,
        "ego": ego_q,
        "opponents": opps,
        "note": ("행은 지우지 않는다(row i ↔ camera/{i:06d}.jpg). 학습 시 "
                 "ego_features.valid 로 프레임을 고르고, 각 opp_<id>_features.valid 로 "
                 "그 프레임에 실제 트랙 위에 있는 상대차만 쓴다."),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="raw → processed/ (E2E/IL 학습용 선별·가공 피처)")
    ap.add_argument("run_dir", help="dataset/<run_id> 폴더")
    args = ap.parse_args()
    build_processed(args.run_dir)


if __name__ == "__main__":
    main()
