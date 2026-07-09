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

출력: <run>/processed/ego_features.{parquet,csv}, opponent_features.{...}, track_features.{...}
"""

from __future__ import annotations

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


def _build_ego(ego: pd.DataFrame) -> pd.DataFrame:
    cols = ["frame", "t_img"] + [c for c in EGO_KEEP if c in ego.columns]
    out = ego[cols].copy()
    if "angular_velocity_y" in ego.columns:
        out["yaw_rate"] = ego["angular_velocity_y"]   # 요레이트(직관적 별칭)
    return out


def _build_opponents(run_dir: Path, ego: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """상대차마다 프레임 정렬된 피처 DataFrame을 만들어 {opp_id: df}로 반환."""
    opp_dirs = sorted((run_dir / "opponent").glob("opp_*"))
    if not opp_dirs:
        return {}
    ref = ego[["frame", "t_img", "world_position_x", "world_position_y", "yaw", "speed"]].rename(
        columns={"world_position_x": "ego_x", "world_position_y": "ego_y", "speed": "ego_speed"})
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
        # ego-frame (앞/옆). forward=+진행방향, lateral=+왼쪽
        o["opp_forward"] = o["rel_x"] * c + o["rel_y"] * s
        o["opp_lateral"] = -o["rel_x"] * s + o["rel_y"] * c
        # 속도: 위치 시간차분은 쓰지 않는다. 상대차 데이터가 프레임 쌍으로 held(≈절반 주기로만 갱신)
        # 되어 시간차분이 0↔폭주로 튀기 때문. 대신 **플러그인 실측 속도(opp_speedKMH) + 진행방향
        # (opp_yaw, 위치기반 방향과 0.2도 일치 검증)** 로 속도벡터를 구성 → 물리적으로 정확, 글리치 없음.
        opp_spd = (o["opp_speedKMH"] / 3.6).to_numpy()   # m/s
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
        keep = ["frame", "t_img", "opp_id", "present",
                "opp_x", "opp_y", "opp_forward", "opp_lateral", "distance",
                "opp_vx", "opp_vy", "opp_speed", "rel_vx", "rel_vy", "closing_speed",
                "opp_yaw", "opp_speedKMH"]
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

    summary = {
        "ego_rows": len(ego_f), "ego_cols": len(ego_f.columns),
        "n_opponents": len(opp_dict),
        "opponent_ids": sorted(opp_dict),
        "track_features": trk_f is not None,
    }
    print(f"[processed] ego {summary['ego_cols']}열×{summary['ego_rows']} / "
          f"opponent {summary['n_opponents']}대(opp별 파일) / "
          f"track {'O' if trk_f is not None else 'X'} -> {out}")
    return summary


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="raw → processed/ (E2E/IL 학습용 선별·가공 피처)")
    ap.add_argument("run_dir", help="dataset/<run_id> 폴더")
    args = ap.parse_args()
    build_processed(args.run_dir)


if __name__ == "__main__":
    main()
