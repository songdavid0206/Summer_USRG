"""
dataset_builder - raw 스트림을 camera / ego_state / opponent 폴더 구조로 정리
===========================================================================

방법 C의 마지막 단계(구 sync.py 대체). 실시간으로 묶지 않고 raw로 쌓아둔
스트림(image_index / telemetry_raw / opponents_raw)을 **카메라 프레임 기준**으로
정렬해서, 조교님 요청 레이아웃대로 폴더를 나눠 저장한다.

출력 레이아웃
------------
  <run_dir>/
    camera/
      000000.png, 000001.png, ...        # 프레임별 이미지 (frame 번호 = 파일명)
      camera_index.parquet / .csv        # frame, t_img, image_path
    ego_state/
      ego_state.parquet / .csv           # 프레임당 1행: frame,t_img,t_tel,dt,<ego...>,steer,throttle,brake
    opponent/
      opp_<id>/
        opp_<id>.parquet / .csv          # 프레임당 1행: frame,t_img,t_tel,present,rel_x,rel_y,distance,opp_speedKMH,opp_yaw
      (상대차 없는 세션이면 opponent/ 자체가 생성되지 않음)

정렬 규칙 (핵심)
--------------
- 기준(축) = 카메라 프레임. 이미지 한 장마다 pandas.merge_asof(direction="nearest",
  tolerance)로 가장 가까운 telemetry tick을 붙인다.
- **모든 프레임을 유지**한다(drop 하지 않음). tolerance 내 telemetry가 없으면 그 프레임의
  ego 값은 NaN. 이렇게 하면 camera / ego_state / 각 opp 폴더의 길이가 정확히 같아지고,
  camera/{i}.png <-> ego_state 행 i <-> opp_<id> 행 i 가 항상 1:1로 대응된다.
- 상대차는 telemetry tick(t_tel)에 물려 기록됐으므로, 프레임의 t_tel로 각 상대차 시계열을
  붙인다. 그 프레임에 그 상대차가 없으면 present=False + 값 NaN.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# ego_state 로 내보낼 때 빼는 보조 컬럼(디버깅/병합 잔재)
_DROP_FOR_EGO = {"image_path", "t", "wall_time", "wall_time_x", "wall_time_y",
                 "_img_mono", "raw_frame"}
# opponents_raw 에서 값 컬럼은 하드코딩하지 않는다. 플러그인/collect 에 채널이 추가되면
# (rel_z, opp_vx/vy/vz 등) 자동으로 따라온다. 키 컬럼만 제외.
_OPP_KEY_COLS = {"t", "t_tel", "opp_id", "wall_time"}


def _save(df: pd.DataFrame, base: Path) -> None:
    """parquet + csv 동시 저장. base 는 확장자 없는 경로."""
    base.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(base.with_suffix(".parquet"), index=False)
    df.to_csv(base.with_suffix(".csv"), index=False)


def _is_angle(col: str) -> bool:
    """각도 컬럼(랩보간 시 sin/cos 로 처리). yaw/pitch/roll/heading 포함."""
    c = col.lower()
    return any(k in c for k in ("yaw", "pitch", "roll", "heading"))


def _interp_col(x_new: np.ndarray, x: np.ndarray, y: np.ndarray, angle: bool) -> np.ndarray:
    """x(단조증가) 위의 y 를 x_new 로 선형보간. 각도면 sin/cos 보간(±180 wrap 안전)."""
    if angle:
        s = np.interp(x_new, x, np.sin(y))
        c = np.interp(x_new, x, np.cos(y))
        return np.arctan2(s, c)
    return np.interp(x_new, x, y)


def _nearest_col(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample y at the nearest x without creating fractional discrete labels."""
    right = np.searchsorted(x, x_new).clip(0, len(x) - 1)
    left = np.maximum(right - 1, 0)
    nearest = np.where(np.abs(x_new - x[left]) <= np.abs(x[right] - x_new),
                       left, right)
    return y[nearest]


def _is_nearest_channel(col: str) -> bool:
    """Channels that must not be linearly interpolated across lap boundaries."""
    return col.lower() in {
        "lapcount", "completedlaps", "lapinvalidated",
        "normalizedsplineposition", "numberoftyresout", "isinpit",
        "currenttime", "ilasttime", "bestlap",
        "actualgear", "gear", "done", "out_of_track",
    }


def _mono_gametime(tel: pd.DataFrame):
    """telemetry 에 **단조증가 게임시간(ms)** 축을 만든다.

    timestamp_ac(초, 랩리셋 없이 단조)를 ms 로 바꿔 monotonic 축으로 쓰고, iCurrentTime
    (=currentTime, 랩마다 0으로 리셋)을 이 축으로 옮기는 lap_offset(= mono - currentTime,
    랩당 상수)도 낸다. 이미지는 currentTime 스케일의 game_time_ms 만 갖고 있으므로,
    game_time_ms + lap_offset 로 같은 monotonic 축에 올린다.

    반환: (tel_sorted, mono_ms, lap_offset, t_wall)  — 모두 timestamp_ac 기준 정렬·중복제거.
    """
    tel = tel.sort_values("timestamp_ac").reset_index(drop=True)
    tac = tel["timestamp_ac"].to_numpy(float)
    # np.interp 는 x 가 순증가여야 안전. 동일 timestamp_ac 중복 제거(뒤 값 유지).
    keep = np.concatenate([np.diff(tac) > 0, [True]])
    tel = tel[keep].reset_index(drop=True)
    tac = tel["timestamp_ac"].to_numpy(float)
    mono = tac * 1000.0
    lap_off = mono - tel["currentTime"].to_numpy(float)
    return tel, mono, lap_off, tel["t"].to_numpy(float)


def _pair_by_gametime(img: pd.DataFrame, tel: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """이미지↔telemetry 를 **게임시간으로 매칭+보간**해 merged(이미지행 + 보간 telemetry) 반환.

    각 이미지의 game_time_ms(그 프레임 화면의 게임시간)를 monotonic 축(timestamp_ac*1000)으로
    올린 뒤, telemetry 의 모든 채널을 그 게임시간으로 선형보간(각도는 wrap 보간)한다. 벽시계
    nearest 매칭의 ±지터가 사라지고, 픽셀에 그려진 그 순간의 차량상태와 정합된다.
    """
    tel, mono, lap_off, t_wall = _mono_gametime(tel)

    ti = img["t"].to_numpy(float)
    gi = img["game_time_ms"].to_numpy(float)
    # 이미지의 lap clock은 매 랩 0으로 리셋되므로 같은 game_time_ms가 여러 랩에 존재한다.
    # 먼저 벽시계로 대략적인 monotonic game time을 구하고, 각 랩의 안정적인 중앙 offset
    # 후보 중 그 시점에 가장 가까운 랩 주기를 고른다. 벽시계는 랩 주기 선택에만 쓰고,
    # 주기 안의 정밀 정렬은 game_time_ms가 담당한다.
    #
    # 이 선택이 필요한 이유: AC는 결승선에서 game_time_ms, LapCount, completedLaps를
    # 서로 다른 render/telemetry tick에 갱신할 수 있다. 예를 들어 lap=3, time=3ms가
    # 세 번째 랩의 시작인지 네 번째 랩 직전 reset tick인지 lap 번호만으로는 모호하다.
    j = np.searchsorted(t_wall, ti).clip(1, len(t_wall) - 1)
    left = j - 1
    near = np.where((ti - t_wall[left]) <= (t_wall[j] - ti), left, j)
    img_mono = gi + lap_off[near]  # legacy fallback
    if "completed_laps" in img.columns and "completedLaps" in tel.columns:
        telemetry_laps = pd.to_numeric(tel["completedLaps"], errors="coerce").to_numpy()
        stable_offsets = []
        for lap in np.unique(telemetry_laps[np.isfinite(telemetry_laps)]):
            offsets = lap_off[telemetry_laps == lap]
            offsets = offsets[np.isfinite(offsets)]
            if len(offsets):
                stable_offsets.append(float(np.median(offsets)))
        if stable_offsets:
            offsets = np.asarray(stable_offsets, dtype=float)
            wall_mono = np.interp(ti, t_wall, mono)
            candidates = gi[:, None] + offsets[None, :]
            choice = np.abs(candidates - wall_mono[:, None]).argmin(axis=1)
            img_mono = candidates[np.arange(len(gi)), choice]

    base = img.rename(columns={"t": "t_img"}).reset_index(drop=True)
    # 보간할 telemetry 채널(키/문자열 제외). 나머지(비수치)는 최근접.
    skip = {"t", "wall_time"}
    tel_cols = [c for c in tel.columns if c not in skip]
    valid = (img_mono >= mono[0]) & (img_mono <= mono[-1])
    # 컬럼을 하나씩 대입하면 DataFrame 이 조각화돼(pandas PerformanceWarning) 느리다.
    # dict 로 모아 한 번에 concat 한다.
    cols: dict[str, np.ndarray] = {}
    for c in tel_cols:
        col = tel[c]
        if np.issubdtype(col.dtype, np.number):
            values = col.to_numpy(float)
            vals = (_nearest_col(img_mono, mono, values)
                    if _is_nearest_channel(c)
                    else _interp_col(img_mono, mono, values, _is_angle(c)))
            cols[c] = np.where(valid, vals, np.nan)
        else:  # 비수치(플래그/문자열): 게임시간 최근접
            cols[c] = _nearest_col(img_mono, mono, col.to_numpy())
    # t_tel(그 게임시간의 telemetry 벽시계, 참고용) + dt.
    # 게임시간 축 밖(범위 초과) 프레임은 t_tel NaN → downstream 에서 drop.
    t_tel = np.where(valid, np.interp(img_mono, mono, t_wall), np.nan)
    cols["t_tel"] = t_tel
    cols["dt"] = np.abs(base["t_img"].to_numpy(float) - t_tel)
    # 상대차도 같은 게임시간으로 보간하도록 프레임별 monotonic 게임시간을 실어 보낸다(저장 전 제거).
    cols["_img_mono"] = img_mono
    return pd.concat([base, pd.DataFrame(cols, index=base.index)], axis=1)


def _opp_gametime_frame(opp_sub: pd.DataFrame, tel: pd.DataFrame,
                        img_mono: np.ndarray, value_cols: list[str]) -> pd.DataFrame:
    """상대차 한 대(opp_sub)를 프레임별 game시간(img_mono)으로 보간. present 포함 DataFrame 반환.

    상대차 행은 게임시간이 없지만 ego tick 벽시계(t)에 물려 있으므로, telemetry 의
    (벽시계 t → monotonic 게임시간) 관계로 상대차 행의 게임시간을 구한 뒤 보간한다.
    """
    tel_s, mono, _off, t_wall = _mono_gametime(tel)
    ot = opp_sub["t"].to_numpy(float)
    opp_mono = np.interp(ot, t_wall, mono)   # 상대차 행의 monotonic 게임시간
    order = np.argsort(opp_mono)
    opp_mono = opp_mono[order]
    keep = np.concatenate([np.diff(opp_mono) > 0, [True]])
    opp_mono = opp_mono[keep]
    res = {}
    inrange = (img_mono >= opp_mono[0]) & (img_mono <= opp_mono[-1]) if len(opp_mono) else np.zeros(len(img_mono), bool)
    for c in value_cols:
        y = opp_sub[c].to_numpy()[order][keep]
        if np.issubdtype(np.asarray(y).dtype, np.number):
            v = _interp_col(img_mono, opp_mono, y.astype(float), _is_angle(c))
            res[c] = np.where(inrange, v, np.nan)
        else:
            k = np.searchsorted(opp_mono, img_mono).clip(0, len(opp_mono) - 1)
            res[c] = y[k]
    df = pd.DataFrame(res)
    df.insert(0, "present", inrange & df["rel_x"].notna() if "rel_x" in df else inrange)
    return df


def build_dataset(run_dir: str | Path, tolerance: float = 0.05) -> dict:
    """한 run 폴더의 raw 스트림을 camera/ego_state/opponent 폴더 구조로 정리.

    tolerance : 이미지 t와 telemetry t의 허용 시간차(초). 기본 50ms.
                이 안에 telemetry가 없는 프레임은 ego 값이 NaN(프레임 자체는 유지).
    반환       : 요약 dict (프레임 수, 상대차 id 목록 등).
    """
    run_dir = Path(run_dir)
    img = pd.read_parquet(run_dir / "image_index.parquet")
    tel = pd.read_parquet(run_dir / "telemetry_raw.parquet")

    img = img.sort_values("t").reset_index(drop=True)
    tel = tel.sort_values("t").reset_index(drop=True)

    # frame 은 이미지 순서대로 0..N-1 로 다시 매긴다(폴더 파일명과 일치시키기 위함).
    img = img.reset_index(drop=True)
    img["raw_frame"] = img["frame"].astype(int)
    img["frame"] = img.index

    # 카메라 프레임마다 telemetry 를 붙인다. **게임시간 페어링**(권장): 이미지의 game_time_ms
    # (그 프레임 화면의 게임시간)를 telemetry 의 monotonic 게임시간(timestamp_ac)에 보간해 매칭
    # → 벽시계 nearest 의 지터·캡처 파이프라인 오차 제거. 구버전 run(게임시간 미기록)은 폴백.
    use_gametime = ("game_time_ms" in img.columns
                    and {"timestamp_ac", "currentTime"} <= set(tel.columns)
                    and img["game_time_ms"].notna().any())
    # 두 스트림의 벽시계 구간이 실제로 겹치는지 확인. 한쪽이 일찍 죽으면(예: telemetry 수신 중단)
    # 겹침이 없는데도 보간이 값을 만들어내 조용히 쓰레기 데이터가 된다.
    img_span = float(img["t"].max() - img["t"].min())
    ov = min(img["t"].max(), tel["t"].max()) - max(img["t"].min(), tel["t"].min())
    if ov < 0.5 * img_span:
        raise RuntimeError(
            "이미지와 telemetry 의 기록 구간이 충분히 겹치지 않습니다"
            " (겹침 {:.2f}s / 이미지 {:.2f}s). 이미지 {:.2f}~{:.2f}s, telemetry {:.2f}~{:.2f}s."
            " 한쪽 스트림이 중간에 끊긴 run 입니다 — 재수집이 필요합니다."
            " (겹치지 않는 구간은 보간이 값을 만들어내 조용히 잘못된 라벨이 됩니다)".format(
                max(ov, 0.0), img_span,
                img["t"].min(), img["t"].max(), tel["t"].min(), tel["t"].max()))

    if use_gametime:
        merged = _pair_by_gametime(img, tel, tolerance)
        print("[build] 페어링 = 게임시간(iCurrentTime↔timestamp_ac) 보간")
    else:
        base = img.rename(columns={"t": "t_img"}).assign(t=lambda d: d["t_img"])
        tel2 = tel.rename(columns={"t": "t_tel"})
        tel2["t"] = tel2["t_tel"]
        merged = pd.merge_asof(base, tel2, on="t", direction="nearest", tolerance=tolerance)
        merged["dt"] = (merged["t_img"] - merged["t_tel"]).abs()
        print("[build] 페어링 = 벽시계 nearest (게임시간 미기록 폴백)")

    n_frames_raw = len(merged)
    # tolerance 안에 telemetry 를 못 찾은 카메라 프레임은 버린다(그 사진에 대응하는 차량 상태를
    # 모르므로 NaN 으로 남기면 학습 불가·검증 실패). 주로 종료 순간 큐에 남은 꼬리 프레임 몇 장.
    merged = merged[merged["t_tel"].notna()].reset_index(drop=True)
    n_dropped = n_frames_raw - len(merged)
    if n_dropped:
        print(f"[build] telemetry 미매칭 {n_dropped}프레임 제외(tol {tolerance*1e3:.0f}ms 밖)")
    # 프레임 번호를 남은 것 기준으로 0..N-1 재부여(폴더 파일명과 1:1 유지).
    merged["frame"] = np.arange(len(merged))

    n_frames = len(merged)
    n_matched = n_frames

    # ------------------------------------------------------------------ camera
    cam_dir = run_dir / "camera"
    cam_dir.mkdir(parents=True, exist_ok=True)
    cam_rows = []
    for row in merged.itertuples(index=False):
        frame = int(row.frame)
        src = run_dir / row.image_path
        ext = Path(row.image_path).suffix or ".png"
        dst = cam_dir / f"{frame:06d}{ext}"
        # 이미 camera/ 에 있으면(신규 수집 경로) 복사 안 함. 다른 폴더(예: 기존 images/)면 복사.
        if src.exists() and src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        cam_rows.append({
            "frame": frame,
            "t_img": row.t_img,
            "image_path": f"camera/{frame:06d}{ext}",
            **{c: getattr(row, c) for c in merged.columns
               if c.startswith("cam_") or c in ("graphics_packet", "game_time_ms",
                                                  "completed_laps")},
        })
    _save(pd.DataFrame(cam_rows), cam_dir / "camera_index")

    # CSP가 카메라와 같은 렌더 tick에서 기록한 차량 full pose + visual AABB.
    # telemetry 매칭에서 제외된 이미지 행도 있으므로 raw image_frame을 새 frame으로 재매핑한다.
    render_path = run_dir / "render_car_state.parquet"
    if render_path.exists() and "raw_frame" in merged.columns:
        render_cars = pd.read_parquet(render_path)
        frame_map = merged[["raw_frame", "frame"]].drop_duplicates("raw_frame")
        render_cars = render_cars.merge(frame_map, left_on="image_frame",
                                        right_on="raw_frame", how="inner")
        render_cars = render_cars.drop(columns=["image_frame", "raw_frame"])
        front = ["frame", "car_id", "render_frame", "sim_time_ms", "car_timestamp_ms",
                 "car_state_age_ms"]
        render_cars = render_cars[[c for c in front if c in render_cars.columns]
                                  + [c for c in render_cars.columns if c not in front]]
        _save(render_cars.sort_values(["frame", "car_id"]).reset_index(drop=True),
              cam_dir / "render_car_state")

    # CSP physics callback에서 받은 실제 ego 충돌 시작 이벤트. 이벤트를 처음 읽은
    # raw 이미지 프레임을 최종 정렬 프레임으로 바꾸고 해당 프레임의 t_img를 붙인다.
    collision_path = run_dir / "collision_events.parquet"
    if collision_path.exists() and "raw_frame" in merged.columns:
        collisions = pd.read_parquet(collision_path)
        frame_map = merged[["raw_frame", "frame", "t_img"]].drop_duplicates("raw_frame")
        collisions = collisions.merge(frame_map, left_on="image_frame",
                                      right_on="raw_frame", how="inner")
        collisions = collisions.drop(columns=["image_frame", "raw_frame"])
        front = ["frame", "t_img", "collision_event_seq", "collision_render_frame",
                 "collision_sim_time_ms", "collision_lap_time_ms",
                 "collision_lap_count", "collision_car_id", "collision_with"]
        collisions = collisions[[c for c in front if c in collisions.columns]
                                + [c for c in collisions.columns if c not in front]]
        _save(collisions.sort_values(["frame", "collision_event_seq"]).reset_index(drop=True),
              cam_dir / "collision_events")

    # index에 없는 자투리 png 정리(예: Ctrl+C가 저장 직후·기록 직전에 걸려 타임스탬프 없이
    # 남은 마지막 프레임). 정렬된 프레임만 남긴다.
    kept = {r["image_path"].split("/")[-1] for r in cam_rows}
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for f in cam_dir.glob(ext):
            if f.name not in kept:
                f.unlink()

    # --------------------------------------------------------------- ego_state
    ego_df = merged[[c for c in merged.columns if c not in _DROP_FOR_EGO]].copy()
    front = ["frame", "t_img", "t_tel", "dt"]
    ego_df = ego_df[[c for c in front if c in ego_df.columns]
                    + [c for c in ego_df.columns if c not in front]]
    _save(ego_df, run_dir / "ego_state" / "ego_state")

    # ---------------------------------------------------------------- opponent
    opp_ids: list[int] = []
    opp_path = run_dir / "opponents_raw.parquet"
    if opp_path.exists():
        opp = pd.read_parquet(opp_path)
        opp_ids = sorted(int(i) for i in opp["opp_id"].dropna().unique())
        value_cols = [c for c in opp.columns if c not in _OPP_KEY_COLS]
        frame_meta = merged[["frame", "t_img", "t_tel"]].reset_index(drop=True)
        for oid in opp_ids:
            sub = opp[opp["opp_id"] == oid]
            if use_gametime:
                # 상대차도 이미지 게임시간으로 보간(ego 와 동일 축). 상대차 행 게임시간은
                # ego telemetry 의 (벽시계→게임시간) 관계로 구한다.
                vals = _opp_gametime_frame(sub[["t"] + value_cols], tel,
                                           merged["_img_mono"].to_numpy(float), value_cols)
                m = pd.concat([frame_meta, vals.reset_index(drop=True)], axis=1)
            else:
                sub2 = sub.rename(columns={"t": "t_tel"})[["t_tel"] + value_cols]
                m = frame_meta.merge(sub2, on="t_tel", how="left")
                m.insert(3, "present", m["rel_x"].notna())
            _save(m, run_dir / "opponent" / f"opp_{oid}" / f"opp_{oid}")

    summary = {
        "frames": n_frames,
        "frames_matched": n_matched,
        "opp_ids": opp_ids,
        "n_opponents": len(opp_ids),
        "tolerance_ms": tolerance * 1000,
    }
    opp_desc = f", opponents={opp_ids}" if opp_ids else ""
    print(f"[build] {n_matched}/{n_frames} frames matched (tol={tolerance*1000:.0f}ms) "
          f"-> camera/ ego_state/ opponent/{opp_desc}")
    return summary


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="raw 스트림 -> camera/ego_state/opponent 폴더 구조로 정리")
    ap.add_argument("run_dir", help="dataset/<run_id> 폴더")
    ap.add_argument("--tolerance", type=float, default=0.05, help="허용 시간차(초)")
    args = ap.parse_args()
    build_dataset(args.run_dir, tolerance=args.tolerance)


if __name__ == "__main__":
    main()
