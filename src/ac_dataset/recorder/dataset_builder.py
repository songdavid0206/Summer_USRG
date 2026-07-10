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

import pandas as pd

# ego_state 로 내보낼 때 빼는 보조 컬럼(디버깅/병합 잔재)
_DROP_FOR_EGO = {"image_path", "t", "wall_time", "wall_time_x", "wall_time_y"}
# opponents_raw 에서 값 컬럼은 하드코딩하지 않는다. 플러그인/collect 에 채널이 추가되면
# (rel_z, opp_vx/vy/vz 등) 자동으로 따라온다. 키 컬럼만 제외.
_OPP_KEY_COLS = {"t", "t_tel", "opp_id", "wall_time"}


def _save(df: pd.DataFrame, base: Path) -> None:
    """parquet + csv 동시 저장. base 는 확장자 없는 경로."""
    base.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(base.with_suffix(".parquet"), index=False)
    df.to_csv(base.with_suffix(".csv"), index=False)


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
    img["frame"] = img.index

    # 카메라 프레임(t_img)마다 가장 가까운 telemetry tick(t_tel)을 붙인다.
    base = img.rename(columns={"t": "t_img"}).assign(t=lambda d: d["t_img"])
    tel2 = tel.rename(columns={"t": "t_tel"})
    tel2["t"] = tel2["t_tel"]
    merged = pd.merge_asof(
        base, tel2, on="t", direction="nearest", tolerance=tolerance,
    )
    merged["dt"] = (merged["t_img"] - merged["t_tel"]).abs()

    n_frames = len(merged)
    n_matched = int(merged["t_tel"].notna().sum())

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
        })
    _save(pd.DataFrame(cam_rows), cam_dir / "camera_index")

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
        frame_key = merged[["frame", "t_img", "t_tel"]]
        opp_ids = sorted(int(i) for i in opp["opp_id"].dropna().unique())
        value_cols = [c for c in opp.columns if c not in _OPP_KEY_COLS]
        for oid in opp_ids:
            sub = opp[opp["opp_id"] == oid].rename(columns={"t": "t_tel"})
            sub = sub[["t_tel"] + value_cols]
            m = frame_key.merge(sub, on="t_tel", how="left")
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
