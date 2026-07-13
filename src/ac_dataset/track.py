"""
track - AC 트랙의 맵정보 + 레이싱 라인 추출
==========================================

AC 트랙 파일 `content/tracks/<track>/ai/fast_lane.ai` (AI 주행선)을 파싱해서
**레이싱 라인 + 트랙 폭(맵정보) + AI 속도 프로파일**을 뽑는다.

용도
----
- 조교님 요청: "맵정보 + 레이싱 라인"을 dataset에 함께 저장.
- run 폴더에 `track/` 를 만들어 camera/ ego_state/ opponent/ 와 나란히 둔다.
- 이후 (1) AI 자동주행 또는 (2) 이 라인을 포인트 팔로잉 하는 데이터 수집의 기준선으로 쓴다.

fast_lane.ai 포맷 (version 7, 리버스 엔지니어링)
------------------------------------------------
- header 16B : int version(7), int count, int 0, int 0
- detail 블록: count × 20B  = float3 position(x,y,z) + 8B(미사용)
- extra_count: int (== count)
- extra 블록 : count × 72B = 18 float
    [speed, gas, brake, obsLatG, radius, sideLeft, sideRight, camber, direction,
     normalX,Y,Z, length, forwardX,Y,Z, tag, grade]   (그 뒤 추가 데이터는 무시)

좌표계 주의
----------
AI 파일의 (x, y, z) 축 순서가 우리 ego telemetry(world_position)와 다르다.
좌표 범위 매칭 + 실주행 오버레이로 확인된 대응: **ego(x, y, z) = AI(z, x, y)**.
그래서 추출 시 ego 프레임으로 변환해 저장 → ego_state.world_position_* 와 바로 겹쳐진다.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd

# AC 기본 설치 경로 (다르면 tracks_dir 인자로 넘길 것)
DEFAULT_TRACKS_DIR = Path(
    r"C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/content/tracks")

_EXTRA_FIELDS = ["speed", "gas", "brake", "obsLatG", "radius", "side_left",
                 "side_right", "camber", "direction", "nX", "nY", "nZ",
                 "length", "fX", "fY", "fZ", "tag", "grade"]


def parse_fast_lane(ai_path: str | Path) -> pd.DataFrame:
    """fast_lane.ai → 레이싱 라인 DataFrame (ego 좌표계로 변환해서 반환).

    컬럼: i, x, y, z(=world_position 프레임), dist(누적거리 m),
          ai_speed_ms, ai_speed_kmh, gas, brake, radius, side_left, side_right
    """
    d = Path(ai_path).read_bytes()
    version, count = struct.unpack("<2i", d[:8])

    # --- detail 블록: 위치 (AI 좌표) ---
    ai = np.frombuffer(d, dtype="<f4", count=count * 5, offset=16).reshape(count, 5)
    ax, ay, az = ai[:, 0], ai[:, 1], ai[:, 2]

    # ego 프레임 변환: ego(x,y,z) = AI(z,x,y)
    x, y, z = az.copy(), ax.copy(), ay.copy()

    df = pd.DataFrame({"i": np.arange(count), "x": x, "y": y, "z": z})

    # 누적 거리(수평면)
    seg = np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0]))
    df["dist"] = np.cumsum(seg)

    # --- extra 블록: 속도/트랙폭 등 (있으면) ---
    off = 16 + count * 20
    try:
        extra_count = struct.unpack("<i", d[off:off + 4])[0]
        if extra_count == count:
            ex = np.frombuffer(d, dtype="<f4", count=count * 18,
                               offset=off + 4).reshape(count, 18)
            emap = {name: ex[:, k] for k, name in enumerate(_EXTRA_FIELDS)}
            df["ai_speed_ms"] = emap["speed"]
            df["ai_speed_kmh"] = emap["speed"] * 3.6
            df["gas"] = emap["gas"]
            df["brake"] = emap["brake"]
            df["radius"] = emap["radius"]
            df["side_left"] = emap["side_left"]
            df["side_right"] = emap["side_right"]
    except Exception:
        pass  # extra 없으면 라인 위치만으로도 충분

    return df


def _line_xy(df: pd.DataFrame) -> np.ndarray:
    return df[["x", "y"]].to_numpy()


def _ai_paths(tracks_dir: Path) -> list[Path]:
    """설치된 모든 fast_lane.ai 경로.

    트랙은 두 형태다: `<track>/ai/fast_lane.ai` (레이아웃 없음) 와
    `<track>/<layout>/ai/fast_lane.ai` (레이아웃 있음, 예: ks_silverstone/gp).
    둘 다 훑지 않으면 레이아웃 트랙을 통째로 놓친다.
    """
    return sorted(tracks_dir.glob("*/ai/fast_lane.ai")) + \
        sorted(tracks_dir.glob("*/*/ai/fast_lane.ai"))


def _track_name(ai_path: Path, tracks_dir: Path) -> str:
    """fast_lane.ai 경로 → 트랙 이름(레이아웃 포함, 예: 'ks_silverstone/gp')."""
    return ai_path.relative_to(tracks_dir).parent.parent.as_posix()


def identify_track(ego: pd.DataFrame, tracks_dir: str | Path = DEFAULT_TRACKS_DIR
                   ) -> tuple[str | None, float]:
    """ego world_position 을 설치된 모든 트랙 AI 라인과 비교해 트랙 이름 추정.

    반환: (track_name, mean_nearest_dist_m). 매칭 실패 시 (None, inf).
    """
    tracks_dir = Path(tracks_dir)
    px = ego["world_position_x"].to_numpy()
    py = ego["world_position_y"].to_numpy()
    # 정렬 시 telemetry 미매칭 프레임은 위치가 NaN(스레드 분리로 tol 밖 프레임이 생길 수 있음).
    # NaN 이 하나라도 섞이면 .min()/거리계산이 전부 NaN → 식별이 inf 로 실패하므로 먼저 제거한다.
    finite = np.isfinite(px) & np.isfinite(py)
    px, py = px[finite], py[finite]
    if len(px) == 0:
        return (None, float("inf"))
    # 주행점 샘플(속도)
    idx = np.linspace(0, len(px) - 1, min(len(px), 200)).astype(int)
    dpts = np.column_stack([px[idx], py[idx]])
    dbox = (px.min(), px.max(), py.min(), py.max())

    best, best_d = None, float("inf")
    for ai_path in _ai_paths(tracks_dir):
        name = _track_name(ai_path, tracks_dir)
        try:
            line = _line_xy(parse_fast_lane(ai_path))
        except Exception:
            continue
        # 1차: bounding box 가 아예 안 겹치면 스킵
        lb = (line[:, 0].min(), line[:, 0].max(), line[:, 1].min(), line[:, 1].max())
        if lb[1] < dbox[0] or lb[0] > dbox[1] or lb[3] < dbox[2] or lb[2] > dbox[3]:
            continue
        # 2차: 주행점 → 라인 최근접 평균거리
        dmin = np.sqrt(((dpts[:, None, :] - line[None, :, :]) ** 2).sum(-1)).min(1)
        md = float(dmin.mean())
        if md < best_d:
            best, best_d = name, md
    # 트랙 폭(수십 m) 이내면 확실한 매칭
    return (best, best_d) if best_d < 30.0 else (None, best_d)


def extract_track_for_run(run_dir: str | Path,
                          tracks_dir: str | Path = DEFAULT_TRACKS_DIR,
                          track_name: str | None = None) -> dict:
    """run 폴더의 ego_state 로 트랙을 식별하고 `track/` 폴더에 맵+라인을 저장.

    track_name 을 직접 주면 식별을 건너뛴다.
    출력: <run_dir>/track/racing_line.parquet|csv , <run_dir>/track/track.json
    """
    run_dir = Path(run_dir)
    tracks_dir = Path(tracks_dir)
    ego = pd.read_parquet(run_dir / "ego_state" / "ego_state.parquet")

    if track_name is None:
        track_name, dist = identify_track(ego, tracks_dir)
        if track_name is None:
            raise RuntimeError(
                f"트랙 자동식별 실패(최근접 {dist:.0f}m). --track 으로 이름을 직접 지정하거나 "
                f"tracks_dir 경로를 확인하세요.")
    else:
        dist = float("nan")

    ai_path = tracks_dir / track_name / "ai" / "fast_lane.ai"
    line = parse_fast_lane(ai_path)

    out = run_dir / "track"
    out.mkdir(parents=True, exist_ok=True)
    line.to_parquet(out / "racing_line.parquet", index=False)
    line.to_csv(out / "racing_line.csv", index=False)

    meta = {
        "track_name": track_name,
        "n_points": int(len(line)),
        "has_speed_profile": "ai_speed_ms" in line.columns,
        "identify_mean_dist_m": None if np.isnan(dist) else round(dist, 2),
        "source": str(ai_path),
        "coord_frame": "ego world_position (x,y,z) = AI file (z,x,y)",
    }
    (out / "track.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[track] '{track_name}' ({len(line)} pts, "
          f"식별거리 {meta['identify_mean_dist_m']}m) -> {out}")
    return meta


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="run 폴더의 트랙을 식별하고 맵+레이싱라인을 track/ 폴더에 저장")
    ap.add_argument("run_dir", help="dataset/<run_id> 폴더")
    ap.add_argument("--track", default=None, help="트랙 이름 직접 지정(식별 건너뜀)")
    ap.add_argument("--tracks-dir", default=str(DEFAULT_TRACKS_DIR),
                    help="AC content/tracks 경로")
    args = ap.parse_args()
    extract_track_for_run(args.run_dir, tracks_dir=args.tracks_dir,
                          track_name=args.track)


if __name__ == "__main__":
    main()
