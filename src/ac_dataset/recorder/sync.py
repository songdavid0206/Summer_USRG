"""
sync - raw 스트림을 frame 단위 sample로 병합 (오프라인)
=====================================================

방법 C의 마지막 단계. 실시간으로 묶지 않고 raw로 쌓아둔 두 스트림
(image_index / telemetry_raw)을 timestamp 기준으로 합쳐서
학습용 samples.parquet 를 만든다.

매칭 규칙
--------
- 기준(축) = 이미지 프레임. 이미지가 보통 telemetry보다 저주파(예: img 20Hz vs tel 50Hz)라
  "이미지 한 장 <- 가장 가까운 telemetry tick 하나"로 붙이는 게 자연스럽다.
- pandas.merge_asof(direction="nearest", tolerance)로 시간차가 tolerance 이내인 것만 매칭.
  tolerance를 넘으면 매칭 실패로 보고 기본적으로 그 프레임은 버린다(drop_unmatched).

출력
----
  <run_dir>/samples.parquet
    한 행 = (frame, image_path, t_img, t_tel, dt,  <ego...>, steer/throttle/brake, opp_count)
    -> IL 학습:  (image + ego + opponents) -> (steer, throttle, brake)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def sync_run(run_dir: str | Path, tolerance: float = 0.05,
             drop_unmatched: bool = True) -> pd.DataFrame:
    """한 run 폴더의 raw 스트림을 병합해 samples.parquet 생성.

    tolerance      : 이미지 t와 telemetry t의 허용 시간차(초). 기본 50ms.
    drop_unmatched : tolerance 내 telemetry가 없는 프레임을 버릴지 여부.
    """
    run_dir = Path(run_dir)
    img = pd.read_parquet(run_dir / "image_index.parquet")
    tel = pd.read_parquet(run_dir / "telemetry_raw.parquet")

    img = img.sort_values("t").reset_index(drop=True)
    tel = tel.sort_values("t").reset_index(drop=True)

    # wall_time은 양쪽에 다 있어 병합 시 _x/_y로 갈린다. 디버깅용이라 image쪽만 남기고 정리.
    img = img.drop(columns=["wall_time"], errors="ignore")

    # telemetry의 t를 별도 컬럼으로 보존(merge_asof는 key 컬럼을 병합해버리므로)
    tel = tel.rename(columns={"t": "t_tel"})
    tel["t"] = tel["t_tel"]

    samples = pd.merge_asof(
        img.rename(columns={"t": "t_img"}).assign(t=lambda d: d["t_img"]),
        tel,
        on="t",
        direction="nearest",
        tolerance=tolerance,
    )

    samples["dt"] = (samples["t_img"] - samples["t_tel"]).abs()
    matched = samples["t_tel"].notna()
    n_total, n_matched = len(samples), int(matched.sum())

    if drop_unmatched:
        samples = samples[matched].reset_index(drop=True)

    # 상대차: 있으면 매칭된 telemetry t 기준으로 개수를 붙인다(상세는 opponents_raw에 그대로)
    opp_path = run_dir / "opponents_raw.parquet"
    if opp_path.exists():
        opp = pd.read_parquet(opp_path)
        opp_count = opp.groupby("t").size().rename("opp_count")
        samples = samples.merge(opp_count, left_on="t_tel", right_index=True, how="left")
        samples["opp_count"] = samples["opp_count"].fillna(0).astype(int)
    else:
        samples["opp_count"] = 0

    # 컬럼 정렬: 식별/시간 -> 이미지 -> 나머지
    front = ["frame", "image_path", "t_img", "t_tel", "dt", "opp_count"]
    front = [c for c in front if c in samples.columns]
    samples = samples[front + [c for c in samples.columns if c not in front and c != "t"]]

    out = run_dir / "samples.parquet"
    samples.to_parquet(out, index=False)

    dropped = n_total - n_matched
    print(f"[sync] {n_matched}/{n_total} frames matched "
          f"(tol={tolerance*1000:.0f}ms, dropped={dropped}) -> {out}")
    if n_matched:
        print(f"[sync] dt: mean={samples['dt'].mean()*1000:.1f}ms "
              f"max={samples['dt'].max()*1000:.1f}ms")
    return samples


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="raw 스트림 -> samples.parquet 병합")
    ap.add_argument("run_dir", help="dataset/<run_id> 폴더")
    ap.add_argument("--tolerance", type=float, default=0.05, help="허용 시간차(초)")
    ap.add_argument("--keep-unmatched", action="store_true",
                    help="tolerance 초과 프레임도 남김(telemetry 컬럼은 NaN)")
    args = ap.parse_args()
    sync_run(args.run_dir, tolerance=args.tolerance,
             drop_unmatched=not args.keep_unmatched)
