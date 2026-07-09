"""
TelemetryLogger - ego + action + opponents raw 스트림 기록
==========================================================

설계 원칙 (방법 C: raw 따로 기록 -> 나중에 sync)
------------------------------------------------
- 이미지와 '같은 루프에서 묶지 않는다'. telemetry는 자기 주기(예: 50Hz)로
  들어오는 대로 timestamp `t`를 달아서 그대로 쌓기만 한다.
- 이미지와의 매칭은 나중에 sync.py가 timestamp 기준으로 수행한다.

기록 대상
--------
- ego  + action : state dict를 통째로 저장(raw 보존) + canonical steer/throttle/brake 컬럼 추가
- opponents     : 가변 개수라 별도 long-format 버퍼에 (t, opp_id, ...) 로 저장 (optional)

출력
----
  <run_dir>/telemetry_raw.parquet   # 한 행 = 한 tick, 컬럼: t, wall_time, <state...>, steer/throttle/brake
  <run_dir>/opponents_raw.parquet   # 한 행 = (t, opp_id, ...) ; 상대차 없으면 생성 안 함
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ..schema import split_state, OPPONENT_FIELDS


class TelemetryLogger:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._ego_rows: list[dict] = []
        self._opp_rows: list[dict] = []

    def log(self, t: float, state: dict, opponents: list[dict] | None = None,
            wall_time: float | None = None) -> None:
        """telemetry 한 tick 기록.

        t         : 수집 시작 기준 상대 시간(초). 이미지 로거와 '같은 시계'를 써야 함.
        state     : 레포 env.state dict (원본 그대로).
        opponents : [{opp_id, rel_x, rel_y, ...}, ...] 또는 None. 없으면 상대차 미기록.
        wall_time : 실제 벽시계(time.time()). 디버깅/외부 동기화용, 선택.
        """
        ego, action = split_state(state)
        row = {"t": t, "wall_time": wall_time if wall_time is not None else time.time()}
        row.update(ego)
        row.update(action)          # steer / throttle / brake
        self._ego_rows.append(row)

        if opponents:
            for opp in opponents:
                opp_row = {"t": t}
                # 스키마 필드만 골라 저장(정의되지 않은 키는 무시)
                for f in OPPONENT_FIELDS:
                    opp_row[f] = opp.get(f)
                self._opp_rows.append(opp_row)

    def flush(self) -> dict:
        """버퍼를 parquet으로 저장하고 요약을 반환."""
        ego_df = pd.DataFrame(self._ego_rows)
        # t, wall_time을 맨 앞으로
        front = [c for c in ("t", "wall_time") if c in ego_df.columns]
        ego_df = ego_df[front + [c for c in ego_df.columns if c not in front]]
        ego_path = self.run_dir / "telemetry_raw.parquet"
        ego_df.to_parquet(ego_path, index=False)

        summary = {
            "telemetry_rows": len(ego_df),
            "telemetry_columns": list(ego_df.columns),
            "opponent_rows": len(self._opp_rows),
        }

        if self._opp_rows:
            opp_df = pd.DataFrame(self._opp_rows)
            opp_df.to_parquet(self.run_dir / "opponents_raw.parquet", index=False)
            summary["has_opponents"] = True
        else:
            summary["has_opponents"] = False

        print(f"[telemetry] {len(ego_df)} ticks, {len(self._opp_rows)} opp rows "
              f"-> {ego_path}")
        return summary
