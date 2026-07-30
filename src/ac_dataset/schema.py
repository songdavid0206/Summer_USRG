"""
데이터셋 필드 정의 (single source of truth)
==========================================

ego / action / opponent 각 그룹의 필드명을 한 곳에서 관리한다.
로거(telemetry_logger), 병합(sync), 학습 코드가 모두 여기를 참조하도록 해서
필드명이 여러 파일에 흩어지지 않게 한다.

키 이름은 참조 레포의 `ac_env.py`와 실제 sensors_par telemetry를 대조해 확정했다.
"""

from __future__ import annotations


# ----------------------------------------------------------------------------
# action(정답 라벨): 사람이 실제로 한 조작값. telemetry에서 '읽기만' 한다.
#   canonical 이름  ->  레포 state dict의 원본 키
#   * vjoy로 보내는 게 아니라, 플러그인 telemetry에 담긴 현재 조작값을 라벨로 씀.
#   실측 키: steerAngle / accStatus / brakeStatus. 페달 범위는 0~1.
# ----------------------------------------------------------------------------
ACTION_MAP: dict[str, str] = {
    "steer":    "steerAngle",   # 조향 (부호 = 좌/우)
    "throttle": "accStatus",    # 가속 페달 (0~1)
    "brake":    "brakeStatus",  # 브레이크 페달 (0~1)
}


# ----------------------------------------------------------------------------
# ego(내 차 상태): 레포 ac_env.py의 expand_state()가 실제로 만드는 키 목록.
#   telemetry_raw에는 state dict '전체'를 저장하므로 이 목록은 참고/검증용
#   (여기 없는 키가 state에 있어도 raw에는 다 들어간다).
#   출처: dasGringuen/assetto_corsa_gym AssettoCorsaEnv/ac_env.py: expand_state()
# ----------------------------------------------------------------------------
EGO_KEYS_HINT: list[str] = [
    # 위치/움직임
    "world_position_x", "world_position_y",
    "speed",
    "velocity_x", "velocity_y", "velocity_z",
    "yaw", "roll",
    # 동역학
    "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
    "accelX", "accelY",
    # 기어/엔진
    "actualGear", "RPM", "LastFF",
    # 랩/트랙
    "LapCount", "LapDist", "NormalizedSplinePosition", "gap",
    # 타이어 (fl/fr/rl/rr)
    "SlipAngle_fl", "SlipAngle_fr", "SlipAngle_rl", "SlipAngle_rr",
    "wheel_speed_fl", "wheel_speed_fr", "wheel_speed_rl", "wheel_speed_rr",
    "tyre_slip_ratio_fl", "tyre_slip_ratio_fr", "tyre_slip_ratio_rl", "tyre_slip_ratio_rr",
    "Dy_fl", "Dy_fr", "Dy_rl", "Dy_rr",
    "numberOfTyresOut",
    # 에피소드 상태
    "done", "out_of_track", "reward", "currentTime",
]

# 참고: 서버로 '보낼' 때의 제어 키는 steer/acc/brake 로 이름이 다르다.
#   우리는 제어를 보내지 않고 state의 steerAngle/accStatus/brakeStatus 를 라벨로 '읽기만' 한다.


# ----------------------------------------------------------------------------
# opponent(상대 차량): 가변 개수(0~N)라 별도 long-format 테이블에 저장.
#   한 행 = (t, opp_id, <아래 필드들>).  ego 기준 상대좌표를 저장하는 걸 권장.
#   opponents_server 실측 확인 완료. 상대 없는 solo에서는 테이블이 비어도 정상이다.
# ----------------------------------------------------------------------------
OPPONENT_FIELDS: list[str] = [
    "opp_id",             # 상대 차량 식별자 (Opponent['id'])
    "rel_x", "rel_y",     # 내 차 기준 상대 위치 (opp world_position - ego world_position)
    "distance",           # 수평면 유클리드 거리 sqrt(rel_x^2 + rel_y^2)
    "opp_speedKMH",       # 상대차 속도 (km/h)
    "opp_yaw",            # 상대차 진행 방향(라디안)
]


def extract_action(state: dict) -> dict:
    """state dict에서 사람 조작값(정답 라벨)만 canonical 이름으로 뽑는다.

    누락된 키는 None. 게임 연결 후 ACTION_MAP만 고치면 전체가 따라감.
    """
    return {canon: state.get(raw) for canon, raw in ACTION_MAP.items()}


def split_state(state: dict) -> tuple[dict, dict]:
    """state를 (ego_part, action_part)로 분리해서 반환.

    ego_part = action 원본 키를 제외한 나머지 전부(raw 보존 목적).
    action_part = canonical steer/throttle/brake.
    """
    action = extract_action(state)
    action_raw_keys = set(ACTION_MAP.values())
    ego = {k: v for k, v in state.items() if k not in action_raw_keys}
    return ego, action
