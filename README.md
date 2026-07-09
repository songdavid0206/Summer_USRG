# Assetto Corsa 데이터셋 취득 파이프라인

`dasGringuen/assetto_corsa_gym` 을 **참조**해, 사람이 직접 주행하는 동안
**이미지 + ego 차량정보 + 상대차 + 사람 조작값**을 timestamp로 동기화해
학습 가능한 dataset으로 저장하는 **취득 파이프라인**.

> 참조 레포는 게임 데이터를 받아 RL을 하는 코드다. 우리가 만드는 건 그 학습 코드가
> 아니라, 그 앞 단계인 **데이터셋 수집 시스템**이다. 모은 데이터는 이후 IL/RL에 쓴다.

## 설계 핵심 — 방법 C (raw 따로 기록 → 오프라인 정렬)

이미지와 telemetry는 **주기가 다르다**(telemetry 50Hz / image 20Hz). 실시간으로 묶으면
나중에 시각 어긋남을 검증/보정할 수 없다. 그래서 각자 timestamp를 달아 raw로 쌓아두고,
수집이 끝난 뒤 **카메라 프레임 기준으로 정렬**한다.

```
[사람이 AC 주행]  ── 같은 시계(Clock) ──►  두 스트림을 각자 timestamp 달아 raw 저장
   telemetry(50Hz) → telemetry_raw.parquet   (ego 전체 + steer/throttle/brake)   ← AC 플러그인 UDP 2345
                     opponents_raw.parquet    (t, opp_id, ...)  ※ 상대 있을 때만  ← AC 플러그인 TCP 2346
   image(20Hz)     → image_index.parquet + camera/*.png                          ← 화면 캡처(dxcam/mss)
                                   │
                                   ▼  recorder.dataset_builder (오프라인, merge_asof nearest)
              camera/ + ego_state/ + opponent/opp_<id>/   ← 스트림별 폴더로 분리한 최종 dataset
```

- **정렬 기준 = 카메라 프레임.** 이미지 한 장마다 시간이 가장 가까운 telemetry tick을 붙인다
  (`merge_asof` nearest, tolerance 50ms). **모든 프레임을 유지**하므로
  `camera/{i}.png` ↔ `ego_state` 행 i ↔ 각 `opp_<id>` 행 i 가 **항상 1:1로 대응**하고
  세 폴더의 행 개수가 정확히 같다.
- **이미지는 우리가 직접 화면 캡처**(dxcam→mss 폴백)한다. 한 프로세스·한 Clock이라
  image↔telemetry 동기화가 정확(실측 dt≈0ms).
- **action(정답 라벨)은 제어를 내보내지 않고 telemetry에서 읽는다.** 사람 주행을 방해하지
  않도록 state의 `steerAngle/accStatus/brakeStatus`를 canonical `steer/throttle/brake`로 라벨링.
- **opponent는 가변 개수**라 그 run에 등장한 `opp_id`마다 폴더를 하나씩 자동 생성(하드코딩 아님).
  상대 없는 세션이면 `opponent/` 자체가 생성되지 않는다.

## 출력 구조

```
dataset/<run_id>/                     # run_id = 수집 시각 (YYYYMMDD_HHMMSS)
    camera/
        000000.png, 000001.png, ...   # 프레임별 이미지 (파일명 = frame 번호)
        camera_index.parquet / .csv   # frame, t_img, image_path
    ego_state/
        ego_state.parquet / .csv      # 프레임당 1행: 자차 state 전체 + steer/throttle/brake
    opponent/
        opp_<id>/opp_<id>.parquet/.csv  # 프레임당 1행: 그 상대차의 상대위치·속도 (present 플래그 포함)
    ── raw 소스(백업, 재정렬용) ──
    image_index.parquet               # frame ↔ t ↔ image_path
    telemetry_raw.parquet             # tick별 ego 전체 + steer/throttle/brake
    opponents_raw.parquet             # (t, opp_id, rel_x, rel_y, ...)  ※ optional
    meta.json                         # 수집 설정/요약
```

- 모든 데이터파일은 **parquet + csv 동시** 저장 (parquet=로딩용, csv=엑셀 확인용).
- raw 3종(`image_index`, `telemetry_raw`, `opponents_raw`)만 있으면 언제든 재정렬 가능:
  `python -m recorder.dataset_builder dataset/<run_id>`

---

## ego_state 필드 사전

자차(ego) 상태. **AC 공유메모리 → sensors_par 플러그인 ego_server(UDP 2345)** 로 tick마다 스트리밍되는
state(약 90필드)를 통째로 저장한 것 + 정답 라벨(steer/throttle/brake) + 정렬 컬럼(frame/t_img/t_tel/dt).
필드명은 참조 레포 `ac_env.py`의 `expand_state()` 기준. **일부 필드는 4바퀴 배열 `[FL, FR, RL, RR]`.**

### 정렬·식별·시간
| 필드 | 의미 |
|---|---|
| `frame` | 카메라 프레임 번호 (`camera/{frame:06d}.png` 와 대응) |
| `t_img` | 이 프레임(이미지) timestamp (수집 시작 기준 초) |
| `t_tel` | 붙은 telemetry tick의 timestamp (초) |
| `dt` | `|t_img − t_tel|` 시간차 (초). 0에 가까울수록 정렬 정확 |
| `timestamp_ac`, `currentTime`, `iLastTime` | AC 내부 시계 / 현재 랩타임 / 직전 랩타임(ms) |
| `packetId`, `steps`, `total_errors_out_of_sync` | 패킷 카운터 / 스텝 수 / 동기 오류 카운트 (디버깅용) |

### 조작 라벨 (정답 — IL 타깃)
| 필드 | 의미 |
|---|---|
| `steer` | 조향 (부호 = 좌/우), 원본 `steerAngle` |
| `throttle` | 가속 페달 0~1, 원본 `accStatus` |
| `brake` | 브레이크 페달 0~1, 원본 `brakeStatus` |

### 위치·자세 (월드 좌표)
| 필드 | 의미 |
|---|---|
| `world_position_x` / `_y` / `_z` | 월드 3D 위치(m). ※ 상대차 rel_x/rel_y는 x,y 차이로 계산됨 |
| `NormalizedSplinePosition` | 트랙 진행도 0~1 (한 바퀴 기준 위치) |
| `yaw` / `pitch` / `roll` | 차체 방향각 (라디안) |
| `cgHeight` | 무게중심 높이(m) |

### 속도·가속
| 필드 | 의미 |
|---|---|
| `speed` | 속력 (m/s) |
| `velocity_x` / `_y` / `_z` | 월드 좌표 속도 벡터(m/s). 수평면 = (x, z), y=수직 |
| `local_velocity_x` / `_y` / `_z` | 차체 기준 속도(m/s). `local_velocity_x`≈전진속도 |
| `accelX` / `accelY` | 종/횡 가속도 (G) |
| `angular_velocity_x` / `_y` / `_z` | 각속도 (rad/s), y=요레이트 |

### 엔진·기어
| 필드 | 의미 |
|---|---|
| `RPM` | 엔진 회전수 |
| `actualGear` | 현재 기어 |
| `LastFF` | 포스피드백 값 (스티어링 부하 지표) |

### 랩·세션 상태
| 필드 | 의미 |
|---|---|
| `LapCount` / `completedLaps` | 현재 랩 / 완주한 랩 수 |
| `BestLap` | 베스트 랩타임(ms, 0=미기록) |
| `LapInvalidated` | 현재 랩 무효 여부(코스이탈 등) |
| `isInPit` | 피트 안 여부 |
| `numberOfTyresOut` | 코스 밖으로 나간 바퀴 수(0~4) |
| `penaltyTime` | 패널티 시간(초) |

### 타이어 — 슬립/그립
| 필드 | 의미 |
|---|---|
| `SlipAngle_fl/fr/rl/rr` | 바퀴별 슬립각 (rad) |
| `tyre_slip_ratio_fl/fr/rl/rr` | 바퀴별 슬립율 (구동/제동 슬립) |
| `NdSlip` `[FL,FR,RL,RR]` | 정규화 슬립 (그립 여유) |
| `Dy_fl/fr/rl/rr` | 바퀴별 횡력 계수 |
| `Mz` `[FL,FR,RL,RR]` | 바퀴별 셀프얼라이닝 토크 |
| `CamberRad` `[FL,FR,RL,RR]` | 캠버각 (rad) |

### 타이어 — 압력/온도/하중/기하
| 필드 | 의미 |
|---|---|
| `fl/fr/rl/rr_tire_pressure`, `dynamic_pressure[4]` | 바퀴별 타이어 압력(psi) |
| `fl/fr/rl/rr_tire_temperature_core` | 바퀴별 코어 온도(°C) |
| `fl/fr/rl/rr_wheel_load`, `tyres_load[4]` | 바퀴별 수직 하중(N) |
| `tyre_loaded_radius[4]` | 하중 받은 타이어 반경(m) |
| `wheel_speed_fl/fr/rl/rr` | 바퀴 회전 속도(rad/s) |

### 서스펜션·접지
| 필드 | 의미 |
|---|---|
| `fl/fr/rl/rr_damper_linear_potentiometer`, `SuspensionTravel[4]` | 바퀴별 서스펜션 스트로크(m) |
| `tyreContactHeading_FL/FR/RL/RR_x/y/z` | 바퀴별 타이어 접지면 방향 벡터 |

> ego_state = `telemetry_raw`(플러그인 state 원본 전체) + 정렬 컬럼. 위 목록에 없는 채널이
> 플러그인에 추가되면 자동으로 함께 저장된다(raw 보존). 조작값(steerAngle/accStatus/brakeStatus)은
> 원본 키 대신 canonical `steer/throttle/brake` 로만 남긴다.

---

## opponent 필드 사전

상대차 정보. **opponents_server(TCP 2346)** 가 세션 내 각 상대차의
`id / world_position / speedKMH / yaw` 만 스트리밍한다(AC가 타차의 조작값은 주지 않음).
이를 **ego 기준 상대좌표**로 변환해 저장. `opp_<id>/` 폴더의 파일은 **프레임당 1행**으로,
그 프레임에 그 상대차가 없으면 `present=False` + 값은 NaN (모든 opp 폴더 길이를 camera와 맞추기 위함).

| 필드 | 의미 |
|---|---|
| `frame`, `t_img`, `t_tel` | 카메라 프레임 정렬 컬럼 (ego_state와 동일 축) |
| `present` | 이 프레임에 이 상대차가 존재했는지 (True/False) |
| `rel_x`, `rel_y` | 내 차 기준 상대 위치 = (상대 world_position − ego world_position)의 x, y |
| `distance` | 수평 유클리드 거리 `sqrt(rel_x² + rel_y²)` (m) |
| `opp_speedKMH` | 상대차 속도 (km/h) |
| `opp_yaw` | 상대차 진행 방향 (rad) |

- 폴더명 `opp_<id>` 의 `<id>` = 플러그인 `Opponent['id']` (그 run에 나온 값 그대로, 예: 1~4).
- 상대차 상세 채널(속도/기어/RPM 등)을 늘리려면 플러그인 `Opponent.update()` 에 채널 추가 가능
  (단, 타차 조작값은 AC가 제공하지 않아 불가).

---

## 패키지 구조 / 파일

```
Summer_USRG/
    pyproject.toml                # 패키지 정의 + 의존성 + CLI 진입점
    README.md
    src/ac_dataset/
        __init__.py               # 공개 API: collect, build_dataset
        collect.py                # 주행 중 두 로거 동시 실행 → 종료 시 자동 정렬
        schema.py                 # ego/action/opponent 필드 정의 + 추출 헬퍼 (필드명 single source)
        view.py                   # parquet 미리보기/CSV 뷰어
        recorder/
            telemetry_logger.py   # ego+action+opponents raw 기록 (AC 플러그인 state)
            ac_telemetry.py       # ego_server(UDP 2345)에서 자차 state 수신
            ac_opponents.py       # opponents_server(TCP 2346)에서 상대차 수신
            screen_capture.py     # 화면 캡처 (dxcam 우선 → mss 폴백, region 지정 가능)
            image_logger.py       # 캡처 프레임을 camera/ 에 PNG + index로 저장
            dataset_builder.py    # raw → camera/ ego_state/ opponent/ 폴더로 정렬·분리
    tests/test_offline.py         # 게임 없이 서로 다른 주기 raw 생성 → 정렬 검증
    dataset/                      # 수집 결과 (git 미추적)
```

## 설치

```bash
git clone https://github.com/songdavid0206/Summer_USRG.git
cd Summer_USRG
pip install -e .            # 의존성 자동 설치 + collect / ac-build / ac-view 명령 생성
pip install -e ".[capture]" # (선택) Windows에서 더 빠른 dxcam 캡처 백엔드까지
```
설치하면 아래 명령들을 **어느 폴더에서든** 쓸 수 있다: `collect`, `ac-build`, `ac-view`.

## 사용법

**실제 수집 (AC + 플러그인 + CSP 실행 후, 트랙에 진입한 상태)**
```bash
collect                       # 기본값 telemetry 50Hz / image 20Hz, 주 모니터 전체 캡처
# 게임 창만 잡기: collect --capture-region L,T,W,H
# 주행 → Alt+Tab 으로 터미널 이동 → Ctrl+C 로 종료 (게임이 키보드를 잡으므로 Alt+Tab 필수)
# 종료 시 자동으로 camera/ ego_state/ opponent/ 폴더 + parquet + csv 생성
```
> 코드로 부르려면: `from ac_dataset import collect; collect(out_dir="./dataset")`

**기존 raw run을 새 폴더 구조로 (재)정렬**
```bash
ac-build dataset/<run_id> --tolerance 0.05
```

**데이터 확인**
- 엑셀: 각 폴더의 `.csv` 를 바로 열기 (`ego_state/ego_state.csv` 등).
- 터미널: `ac-view dataset/<run_id>/ego_state/ego_state.parquet --csv`

**오프라인 테스트 (게임 불필요)**
```bash
python tests/test_offline.py
```

## 참고 / 주의

- **HUD**: 속도계/맵이 이미지에 찍히면 학습이 편법으로 흐를 수 있음 → 게임 HUD 끄거나 region으로 잘라내기.
- **캡처 세팅**: AC를 테두리 없는 창(borderless windowed)으로 실행 권장(전체화면 독점은 캡처가 검게 나올 수 있음).
- **수집 속도**: 현재 실측 ~5Hz. 원인은 dxcam 전체모니터 캡처가 메인루프를 블로킹 → `--capture-region`으로
  게임창만 잡거나 이미지 캡처를 별도 스레드로 분리하면 개선.
- **좌표축**: 속도 벡터의 수평면은 (x, z)이고 y=수직. 상대차 rel_x/rel_y는 위치 x,y 차이로 계산된 값이라,
  등수 등 진행방향 판정 시에는 ego 위치 변화로 heading을 구해 투영하는 방식이 안전.
