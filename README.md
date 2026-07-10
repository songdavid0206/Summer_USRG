# Assetto Corsa 데이터셋 취득 파이프라인

`dasGringuen/assetto_corsa_gym` 을 **참조**해, 사람이 직접 주행하는 동안
**이미지 + ego 차량정보 + 상대차 + 사람 조작값**을 timestamp로 동기화해
학습 가능한 dataset으로 저장하는 **취득 파이프라인**.

> 참조 레포는 게임 데이터를 받아 RL을 하는 코드다. 우리가 만드는 건 그 학습 코드가
> 아니라, 그 앞 단계인 **데이터셋 수집 시스템**이다. 모은 데이터는 이후 IL/RL에 쓴다.

## 현재 상태 (2026-07-10)

**게임 실주행 + AI 자동주행 수집 전 구간 동작.** 한 번의 `collect` 로 아래를 **모두 ~45-50Hz**로 취득:
`카메라 이미지(JPG) + ego(자차) + opponent(상대차) + 조작값 라벨 + track(맵/레이싱라인)`.
Assetto Corsa + sensors_par 플러그인(CSP 필요)에서 telemetry를, dxcam으로 화면을 캡처한다.
후처리(`processed/`)에서 **리타이어/핏 프레임을 마스킹**해 학습에 쓸 프레임만 고를 수 있다.

## 설계 핵심 — 방법 C (raw 따로 기록 → 오프라인 정렬)

telemetry와 image는 **각자 주기로** 들어온다(현재 둘 다 목표 50Hz). 실시간으로 묶으면
나중에 시각 어긋남을 검증/보정할 수 없다. 그래서 각자 timestamp를 달아 raw로 쌓아두고,
수집이 끝난 뒤 **카메라 프레임 기준으로 정렬**한다.

```
[AC 주행 (사람 또는 AI)]  ── 같은 시계(Clock) ──►  각 스트림을 각자 timestamp 달아 raw 저장
   telemetry(~50Hz) → telemetry_raw.parquet   (ego 전체 + steer/throttle/brake)  ← 플러그인 UDP 2345
                      opponents_raw.parquet    (t, opp_id, ...)  ※ 상대 있을 때만 ← 플러그인 TCP 2346
   image(~50Hz)     → image_index.parquet + camera/*.jpg                         ← 화면 캡처(dxcam)
                                   │
                                   ▼  dataset_builder (오프라인, merge_asof nearest)
              camera/ + ego_state/ + opponent/opp_<id>/   ← 스트림별 폴더로 분리
                                   +  track/  ← 트랙 맵 + 레이싱 라인 (fast_lane.ai 파싱)
```

- **정렬 기준 = 카메라 프레임.** 이미지 한 장마다 시간이 가장 가까운 telemetry tick을 붙인다
  (`merge_asof` nearest, tolerance 50ms). **모든 프레임을 유지**하므로
  `camera/{i}.jpg` ↔ `ego_state` 행 i ↔ 각 `opp_<id>` 행 i 가 **항상 1:1로 대응**하고
  세 폴더의 행 개수가 정확히 같다.
- **이미지는 우리가 직접 화면 캡처**(dxcam→mss 폴백). 한 프로세스·한 Clock이라
  image↔telemetry 동기화가 정확(둘 다 50Hz면 실측 dt≈0ms). JPG 저장(PNG보다 12배 빠름).
- **action(정답 라벨)은 제어를 내보내지 않고 telemetry에서 읽는다.** 주행을 방해하지
  않도록 state의 `steerAngle/accStatus/brakeStatus`를 canonical `steer/throttle/brake`로 라벨링.
  (AI 자동주행 시엔 AI의 조작값이 그대로 라벨 = IL 전문가 데이터)
- **opponent는 가변 개수**라 그 run에 등장한 `opp_id`마다 폴더를 하나씩 자동 생성(하드코딩 아님).
  상대 없는 세션이면 `opponent/` 자체가 생성되지 않는다.
- **track은 자동**: 주행 좌표로 설치된 트랙을 식별해 `fast_lane.ai`에서 레이싱 라인+맵을 뽑는다.

## 출력 구조

```
dataset/<run_id>/                     # run_id = 수집 시각 (YYYYMMDD_HHMMSS)
    camera/
        000000.jpg, 000001.jpg, ...   # 프레임별 이미지 (파일명 = frame 번호, JPG q95)
        camera_index.parquet / .csv   # frame, t_img, image_path
    ego_state/
        ego_state.parquet / .csv      # 프레임당 1행: 자차 state 전체 + steer/throttle/brake
    opponent/
        opp_<id>/opp_<id>.parquet/.csv  # 프레임당 1행: 그 상대차의 상대위치·속도 (present 플래그 포함)
    track/
        racing_line.parquet / .csv    # 레이싱 라인 + 트랙 폭(맵) + AI 속도 프로파일
        track.json                    # 트랙 이름, 점 개수, 식별 정확도
    processed/                        # E2E/IL 학습용 후처리 피처 (선별·가공)
        ego_features.parquet / .csv       # 선별된 자차 운동상태 + 라벨 + in_pit/valid 마스크
        opponent/opp_<id>_features.parquet/.csv  # 상대차별 절대좌표+ego-frame+속도 + retired/valid 마스크
        track_features.parquet / .csv     # 레이싱 라인 대비 오프셋/heading/곡률/경계거리
        quality.json                      # 유효 프레임 요약 (핏 구간, opp별 리타이어 시점)
    ── raw 소스(백업, 재정렬용) ──
    image_index.parquet               # frame ↔ t ↔ image_path
    telemetry_raw.parquet             # tick별 ego 전체 + steer/throttle/brake (고주파 원본)
    opponents_raw.parquet             # (t, opp_id, rel_x, rel_y, ...)  ※ optional
    meta.json                         # 수집 설정/요약 (실측 Hz 포함)
```

- 모든 데이터파일은 **parquet + csv 동시** 저장 (parquet=로딩용, csv=엑셀 확인용).
- raw 3종(`image_index`, `telemetry_raw`, `opponents_raw`)만 있으면 언제든 재정렬 가능: `ac-build dataset/<run_id>`
- 정렬된 폴더(camera/ego_state/opponent)는 결국 **image Hz로 캡핑**된다. telemetry가 더 높아도
  ego_state는 이미지에 맞춰진 rate이며, **고주파 원본은 `telemetry_raw.parquet`에 그대로 보존**된다.

---

## ego_state 필드 사전

자차(ego) 상태. **AC 공유메모리 → sensors_par 플러그인 ego_server(UDP 2345)** 로 tick마다 스트리밍되는
state(약 90필드)를 통째로 저장한 것 + 정답 라벨(steer/throttle/brake) + 정렬 컬럼(frame/t_img/t_tel/dt).
필드명은 참조 레포 `ac_env.py`의 `expand_state()` 기준. **일부 필드는 4바퀴 배열 `[FL, FR, RL, RR]`.**

### 정렬·식별·시간
| 필드 | 의미 |
|---|---|
| `frame` | 카메라 프레임 번호 (`camera/{frame:06d}.jpg` 와 대응) |
| `t_img` | 이 프레임(이미지) timestamp (수집 시작 기준 초) |
| `t_tel` | 붙은 telemetry tick의 timestamp (초) |
| `dt` | `|t_img − t_tel|` 시간차 (초). 0에 가까울수록 정렬 정확 |
| `timestamp_ac`, `currentTime`, `iLastTime` | AC 내부 시계 / 현재 랩타임 / 직전 랩타임(ms) |
| `packetId`, `steps`, `total_errors_out_of_sync` | 패킷 카운터 / 스텝 수 / 동기 오류 카운트 (디버깅용) |

### 조작 라벨 (정답 — IL 타깃)
| 필드 | 의미 |
|---|---|
| `steer` | 조향각 **도(degree) 단위**, 부호 = 좌/우. 원본 `steerAngle` (`ac.getCarState(id, CS.Steer)`) |
| `throttle` | 가속 페달 0~1, 원본 `accStatus` |
| `brake` | 브레이크 페달 0~1, 원본 `brakeStatus` |

> ⚠️ **`steer`는 [-1,1]로 정규화돼 있지 않다.** 관측 범위 예: −138.9°~+107.2° (비대칭은 그 주행의
> 편향일 뿐 차의 조향 한계가 아니다). 정규화하려면 **차종의 steer lock 상수**로 나눠야 하는데
> telemetry 90필드에 없다. run별 관측 min/max로 나누면 run마다 라벨 스케일이 어긋나므로 **틀린다.**
> 정보 손실 없는 순수 나눗셈이라 **차종 확정 후 학습 직전에** 상수로 처리하면 된다.

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

## 후처리 피처 (`processed/`)

raw(ego_state/opponent/track)를 **E2E/IL 학습 입력에 맞게 선별·가공**한 것. `collect` 종료 시 자동 생성되고
`ac-process dataset/<run>` 으로 수동 생성도 된다. **raw는 그대로 보존**되므로 언제든 재생성/변경 가능.
설계는 E2E 자율주행·레이싱 IL의 표준(운동상태 + 트랙상대 기하 + ego-centric)을 따른다.

### `ego_features` (프레임당 1행) — 자차
운동상태 중심으로 선별(타이어/서스펜션/휠하중 등 **동역학 디테일은 제외**, 필요 시 `telemetry_raw`에서 추출).
- 키: `frame, t_img`
- 운동: `speed`, `velocity_x/y/z`, `local_velocity_x/y/z`(차체기준), `accelX/Y`, `angular_velocity_x/y/z`, `yaw_rate`(=angular_velocity_y)
- 자세: `yaw, pitch, roll` / 위치: `world_position_x/y/z`, `NormalizedSplinePosition`
- 상태: `RPM, actualGear, LapCount, numberOfTyresOut, LapInvalidated, isInPit`
- 마스크: `in_pit`(=isInPit>0), `valid`(=~in_pit) — 아래 "유효 프레임 마스크" 참조
- 라벨: `steer, throttle, brake`

### `opponent/opp_<id>_features` (상대차별 파일, 프레임당 1행) — 상대차
raw `opponent/opp_<id>/` 와 동일하게 **상대차마다 파일 하나**. camera/ego_state 와 프레임 정렬(행 i = 프레임 i).
| 필드 | 의미 |
|---|---|
| `frame, t_img, opp_id, present` | 정렬 키 + 존재여부 |
| `retired, valid` | 이 차가 이미 핏으로 회수됐는지 / 이 프레임에 쓸 수 있는지(=present & ~retired) |
| `opp_x, opp_y` | **절대 좌표** (= ego world_position + rel). BEV/궤적 분석용 |
| `opp_forward, opp_lateral` | **ego-frame** 상대위치 (자차 진행방향 기준 앞/옆 거리). 모델 입력 표준 |
| `distance` | 자차-상대차 수평거리 (m) |
| `opp_vx, opp_vy, opp_speed` | 상대차 속도벡터·속력. **플러그인 실측 속도(SpeedMS) + 진행방향(opp_yaw)** 으로 구성 |
| `rel_vx, rel_vy, closing_speed` | 자차 대비 상대속도 / 접근속도(+면 가까워짐) |
| `opp_yaw, opp_speedKMH` | 상대차 방향·속도(플러그인 원본) |

> **속도는 위치 시간차분을 쓰지 않는다.** 상대차 위치는 ~47Hz로 갱신되지만 드물게(~0.5%) stale/held
> 프레임이 있어 시간차분이 0↔폭주로 튄다(대부분의 중복은 차가 실제 정지한 구간이라 정상). 그래서 시간차분
> 대신 플러그인 실측 속도 + 진행방향(위치기반 방향과 0.2도 일치 확인)으로 속도벡터를 만든다 → 물리적으로
> 정확·글리치 없음. (opp_id는 세션 내내 고정 = 같은 차.)

### `track_features` (프레임당 1행) — 레이싱 라인 대비
| 필드 | 의미 |
|---|---|
| `frame, t_img` | 정렬 키 |
| `lateral_offset` | 레이싱 라인에서 벗어난 횡거리 (부호 +=왼쪽) |
| `heading_error` | 자차 방향 − 라인 접선 방향 (rad) |
| `curvature` | 현재 지점 곡률 (1/m, 클수록 급코너) |
| `dist_left_edge, dist_right_edge` | 좌/우 트랙 경계까지 거리 (m) |

### 유효 프레임 마스크 — 리타이어 / 핏 순간이동

AC는 차가 **리타이어하거나 세션이 끝나면 그 차를 핏 박스로 순간이동시킨 뒤 거기 고정**한다.
그래도 플러그인은 그 차를 계속 보내므로 `present`는 True로 남는다 — 즉 **`present`는 리타이어
플래그가 아니다.** 이대로 두면 카메라에는 보이지도 않는 차가 "내 뒤 20m"에 있는 것처럼 피처에
남아 학습을 오염시킨다(핏 박스들은 10m 간격으로 나란히 놓여 있다). 자차도 세션 종료 시 같은
방식으로 개러지에 되돌아간다(주행 중 86m/s → 다음 프레임 speed 0, `isInPit=1`).

그래서 후처리가 이를 검출해 **bool 마스크 컬럼**으로 표시한다.

| 컬럼 | 파일 | 의미 |
|---|---|---|
| `in_pit` / `valid` | `ego_features` | 자차가 핏에 있는 프레임 → 주행 데이터 아님 |
| `retired` / `valid` | `opp_<id>_features` | 그 상대차가 이미 핏 박스에 있음 → 그 차만 제외 |

- **행은 지우지 않는다.** `row i ↔ camera/{i:06d}.jpg` 1:1 대응이 데이터셋 전체의 전제라, 행을
  삭제하면 이미지 인덱스가 어긋난다. 값도 그대로 두고 마스크만 추가한다(raw 보존 원칙).
- **검출 규칙**(`_detect_retirement`): ① 한 프레임에 10m 넘게 이동(=순간이동) **그리고**
  ② 그 뒤 끝까지 그 자리에 고정(1m 이내, 속도<1km/h). **두 조건이 모두** 필요하다.
  "멈춤"만으로 판정하면 **크래시로 트랙에 서 있는 차까지 지워버리는데**, 그 차는 카메라에
  실제로 보이므로 오히려 남겨야 할 장애물 데이터다. 10m 임계값은 45Hz·300km/h의 프레임당
  이동(1.9m)과 위치 held를 감안한 실측 최대(3.4m)의 3배 여유.
- **`quality.json`** 에 요약이 남는다: 총 프레임 수, 자차 핏 진입 프레임/시각,
  상대차별 `retired_frame / retired_t / teleport_m / pit_box_xy / valid_frames`.
- 마스킹은 `processed/` 에만 있다. `telemetry_raw` / `opponents_raw` / `ego_state` / `opponent/`
  (raw·정렬층)에는 없다. 임계값을 바꾸고 싶으면 재수집 없이 `ac-process` 만 다시 돌리면 된다.

**학습 시 사용법**
```python
ego = pd.read_parquet("dataset/<run>/processed/ego_features.parquet")
opp = pd.read_parquet("dataset/<run>/processed/opponent/opp_2_features.parquet")

frames = ego.loc[ego.valid, "frame"].to_numpy()   # 자차가 주행 중인 프레임만
# 그 프레임에서 실제 트랙 위에 있는 상대차만 (차마다 따로 판정)
opp_now = opp[opp.frame.isin(frames) & opp.valid]
```

---

## 패키지 구조 / 파일

```
Summer_USRG/
    pyproject.toml                # 패키지 정의 + 의존성 + CLI 진입점
    README.md
    src/ac_dataset/
        __init__.py               # 공개 API: collect, build_dataset, extract_track_for_run, build_processed
        collect.py                # 수집 메인 → 종료 시 정렬/track/후처리까지 자동
        schema.py                 # ego/action/opponent 필드 정의 + 추출 헬퍼 (필드명 single source)
        track.py                  # 트랙 자동식별 + fast_lane.ai 파싱 → track/ (맵+레이싱라인)
        processed_builder.py      # raw → processed/ (E2E/IL용 선별·가공 피처)
        view.py                   # parquet 미리보기/CSV 뷰어
        recorder/
            telemetry_logger.py   # ego+action+opponents raw 기록 (AC 플러그인 state)
            ac_telemetry.py       # ego_server(UDP 2345)에서 자차 state 수신
            ac_opponents.py       # opponents_server(TCP 2346)에서 상대차 수신
            screen_capture.py     # 화면 캡처 (dxcam 우선 → mss 폴백, region 지정 가능)
            image_logger.py       # 캡처 프레임을 camera/ 에 JPG + index로 저장
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
설치하면 아래 명령들을 **어느 폴더에서든** 쓸 수 있다: `collect`, `ac-build`, `ac-view`, `ac-track`, `ac-process`.

## 사용법

**실제 수집 (AC + 플러그인 + CSP 실행 후, 트랙에 진입한 상태)**
```bash
collect --capture-region 0,80,1920,945   # 기본 telemetry 50Hz / image 50Hz(JPG), 지정 영역만 캡처
# 주행(사람 또는 AI) → Alt+Tab 으로 터미널 이동 → Ctrl+C 로 종료 (게임이 키보드를 잡으므로 Alt+Tab 필수)
# 종료 시 자동으로 camera/ ego_state/ opponent/ track/ 폴더 + parquet + csv 생성
```
- **HUD는 게임에서 꺼두되, `--capture-region`으로 남는 UI 요소(상단 배너/하단 스트립 등)까지 잘라낸다.**
  값 `L,T,W,H`는 게임 해상도/화면에 맞게 조정(예: 1920×1080에서 위 80px·아래 55px 잘라내면 `0,80,1920,945`).
- `--telemetry-hz`, `--image-hz` : 목표 주파수(기본 둘 다 50). `--image-format jpg|png`.
- **AI 자동주행 수집**: 트랙 주행 중 게임 창에서 `Ctrl+C` → AC 내장 AI가 대신 운전. 그 상태로 `collect` 하면
  AI 조작값이 라벨로 저장돼 IL 전문가 데이터가 된다. (게임 Ctrl+C = AI토글, 터미널 Ctrl+C = 수집종료)

> 코드로 부르려면: `from ac_dataset import collect; collect(out_dir="./dataset")`

**트랙/후처리 (재)생성** (`collect` 시 자동 생성되지만 수동으로도)
```bash
ac-track   dataset/<run_id>          # 주행 좌표로 트랙 자동 식별 → track/ 생성
ac-process dataset/<run_id>          # raw → processed/ (E2E/IL용 선별·가공 피처)
```

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

## 수집 주파수(Hz) — 실측 결과와 한계

목표 Hz는 `--telemetry-hz`, `--image-hz`로 준다(기본 둘 다 50). **수집 종료 시 `[rate]`로 실측 Hz가 출력**되고
`meta.json`에도 저장된다. 각 스트림의 실제 한계:

| 스트림 | 실측 한계 | 무엇이 막나 |
|---|---|---|
| **telemetry(ego)** | ~90Hz 수신 / **기록은 루프 속도** | 플러그인은 ~90Hz로 보내준다(중복 0%). 기록 rate는 아래 루프 병목에 묶임 |
| **image** | ~45Hz | 캡처(dxcam) + JPG 인코딩·저장. 실측 q95 1920×945 = **9.6ms/장**. PNG였으면 ~5Hz(176ms) |
| **opponents** | ~50Hz | 플러그인 `sensors_par.py`의 opponents 송신 `time.sleep()` 값. 낮출수록↑(0.008≈50Hz), 단 AC 구형 파이썬 불안정 위험 |

- **단일 루프 lock-step**: collect는 아직 telemetry와 image가 **한 루프**에 있고 둘 다 50Hz로
  스케줄되므로 **항상 같은 반복에서 함께** 발화한다. 그래서 `[rate]` 출력의 두 값이 소수점까지
  같으면(예: `45.3 / 45.3`) 루프가 목표 주기를 못 지킨다는 뜻이다.
  실측 루프 1바퀴 = **22.07ms** > 목표 20ms → `1/0.02207 = 45.3Hz`. 스케줄러가 catch-up을 하지
  않으므로 **한 번 밀리면 실측 Hz = 루프 속도**가 된다.
  - 이때 병목은 **telemetry가 아니다**: 플러그인에서 89.8Hz로 들어오고 기록된 행의 `packetId`가
    전부 고유(중복 0%) — 게임 틱의 절반을 버리는 중이다. 범인은 이미지 경로(인코딩+디스크).
  - `--image-hz`를 과하게 올리면(예 100) 이미지가 루프를 잡아 telemetry까지 더 끌어내린다.
  - 50Hz를 확실히 지키려면 **이미지 캡처/저장을 별도 스레드+큐로 분리**해야 한다(미구현).
    분리하면 현재 `dt≈0`(둘이 같은 반복에서 찍힘)이 최대 10ms로 벌어지지만 `merge_asof`가 처리한다.
- **원격 데스크톱(RDP)에서는 ~33Hz**로 떨어진다. dxcam이 실제 GPU 출력 대신 RDP 화면을 잡기 때문.
  **로컬(그 PC 앞)에서 수집할 것.**
- **용량**: image 45Hz JPG(q95, 1920×945) ≈ 초당 32MB (686KB/장). 대량 수집 시 품질(q85 → 8.4ms,
  파일 ~30%↓) 조절.

## 참고 / 주의

- **HUD**: 게임에서 HUD를 꺼도 일부 UI(상단 배너·하단 스트립 등)가 남으므로, `--capture-region`으로 그 영역까지 잘라낸다. (속도계/맵이 이미지에 찍히면 학습이 편법으로 흐름)
- **캡처 세팅**: AC를 테두리 없는 창(borderless windowed)으로 실행 권장(전체화면 독점은 캡처가 검게 나올 수 있음).
- **raw 카메라**: AC는 렌더 이미지를 API로 주지 않음 → **화면 캡처가 유일**. 대신 숫자로 **BEV(탑다운) 이미지**는 생성 가능.
- **좌표축**: ego world_position 수평면은 (x, y), z=높이. AI 파일(fast_lane.ai)은 축 순서가 달라 `ego(x,y,z)=AI(z,x,y)`로
  변환해 저장(track.py). 등수 등 진행방향 판정 시엔 ego 위치 변화로 heading을 구해 투영하는 방식이 안전.
