# Summer_USRG — Assetto Corsa E2E 데이터셋 파이프라인

Assetto Corsa 주행에서 **카메라 이미지, ego telemetry, 조작 라벨, 상대차 3D 상태와 트랙 정보**를
동기화해 E2E imitation learning 및 autonomous racing 연구용 데이터셋으로 만드는 파이프라인이다.

> [`dasGringuen/assetto_corsa_gym`](https://github.com/dasGringuen/assetto_corsa_gym)을 기반으로
> telemetry 인터페이스를 구성했으며, 이 저장소는 학습 알고리즘보다 앞 단계인 **데이터 취득·동기화·
> 후처리·품질검증**에 집중한다.

## 문서 안내

| 문서 | 먼저 읽을 때 |
|---|---|
| **이 README** | 프로젝트 목적, 빠른 실행, 데이터 구조와 설계 이해 |
| **[SETUP.md](SETUP.md)** | 새 PC 설치, Content Manager 화면, 카메라·CSP·플러그인 설정 |
| [Summer_USRG_Validation](https://github.com/songdavid0206/Summer_USRG_Validation) | 3D bbox·BEV·물리 일관성 검증 도구 사용 |

## 목차

- [빠른 시작](#빠른-시작)
- [현재 상태](#현재-상태-2026-07-21)
- [파이프라인 구조](#파이프라인-구조)
- [출력 구조](#출력-구조)
- [데이터 필드 사전](#데이터-필드-사전)
- [프레임별 txt export](#프레임별-txt-export)
- [설치와 CSP 브리지](#설치)
- [명령어](#사용법)
- [성능과 수집 주파수](#성능과-수집-주파수)

## 빠른 시작

전체 설치와 Content Manager 클릭 순서는 [SETUP.md](SETUP.md)에 있다. 이미 환경이 설치된 PC에서는
다음 순서만 따르면 된다.

```powershell
# 1) Assetto Corsa에서 RACE 세션 진입
# 2) sensors_par와 AC Dataset Camera Pose 앱 활성화
# 3) Summer_USRG 저장소 루트에서 수집
collect --capture-region 0,80,1920,945 --telemetry-hz 100 --image-hz 40 --onboard-camera 0

# 4) 종료 후 기본 무결성 검사
ac-validate dataset/<run_id>

# 5) 3D bbox와 물리 일관성 확인
av-check dataset/<run_id>
av-view dataset/<run_id> --frame 0
```

정상 시작 로그에는 `telemetry 연결 OK`, `opponents 연결 OK`, `AC Graphics 공유메모리 = 연결`,
`CSP 실제 카메라 pose = 연결`이 모두 보여야 한다.

## 현재 상태 (2026-07-21)

**게임 실주행·AI 자동주행에서 수집부터 검증용 export까지 전 구간이 동작한다.** 한 번의
`collect`로 다음을 함께 취득한다.

- dxcam 화면 이미지(JPG), ego 전체 telemetry, `steer/throttle/brake` 조작 라벨
- 상대차 telemetry와 각 렌더 프레임의 상대차 full 6DoF pose·visual AABB
- CSP가 실제로 렌더한 카메라의 position/forward/up/FOV
- AC graphics packet·게임시간·랩 정보, 트랙 맵·레이싱 라인

현재 권장 운용값은 **telemetry 100 Hz / image 40 Hz**다. 이미지 캡처·JPG 인코딩은 별도
스레드에서 처리하고, 종료 후 25 ms 균일 격자로 export한다. `ac-validate`로 구조와 무결성을,
별도 `ac_validation` 패키지의 `av-check`·`av-view`로 물리 일관성과 3D 재투영을 검사한다.

새 run의 3D bbox는 **사진 속 차량을 검출하거나 픽셀에 맞춰 만든 결과가 아니다.** CSP가 한 번의
`script.update`에서 제공한 카메라 pose와 차량 pose/AABB만으로 8개 월드 좌표 꼭짓점을 만들고
카메라 intrinsic으로 투영한다. 이미지는 결과가 실제 렌더와 맞는지 사람이 독립적으로 확인하는
검증 수단이다.

### 대표 실측 run

2026-07-20 Laguna Seca run(`20260720_171110`) 기준:

| 항목 | 결과 |
|---|---:|
| 수집 시간 | 79.99 s |
| telemetry | 7,999 ticks, 실측 100.0 Hz |
| raw image | 3,235장, 실측 40.44 Hz |
| 실제 카메라 pose | 3,235 / 3,235 (100%) |
| same-render 차량 상태 | 3,235 / 3,235 (100%), 19,410행 |
| build 유효 프레임 | 3,200 |
| 40 Hz export | 3,197프레임, 중복 0 / 빈 격자 3 |

## 파이프라인 구조

### 설계 핵심 — 독립 raw 기록 → AC 게임시간/렌더 기준 오프라인 정렬

telemetry와 image는 서로 다른 주기로 들어오므로 먼저 각각 원본을 보존하고, 수집 종료 후
**카메라 프레임을 기준으로** 정렬한다. 새 run은 PC wall-clock의 가장 가까운 샘플을 단순히 고르는
방식이 아니라 AC 게임시간을 공통 축으로 사용한다.

```
[Assetto Corsa / CSP]
  sensors_par UDP 2345 ── ego telemetry 100 Hz ──► telemetry_raw.parquet
  opponents TCP 2346 ─── opponent telemetry ─────► opponents_raw.parquet
  AC Graphics SHM ─────── packetId/game time ────┐
  CSP Lua bridge ───────── camera pose + cars ────┼─► image_index/render_car_state
  dxcam ────────────────── image 40 Hz ───────────┘
                                                    │
                 game-time interpolation + packet/render relation matching
                                                    ▼
       camera/ + ego_state/ + opponent/ + track/ + processed/ + export
```

- **공통 시간축:** 이미지의 `game_time_ms`와 telemetry의 `timestamp_ac`를 랩 누적시간으로
  바꾼 뒤 선형 보간한다. 각도는 `sin/cos`로 보간해 ±π 경계 점프를 피한다.
- **같은 렌더 확인:** AC Graphics `packetId`와 CSP `render_frame` 사이의 안정적인 선형 관계를
  run 중 학습한다. 게임시간 후보 중 관계 오차가 가장 작은 CSP slot을 선택하고
  `cam_packet_render_error` 등 진단값을 저장한다.
- **프레임 정책:** 허용 오차 밖의 미매칭 이미지는 제외한 뒤 프레임을 다시 0부터 번호 매긴다.
  따라서 최종 `camera/{i}.jpg`, `ego_state`의 i행, 각 `opp_<id>`의 i행은 항상 1:1이다.
- **구 run 호환:** AC 게임시간이 없는 예전 run만 wall-clock `merge_asof(nearest)`로 fallback한다.
- **캡처 분리:** dxcam→mss 폴백과 JPG 인코딩은 별도 스레드라 100 Hz telemetry 수집을 막지 않는다.
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
        camera_index.parquet / .csv   # 이미지·게임시간·실제 렌더 카메라 pose/FOV·동기 진단값
        render_car_state.parquet/.csv # 같은 렌더 tick의 차량 full 6DoF pose + visual AABB
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
    render_car_state.parquet          # CSP ring에서 받은 렌더 차량 상태 원본
    meta.json                         # 수집 설정/요약 (실측 Hz 포함)
```

- 모든 데이터파일은 **parquet + csv 동시** 저장 (parquet=로딩용, csv=엑셀 확인용).
- raw 3종(`image_index`, `telemetry_raw`, `opponents_raw`)만 있으면 언제든 재정렬 가능: `ac-build dataset/<run_id>`
- 정렬된 폴더(camera/ego_state/opponent)는 결국 **image Hz로 캡핑**된다. telemetry가 더 높아도
  ego_state는 이미지에 맞춰진 rate이며, **고주파 원본은 `telemetry_raw.parquet`에 그대로 보존**된다.

---

## 데이터 필드 사전

GitHub에서는 아래 영역을 접어 두었다. 필요한 데이터 종류를 확인할 때 펼치면 된다.

<details>
<summary><strong>카메라, ego, opponent, processed 필드 전체 보기</strong></summary>

## 카메라·same-render 차량 필드

`camera/camera_index.parquet`에는 일반 이미지 인덱스 외에 다음 렌더 진단값이 저장된다.

| 필드 | 의미 |
|---|---|
| `game_time_ms`, `graphics_packet`, `completed_laps` | 이미지 캡처 시 AC Graphics 공유메모리 상태 |
| `cam_pos_x/y/z` | CSP가 제공한 실제 렌더 카메라 월드 위치(m) |
| `cam_forward_x/y/z`, `cam_up_x/y/z` | 실제 카메라 전방·상방 단위벡터(extrinsic 회전) |
| `cam_fov_deg` | 해당 렌더 프레임의 수직 FOV. 해상도와 함께 intrinsic 계산에 사용 |
| `cam_render_frame`, `cam_game_time_ms` | 선택된 CSP 렌더 slot의 프레임·게임시간 |
| `cam_packet_render_error` | 학습된 graphics-packet/render-frame 관계에 대한 잔차 |
| `cam_pose_game_delta_ms` | 이미지와 CSP slot 사이 게임시간 오차 |

`camera/render_car_state.parquet`는 `frame`, `car_id`별로 같은 CSP callback에서 읽은 차량
`position/look/up`과 visual AABB 최소·최대점을 담는다. 좌표는 저장 시 AC의 축을 데이터셋 표준
`(x, y, z) = (AC z, AC x, AC y)`로 변환한다. 카메라와 차량이 같은 렌더 상태에서 나오므로
telemetry 서버 지연이나 속도 기반 외삽으로 bbox를 움직이지 않는다.

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

| `steerNorm` | **정규화된 조작 커맨드 [-1,1]** (`info.physics.steerAngle`) |

> ⚠️ **`steer`(도)와 `steerNorm` 은 서로 다른 물리량이다.** `steerNorm` 은 드라이버(사람/AI)의 **조작
> 커맨드**이고, `steer` 는 AC 조향필터를 거친 **결과 조향각**이다. 그래서 단일 상수(STEER_LOCK)로
> 환산되지 않는다: 소신호 이득은 178.6(≈180)이지만 `|steerNorm|>0.4` 부터 173 으로 포화하고, 커맨드가
> 결과보다 **앞선다**(AI 주행 시 6프레임=122ms, R²=0.994). **IL 라벨로는 `steerNorm` 을 그대로 쓴다**
> (정책이 내놓아야 할 값이므로). `steer`(도)는 상태 피처로 보존.
> 실제 조향각(도)이 필요하면 `steerNorm × STEER_LOCK`. **STEER_LOCK=180°** 는 `data.acd` 의
> `car.ini [CONTROLS] STEER_LOCK` 에서 확인됨(데이터 회귀의 소신호 이득 178.6 과 일치).

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
| `present` | 플러그인이 이 프레임에 이 차를 보냈는지. **항상 True** (가시성/리타이어 플래그가 아님) |
| `rel_x`, `rel_y`, `rel_z` | 내 차 기준 상대 위치 = (상대 world_position − ego world_position) |
| `distance` | 수평 유클리드 거리 `sqrt(rel_x² + rel_y²)` (m) |
| `opp_speedKMH` | 상대차 속력 (km/h) |
| `opp_vx`, `opp_vy`, `opp_vz` | 상대차 **월드 속도벡터** (m/s). 플러그인 `CS.Velocity` 실측 |
| `opp_yaw` | 상대차 **속도벡터의 방향** (rad). 정지 시 무의미 |
| `yaw_body` | 상대차 **차체 방향** (rad). 앞바퀴중점−뒷바퀴중점. 정지·스핀에도 유효 |
| `wheelbase`, `track_width` | 접지점 실측 휠베이스/앞윤거 (m). 서스펜션·조향으로 변하는 **동적** 값 |
| `tcp_ok`, `tcp_center_x/y`, `tcp_err` | TyreContactPoint 성공 여부와 검증용 값 |

- 폴더명 `opp_<id>` 의 `<id>` = 플러그인 `Opponent['id']` (그 run에 나온 값 그대로, 예: 1~5).
- **`present` 를 가시성이나 리타이어 판정에 쓰면 안 된다.** 플러그인은 트랙의 모든 차를 항상 보낸다.
  리타이어는 `processed/` 의 `retired`, 가시성은 `ac-export` 의 화각 판정이 담당한다.
- **속도는 시간차분으로 구하지 않는다.** 상대차 위치는 드물게 held 되어 차분이 0↔폭주로 튄다.
  플러그인 실측 `velocity` 를 쓴다(검증: `hypot(vx,vy)` vs `speedKMH/3.6` 최대 오차 0.064 m/s = 수직 성분).
- 상대차 상세 채널을 늘리려면 플러그인 `Opponent.update()` 에 추가 가능
  (단, **타차 조작값은 AC가 제공하지 않아 불가**).

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
  ② 도착 직후 20프레임 동안 그 자리에 고정(1m 이내, 속도<1km/h). **두 조건이 모두** 필요하다.
  CSP가 이미 회수한 차량을 세션 종료 때 다른 피트 슬롯으로 한 번 더 옮기는 경우가 있어,
  run 끝 좌표가 아니라 순간이동 직후의 연속 정지 구간을 판정한다.
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

</details>

---

## 프레임별 txt export

parquet 대신 **프레임 하나당 파일 하나**로 푼 포맷. `collect` 종료 시 자동 생성되고
`ac-export dataset/<run_id>` 로 수동 생성도 된다. 파일명은 **균일 격자 위의 시각(ms)** 이다.

```
dataset/<run_id>_<track>_<hz>hz/
    CAMERA/0.jpg  25.jpg  50.jpg  75.jpg ...   # 권장 40Hz → 25ms 간격
    EGO_INFO/0.txt  25.txt ...          # 자차 상태 (key: values)
    OBJECT_INFO/0.txt  25.txt ...       # 그 프레임에 '보이는' 상대차만 (헤더 + 차당 1행)
    export_info.json                    # 화각/차종/격자 Hz/프레임수 등 요약
```

- **파일명 ms = 균일 격자.** 실측 캡처는 미세하게 흔들리지만(19.8, 20.1ms...), 출력은 목표 Hz 의
  **정확한 격자**(0, step, 2·step...)에 리샘플한다. 각 격자점에는 시각이 **가장 가까운** 캡처
  프레임을 배치한다(리샘플 오차 ~2ms). `--grid-hz` 로 격자 Hz 를 바꿀 수 있다(기본 = 수집 목표 Hz).
- **격자 Hz 는 캡처가 꾸준히 낼 수 있는 값 이하로.** 격자가 캡처보다 빠르면 같은 프레임이 중복된다
  (`export_info.grid_duplicates` 로 보고, 경고 출력). 그때는 재수집 없이 `--grid-hz` 만 낮춰 재실행.
  캡처가 스톨한 구간은 그 격자 칸을 **비운다**(`grid_gaps`) — 없는 프레임을 만들어내지 않는다.
- 자차가 핏에 있는 프레임은 기본적으로 **제외**된다(`--all-frames` 로 포함 가능).
- 이미지는 하드링크로 이어붙이므로 같은 볼륨이면 **추가 용량 0**.

### 가시성 판정 — "그 프레임에 보이는 차"

AC 는 "이 차가 화면에 보이는가"를 알려주지 않는다. 플러그인은 트랙의 **모든 차를 항상** 보내며,
`present` 는 전 run 에서 100% True 라 **리타이어 플래그도 가시성 플래그도 아니다.**
그래서 기하로 계산한다. 상대차가 아래 세 조건을 모두 만족하면 그 프레임에 기록한다:

1. `opp_forward > 0` (자차 앞쪽)
2. `|atan2(opp_lateral, opp_forward)| <= 수평FOV/2`
3. `retired == False` (핏으로 회수되지 않음)

수평 FOV는 `--onboard-camera`가 가리키는 차종 `cameras.ini`의 수직 FOV와 실제 캡처 종횡비에서
`hfov = 2·atan(tan(vfov/2)·(W/H))`로 구한다. 현재 CAMERA_0(FOV 60°), 1920×945 캡처에서는
**hfov 91.49°, f≈935 px**다. 새 run의 정밀 3D bbox 검증에서는 이 고정 extrinsic을 쓰지 않고
프레임별 CSP actual pose/FOV를 사용한다.

- **차폐(occlusion)는 처리하지 않는다.** 앞차 뒤에 완전히 가려진 차도 '보임'으로 들어간다.
- 카메라가 차 원점에 있다고 근사한다(실제로는 약간의 오프셋. 먼 거리에선 무시 가능).
- **크래시로 트랙에 서 있는 차는 제외되지 않는다** — 카메라에 실제로 보이는 장애물이기 때문.
  핏으로 순간이동(`retired`)한 뒤에야 빠진다.

### 좌표계와 단위 (요약)

- **길이 = 미터(m), 시간 = 초(s), 각속도 = 도/초(deg/s), 각도 = 쿼터니언(무차원).**
- 파일명 `<ms>` = **밀리초**, 수집 시작이 0.
- `position` 의 좌표계는 **UTM 이 아니다.** Assetto Corsa 의 **트랙별 로컬 월드 좌표**(원점은 트랙마다
  다르고 임의). 단위만 미터일 뿐 측지 기준·투영대가 없어 위경도로 변환할 수 없다.
  대신 **한 트랙 안에서는 완전히 고정**이라 차량 간 거리·방향·궤적 계산에 그대로 쓴다.
  (`opp_forward` 같은 자차 기준 상대좌표와 구분해 "월드 좌표"라고 부른다.)
- 수평면 = (x, y), **z = 높이**. EGO_INFO 와 OBJECT_INFO 의 `position` 은 **같은 좌표계**다.

### `EGO_INFO/<ms>.txt` (8줄)

| 줄 | 성분 | 단위 | 좌표계 / 부호 | 원천 |
|---|---|---|---|---|
| `position` | x y z | **m** | 트랙 로컬 월드, z=높이 | `world_position_x/y/z` |
| `orientation` | qx qy qz qw | **무차원** | 단위 쿼터니언, ZYX(yaw→pitch→roll) | `yaw/pitch/roll` 변환 |
| `velocity` | 전진 좌우 상하 | **m/s** | **차체 기준**(body) | `local_velocity_x/y/z` |
| `angular_velocity` | roll pitch yaw rate | **deg/s** | 차체 기준, z=yaw rate | `angular_velocity_z/x/y` |
| `acceleration` | 종 횡 수직 | **m/s²** | 차체 기준, **수직은 항상 0** | `accelY·g`, `accelX·g` |
| `throttle` | — | **무차원 0~1** | 1 = 완전 개방 | `accStatus` |
| `brake` | — | **무차원 0~1** | 1 = 완전 제동 | `brakeStatus` |
| `steer_angle` | — | **무차원 −1~1** | + 좌 / − 우 | `steerNorm` |

> ⚠️ 함정 셋. ① `angular_velocity` 는 rad/s 가 아니라 **deg/s**(원본 telemetry 는 rad/s, 여기서 변환).
> ② `acceleration` 의 **수직 성분은 항상 0** — AC 가 주지 않아 자리만 채운 것이므로 실제 값으로 믿으면 안 된다.
> ③ `steer_angle` 은 **각도가 아니다.** 정규화된 조작 커맨드다. 실제 조향각(도)이 필요하면
> `steer_angle × STEER_LOCK`(F2004/SF15-T 는 180). STEER_LOCK 은 `export_info.json` 에 있다.

> ⚠️ **원본 필드의 축 순서가 제각각이라 여기서 정리한다.** `local_velocity` 는 플러그인이 재배열했지만
> **`angular_velocity` 는 재배열하지 않은 원본 축**이다(`angular_velocity_y` 가 yaw rate, 실측 상관 0.948).
> 그리고 **`accelX` 는 사실 횡가속도, `accelY` 가 종가속도**다(이름이 뒤바뀜; 종가속도가 `d(speed)/dt` 와
> 상관 0.895, 비율 9.51≈9.81 로 G 단위임도 확인). 필드명을 그대로 믿으면 조용히 틀린다.

### `OBJECT_INFO/<ms>.txt` (헤더 + 보이는 차마다 1행, CSV)

```
position.x, position.y, position.z, orientation.x, orientation.y, orientation.z, orientation.w,
dimension.length, dimension.width, dimension.height, velocity.x, velocity.y, velocity.z, track_id
```

| 필드 | 단위 | 의미 / 좌표계 | 원천 |
|---|---|---|---|
| `position.x/y/z` | **m** | 트랙 로컬 월드 (EGO_INFO 와 **같은 좌표계**) | `ego world_position + rel_x/y/z` |
| `orientation.x/y/z/w` | **무차원** | 차체 방향 쿼터니언. **yaw 만**(pitch=roll=0) | `yaw_body` (바퀴 접지점 기반) |
| `dimension.length/width/height` | **m** | 차체 박스 | `data.acd` 의 `colliders.ini` 박스 합집합 |
| `velocity.x/y/z` | **m/s** | **그 상대차의 차체 기준** 전진/좌우/상하 | 플러그인 `velocity`(월드)를 `yaw_body` 로 회전 |
| `track_id` | **정수** | 상대차 ID. 세션 내내 고정. **0 은 안 나옴**(자차) | `opp_id` |

> ⚠️ 함정 둘. ① **`velocity` 는 상대속도가 아니다.** 자차 대비가 아니라 그 차 자신의 절대 속도이며,
> 그것도 그 차의 차체 기준이다(예: `54.7` = 그 차가 자기 앞방향으로 초속 54.7m). 자차 대비가 필요하면
> `processed/opponent/` 의 `rel_vx/rel_vy/closing_speed` 를 쓴다.
> ② **`orientation` 은 yaw 만 있다**(qx=qy=0). AC 가 상대차의 pitch·roll 을 주지 않는다.
>
> 자차의 `dimension`/차종/`STEER_LOCK` 은 프레임마다 변하지 않으므로 EGO_INFO 에 넣지 않고
> **`export_info.json` 에 run 당 한 번** 기록한다(`car_0` = 자차, `car_1..N` = 상대차).

> **`orientation` 은 `opp_yaw` 가 아니라 `yaw_body` 를 쓴다.** `opp_yaw` 는 플러그인이 **속도벡터**로
> 계산하는 값이라(`atan2(f, l)`) 차가 멈추면 무의미해진다 — 실측: 정지 프레임의 96% 가 정확히 `+45.0°`.
> `yaw_body` 는 앞바퀴 중점 − 뒷바퀴 중점 벡터라 **정지·스핀 중에도 유효**하다(크래시로 20초간 멈춘
> 차에서 표준편차 0.15°로 고정, 같은 구간 `opp_yaw` 는 ±90° 로 요동).
> `dimension` 은 **`data.acd` 를 직접 복호화**해서 `colliders.ini` 의 충돌박스 합집합 bounding box 로
> 얻는다(`acd.py`). AC 파이썬 API 로는 못 얻지만(ui_car.json 에도 없음) 게임 파일 안에는 있다.
> Content Manager 없이 폴더명 기반 키로 푼다(F2004: 길이 4.80 / 폭 1.65 / 높이 0.985m, STEER_LOCK 180°).
> 상대차 차종은 `cfg/race.ini` 의 `[CAR_n] MODEL` 로 안다. `data.acd` 읽기 실패 시에만 기본값 사용.
> 실측 휠베이스/윤거(`processed/opponent/` 의 `wheelbase`, `track_width`)는 별개로, 접지점 기반 **동적** 값이다.

---

## 패키지 구조 / 파일

```
Summer_USRG/
    pyproject.toml                # 패키지 정의 + 의존성 + CLI 진입점
    README.md
    src/ac_dataset/
        __init__.py               # 공개 API: collect, build_dataset, extract_track_for_run, build_processed, export_run, validate_run
        collect.py                # 수집 메인 → 종료 시 정렬/track/후처리/export까지 자동
        schema.py                 # ego/action/opponent 필드 정의 + 추출 헬퍼 (필드명 single source)
        track.py                  # 트랙 자동식별 + fast_lane.ai 파싱 → track/ (맵+레이싱라인)
        processed_builder.py      # raw → processed/ (E2E/IL용 선별·가공 피처 + 유효 마스크)
        export_frames.py          # processed → CAMERA/EGO_INFO/OBJECT_INFO (균일격자 프레임별 txt)
        acd.py                    # data.acd 복호화 → 차량 제원(STEER_LOCK, 충돌박스 치수)
        ac_shmem.py               # AC Graphics SHM + CSP v2 렌더 ring 안전 읽기
        validate.py               # 수집 run 자동 검증(ac-validate) → validation.json
        view.py                   # parquet 미리보기/CSV 뷰어
        recorder/
            telemetry_logger.py   # ego+action+opponents raw 기록 (AC 플러그인 state)
            ac_telemetry.py       # ego_server(UDP 2345)에서 자차 state 수신
            ac_opponents.py       # opponents_server(TCP 2346)에서 상대차 수신
            screen_capture.py     # 화면 캡처 (dxcam 우선 → mss 폴백, region 지정 가능)
            image_logger.py       # 캡처 프레임을 camera/ 에 JPG + index로 저장
            threaded_image.py     # 이미지 캡처/인코딩 스레드 분리(telemetry 부하 격리)
            sync.py               # 고정 주기 실행·공통 monotonic clock 유틸리티
            dataset_builder.py    # raw → camera/ ego_state/ opponent/ 정렬·분리(미매칭 프레임 드롭)
    csp_apps/ac_camera_pose_bridge/ # 실제 렌더 카메라·차량 상태를 공유메모리에 쓰는 Lua 앱
    tests/test_offline.py         # 게임 없이 서로 다른 주기 raw 생성 → 정렬 검증
    tests/test_camera_pose.py     # 좌표축·렌더 packet 관계·카메라 pose 단위 테스트
    dataset/                      # 수집 결과 (git 미추적)
```

## 설치

```bash
git clone https://github.com/songdavid0206/Summer_USRG.git
cd Summer_USRG
pip install -e .            # 의존성 자동 설치 + collect / ac-build / ac-view 명령 생성
pip install -e ".[capture]" # (선택) Windows에서 더 빠른 dxcam 캡처 백엔드까지
```
설치하면 아래 명령들을 **어느 폴더에서든** 쓸 수 있다:
`collect`, `ac-build`, `ac-view`, `ac-track`, `ac-process`, `ac-export`.

3D bbox 검증용 실제 렌더 카메라 pose를 기록하려면 CSP Lua 브리지도 설치한다(폴더째 복사):

```text
csp_apps/ac_camera_pose_bridge
  → <Assetto Corsa>/apps/lua/ac_camera_pose_bridge
```

게임 내 CSP 앱 목록에서 **AC Dataset Camera Pose**를 활성화한다. `collect` 시작 로그에
`CSP 실제 카메라 pose = 연결`이 떠야 하며, 이후 `camera_index.parquet`에는 각 이미지의
`cam_pos_*`, `cam_forward_*`, `cam_up_*`, `cam_fov_deg`가 저장된다. 이 값은 CSP 렌더 상태에서
직접 오며 이미지 픽셀을 읽거나 bbox를 이미지에 맞춰 보정하지 않는다.
F1/F6/F5/트랙 카메라 전환은 프레임별 `cam_mode`, `cam_drivable_mode`, `cam_car_index`,
`cam_focused_car`에도 함께 기록된다.

브리지 v2는 같은 렌더 callback에서 모든 차량의 `position/look/up`, visual AABB와 상태 timestamp도
기록한다. 128-slot ring buffer(슬롯당 최대 32대)와 odd/even sequence를 사용해 Lua가 쓰는 도중 Python이
반쪽짜리 프레임을 읽지 않게 한다. 정렬 후 `camera/render_car_state.parquet`에 저장되며, 카메라와
상대차가 동일 렌더 tick을 사용하므로 상대차 서버 지연이나 yaw-only 근사를 3D bbox에 사용하지 않는다.

### ⚠️ 게임 플러그인도 수정해야 한다 (레포 밖)

이 파이프라인은 `sensors_par` 플러그인의 **수정된 버전**을 전제로 한다. 수정 파일은 게임 폴더에 있어
git 으로 추적되지 않으므로, 게임을 재설치하거나 플러그인을 덮어쓰면 **다시 적용해야 한다.**
대상: `<AC>/apps/python/sensors_par/` (원본은 `.bak` 으로 백업해 둘 것)

| 파일 | 수정 | 이유 |
|---|---|---|
| `sensors_par.py` | 상단에 `import encodings.idna` 사전로드, opponents_server 에 try/except + SO_REUSEADDR, 송신 루프 `time.sleep(0.008)` | 없으면 2346 포트가 조용히 안 열림 / 상대차 ~50Hz |
| `structures.py` | `Car.update()` 에 `steerNorm = info.physics.steerAngle` | steer 를 [-1,1] 로 쓰기 위해 |
| `structures.py` | `Opponent.update()` 에 `velocity`, 그리고 `update_body_pose()` (→ `yaw_body`, `wheelbase`, `track_width`, `tcp_center_x/y`, `tcp_ok`) | 상대차 실측 속도벡터 + **차체 방향**(AC 가 안 주는 값) |

- 플러그인 수정은 **세션 재로드**(메뉴 나갔다 트랙 재진입) 후에 적용된다.
- `Opponent.update_body_pose()` 는 `acsys.CS.TyreContactPoint` 가 **상대차에도 먹힌다**는 사실을 이용한다.
  검증: 접지점 중심과 `world_position` 의 거리(`tcp_err`)가 5대 모두 0.16m 로 동일(ego 것만 반환했다면
  차마다 수백 m 로 제각각). 실측 휠베이스 3.05m / 앞윤거 1.47m 는 F2004 실제 제원(3.050 / 1.470)과 일치.
- 실패해도 수집은 계속된다(`tcp_ok=0`). 구버전 플러그인이면 `processed_builder` 가 옛 방식으로 폴백한다.

## 사용법

**실제 수집 (AC + 플러그인 + CSP 실행 후, 트랙에 진입한 상태)**
```bash
collect --capture-region 0,80,1920,945 --telemetry-hz 100 --image-hz 40 --onboard-camera 0
# 주행(사람 또는 AI) → Alt+Tab 으로 터미널 이동 → Ctrl+C 로 종료 (게임이 키보드를 잡으므로 Alt+Tab 필수)
# 종료 시 정렬 → track → processed → 40Hz export까지 자동 실행
```
- **HUD는 게임에서 꺼두되, `--capture-region`으로 남는 UI 요소(상단 배너/하단 스트립 등)까지 잘라낸다.**
  값 `L,T,W,H`는 게임 해상도/화면에 맞게 조정(예: 1920×1080에서 위 80px·아래 55px 잘라내면 `0,80,1920,945`).
- `--telemetry-hz`, `--image-hz` : 목표 주파수(기본 둘 다 50). `--image-format jpg|png`.
- `--onboard-camera 0`은 차종·FOV metadata와 구 run fallback용 CAMERA_0 지정이다. **새 run의 실제
  extrinsic은 CSP가 매 프레임 직접 기록**하므로 수집 중 F1/F5/F6 등으로 시점을 바꾸어도 새 pose가
  저장된다. 단 학습 데이터의 시점 일관성이 필요하면 한 카메라 모드를 유지해야 한다.
- **AI 자동주행 수집**: 트랙 주행 중 게임 창에서 `Ctrl+C` → AC 내장 AI가 대신 운전. 그 상태로 `collect` 하면
  AI 조작값이 라벨로 저장돼 IL 전문가 데이터가 된다. (게임 Ctrl+C = AI토글, 터미널 Ctrl+C = 수집종료)

> 코드로 부르려면: `from ac_dataset import collect; collect(out_dir="./dataset")`

`collect` 는 종료 시 **정렬 → track → processed → export** 까지 자동으로 이어진다.
검증(`ac-validate`)은 여기 포함되지 않는다 — 데이터가 저장된 뒤 필요할 때 따로 돌린다(아래 참조).
수집 중에는 **무거운 작업(대량 파일 I/O 등)을 같이 돌리지 말 것** — CPU/디스크를 뺏기면 캡처 스톨이
늘어 이미지 Hz 가 떨어진다(telemetry 는 스레드 분리로 대체로 유지).

**트랙/후처리/내보내기 (재)생성** (`collect` 시 자동 생성되지만 수동으로도)
```bash
ac-track   dataset/<run_id>          # 주행 좌표로 트랙 자동 식별 → track/ 생성
ac-process dataset/<run_id>          # raw → processed/ (E2E/IL용 선별·가공 피처 + 유효 마스크)
ac-export  dataset/<run_id>          # processed → CAMERA/EGO_INFO/OBJECT_INFO (프레임별 txt)
#   --grid-hz 40      파일명 균일 격자 Hz (기본: 수집 목표 Hz). 중복 뜨면 낮춰 재실행
#   --hfov 91.49      화각 수동 override (기본: 차종 CAMERA_N 설정에서 읽음)
#   --max-range 150   이 거리보다 먼 상대차 제외
#   --all-frames      자차가 핏에 있는 프레임도 포함
```

**기존 raw run을 새 폴더 구조로 (재)정렬**
```bash
ac-build dataset/<run_id> --tolerance 0.05
```

**데이터 확인**
- 엑셀: 각 폴더의 `.csv` 를 바로 열기 (`ego_state/ego_state.csv` 등).
- 터미널: `ac-view dataset/<run_id>/ego_state/ego_state.parquet --csv`

**자동 검증 (`ac-validate`)** — 수집과 **분리**돼 있어, 저장된 뒤 원할 때 직접 돌린다.
```bash
ac-validate dataset/<run_id>          # 원본 run 폴더를 넣는다 (export 폴더 아님)
```
run 의 구조 일관성 10개 항목을 PASS/FAIL 로 검사하고 `<run>/validation.json` 에 저장한다:
행 개수 일치 / jpg 수 / 정렬 dt(≤50ms) / NaN 없음 / 시간 단조성 / export 3폴더 파일명 일치 /
ms 중복 없음 / **리타이어 상대차가 OBJECT_INFO 에서 빠졌는지**. 하나라도 FAIL 이면 exit code 1.
(행삭제·NaN·dt오염·시간역행 음성테스트로 FAIL 감지를 확인했다.)

**오프라인 테스트 (게임 불필요)**
```bash
python tests/test_offline.py
```

## 성능과 수집 주파수

목표 Hz는 `--telemetry-hz`, `--image-hz`로 준다(기본 둘 다 50, 서로 달라도 됨).
**수집 종료 시 `[rate]`로 실측 Hz가 출력**되고 `meta.json` 의 `telemetry_hz_actual` /
`image_hz_actual` 에 저장된다.

```bash
collect --capture-region 0,80,1920,945 --telemetry-hz 100 --image-hz 40 --onboard-camera 0
```

> **telemetry는 이미지에서 분리돼 있다.** 이미지 캡처/인코딩은 별도 스레드가 처리하므로
> 현재 수정된 플러그인에서 ego 100 Hz와 image 40 Hz를 동시에 유지한다. 두 rate가 달라도
> AC 게임시간 보간으로 이미지 시각의 ego state를 만든다.
>
> **최종 출력 파일명은 균일 격자다.** 실측 캡처는 미세하게 흔들리지만(std ~2ms), `ac-export` 가
> 목표 Hz 의 정확한 격자로 리샘플한다(위 "프레임별 txt" 참조). 그래서 지정한 Hz 가 파일명에
> 그대로 나온다. 격자 Hz 를 캡처 지속 속도 이하로 두면(예: 50 캡처 → 40 격자) 중복·빈칸이 없다.

### 이미지 캡처 스레드 분리

`collect` 는 이미지 경로(grab ~6ms + JPG 인코딩·저장 ~17ms = ~23ms)를 **두 스레드로 분리**한다
(`recorder/threaded_image.py`):
- **캡처 스레드**: `image_hz` 주기로 `screen.grab()` 만 하고 (t, frame) 을 큐에 넣는다.
- **인코더 워커**: 큐에서 꺼내 JPG 인코딩+저장(순서 유지, 프레임 유실 없음).

dxcam grab과 PIL 인코딩은 네이티브 호출 중 GIL을 놓아 실제로 병렬 실행된다. 각 스트림은 자신의
wall-clock과 AC 게임시간을 모두 남긴다. tolerance(기본 50 ms) 안에 게임시간으로 대응할 telemetry가
없는 프레임(주로 종료 직후 큐 잔여)은 `build_dataset`이 버린다.

각 스트림의 실제 한계:

| 스트림 | 실측 한계 | 무엇이 막나 |
|---|---|---|
| **telemetry(ego)** | 100.0Hz | 수정된 sensors_par UDP 송신과 이미지 스레드 분리 |
| **image** | ~48-50Hz | 캡처(grab ~6ms) + JPG 인코딩·저장(~17ms) = ~23ms. 파이프라인으로 겹쳐 ~50Hz 상한. 순간 스톨이 평균을 조금 깎음 |
| **opponents** | 약 50Hz | 플러그인 `sensors_par.py`의 opponents 송신 주기. bbox는 이 스트림 대신 CSP same-render 상태 사용 |

- **이 PC 의 스윗스팟은 40Hz.** 이미지 캡처가 한 장 ~23ms 라 50Hz(20ms 주기)는 여유가 거의 없어
  순간 스톨에서 48-50Hz 로 흔들린다. **40Hz(25ms 주기)면 여유가 생겨 캡처가 목표를 정확히 낸다** →
  균일 격자 export에서 중복 0. 최신 실측은 image 40.44Hz, telemetry 100.0Hz, 3,197개 export에서
  중복 0/캡처 스톨 빈칸 3이다.
- **원격 데스크톱(RDP)에서는 더 떨어진다.** dxcam이 실제 GPU 출력 대신 RDP 화면을 잡기 때문.
  **로컬(그 PC 앞)에서 수집할 것.**
- **용량**: image 45Hz JPG(q95, 1920×945) ≈ 초당 32MB (686KB/장). 대량 수집 시 품질(q85 → 8.4ms,
  파일 ~30%↓) 조절.

## 참고 / 주의

- **HUD**: 게임에서 HUD를 꺼도 일부 UI(상단 배너·하단 스트립 등)가 남으므로, `--capture-region`으로 그 영역까지 잘라낸다. (속도계/맵이 이미지에 찍히면 학습이 편법으로 흐름)
- **캡처 세팅**: AC를 테두리 없는 창(borderless windowed)으로 실행 권장(전체화면 독점은 캡처가 검게 나올 수 있음).
- **raw 카메라**: AC는 렌더 이미지를 API로 주지 않음 → **화면 캡처가 유일**. 대신 숫자로 **BEV(탑다운) 이미지**는 생성 가능.
- **좌표축**: ego world_position 수평면은 (x, y), z=높이. AI 파일(fast_lane.ai)은 축 순서가 달라 `ego(x,y,z)=AI(z,x,y)`로
  변환해 저장(track.py). 등수 등 진행방향 판정 시엔 ego 위치 변화로 heading을 구해 투영하는 방식이 안전.
