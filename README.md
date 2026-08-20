# Summer_USRG

Assetto Corsa에서 주행 화면, 자차 동역학, 운전 조작값, 상대차 상태와 트랙 정보를 수집하고, 이를 이용해 고속 자율주행 모델을 학습·평가하는 Windows 기반 연구 프로젝트다.

현재 기준 환경은 **Laguna Seca, Ferrari SF15-T, 40 Hz**다. 데이터 수집부터 시간 동기화, 3D ground truth 검증, TCP 계열 imitation learning, read-only shadow inference, vJoy guarded closed loop까지 한 저장소에서 다룬다.

최종 폐루프 구조는 모델이 미래 waypoint를 예측하고, pure pursuit가 조향을 계산하며, 전문가 속도 프로파일 기반 PI/PID가 throttle과 brake를 제어하는 하이브리드 방식이다.

```text
RGB + ego state + racing line + target speed
                    │
                    ▼
             TCP-AC V4 planner
                    │
           future waypoints 6개
                    │
                    ▼
 Pure Pursuit + cross-track correction ──> steering

 normal-train expert speed profile
                    │
          braking-distance preview
                    │
                    ▼
              PI/PID + guard ────────────> throttle / brake
```

## 현재 상태

| 항목 | 상태 |
|---|---|
| RGB 수집 | DXCAM 기준 40 Hz |
| Ego telemetry | 약 100 Hz |
| 시간 정합 | AC game time 기반 보간 |
| 렌더 상태 | CSP actual camera pose와 same-render vehicle 6DoF/AABB |
| 데이터 검증 | 구조·시각·물리 일관성 검사 |
| 학습 데이터 | Normal, synthetic recovery, DAgger 통합 |
| 모델 | TCP-AC V4~V8 실험 구현 |
| 실시간 추론 | CUDA 40 Hz shadow/closed loop |
| 최종 제어 | V4 waypoint + classical lateral + profile PI/PID |
| 테스트 | 99 unit tests 통과 |

대표 hybrid run은 120초 동안 4,800 control frame을 처리해 `39.9996 Hz`를 기록했다. 학습 데이터와 checkpoint는 용량 문제로 Git에 포함하지 않는다.

## 주요 기능

- Windows 화면을 DXCAM 또는 MSS로 캡처
- `sensors_par`에서 ego/opponent telemetry 수신
- AC graphics shared memory의 game time과 lap 정보 기록
- CSP Lua bridge로 실제 렌더 카메라 pose와 차량 상태 기록
- 카메라 frame 기준 telemetry 보간 정합
- `fast_lane.ai` parsing과 트랙 자동 식별
- 정적 AI racing line과 track edge 추출
- 차량 `data.acd`에서 steering lock과 collider 정보 추출
- 동일 렌더 시점의 차량 3D bounding box 생성
- Pit, retire, teleport와 invalid frame 자동 마스킹
- 25 ms 균일 grid의 structured dataset export
- Normal/recovery/DAgger trajectory-safe 학습 manifest 생성
- TCP-style offline 평가와 live shadow inference
- Watchdog가 포함된 vJoy closed-loop 제어
- Pure pursuit, cross-track feedback, preview PI/PID와 slip guard

## 설치 전에 필요한 것

이 저장소만 설치해서 Assetto Corsa가 자동으로 구성되지는 않는다. 다음 항목이 먼저 필요하다.

| 구성요소 | 용도 | 필수 여부 |
|---|---|---|
| Windows 10/11 | 화면 capture와 vJoy | 필수 |
| Assetto Corsa | Simulator | 필수 |
| Content Manager | 세션과 control 설정 | 권장 |
| Custom Shaders Patch | Lua render bridge | 필수 |
| `sensors_par` | Ego/opponent telemetry | 필수 |
| Python 3.9 이상 | 수집·학습·평가 | 필수 |
| NVIDIA GPU | 실시간 학습/추론 | 권장 |
| vJoy 2.1.9.1 | Closed-loop actuation | 제어 시 필수 |

게임, CSP, telemetry app과 camera bridge 설치는 [SETUP.md](SETUP.md)에 정리돼 있다. 수집만 할 때는 vJoy가 필요 없다.

## Python 설치

PowerShell에서 저장소를 clone하고 editable package로 설치한다.

```powershell
git clone https://github.com/songdavid0206/Summer_USRG.git
cd Summer_USRG
python -m pip install --upgrade pip
python -m pip install -e ".[capture,train]"
```

학습 기능이 필요 없으면 다음처럼 설치할 수 있다.

```powershell
python -m pip install -e ".[capture]"
```

설치 확인:

```powershell
python -c "import ac_dataset; print('ac_dataset import OK')"
python -m unittest discover -s tests -p "test_*.py"
```

## Assetto Corsa 연결 확인

트랙 주행 화면에 진입한 뒤 collector를 실행하면 다음 연결이 준비돼 있어야 한다.

```text
ego telemetry UDP 2345       연결
opponents TCP 2346           multi session에서 연결
AC Graphics shared memory    연결
CSP render pose bridge       연결
```

`sensors_par`와 CSP app을 설치하거나 패치하는 과정은 [SETUP.md](SETUP.md)를 따른다. Game window 위에 다른 창이 겹치면 그 화면이 그대로 학습 데이터에 들어가므로 수집 중에는 창을 가리지 않는다.

## 데이터 수집

### Solo 7랩

```powershell
python -m ac_dataset.collect `
  --session-mode solo `
  --laps 7 `
  --capture-backend dxcam `
  --capture-region 0,80,1920,945 `
  --telemetry-hz 100 `
  --image-hz 40 `
  --jpg-quality 80
```

### Multi 7랩

```powershell
python -m ac_dataset.collect `
  --session-mode multi `
  --laps 7 `
  --capture-backend dxcam `
  --capture-region 0,80,1920,945 `
  --telemetry-hz 100 `
  --image-hz 40 `
  --jpg-quality 80
```

정상 종료하면 raw build, quality 검사와 structured export가 이어서 실행된다. 수집만 하고 자동 export를 생략하려면 `--no-structured-export`를 추가한다.

주의할 점:

- `--laps`는 Content Manager의 race length를 변경하지 않는다.
- 목표 랩을 완전히 저장할 수 있도록 게임의 lap 수를 충분히 설정한다.
- 수집 도중 설정 창, telemetry overlay 또는 desktop이 보이면 해당 구간을 학습에서 제외한다.
- Ego가 retire하거나 세션이 비정상 종료되면 해당 run의 품질을 먼저 확인한다.

## 데이터 구조

### Raw run

```text
dataset/<run_id>/
├─ camera/
├─ ego_state/
├─ opponent/
├─ processed/
├─ track/
├─ image_index.parquet
├─ telemetry_raw.parquet
├─ opponents_raw.parquet
└─ meta.json
```

Raw stream은 재가공의 기준 자료다. 동기화 방식이나 exporter를 수정해도 원본을 다시 수집하지 않도록 보존한다.

### Structured dataset

```text
dataset_v2/
├─ solo/<session>/
│  ├─ SESSION_INFO.txt
│  ├─ SCHEMA.txt
│  ├─ FRAME_INDEX.csv
│  ├─ TRACK_INFO/
│  └─ lap_XX/{CAMERA,EGO_INFO}
│
└─ multi/<session>/
   ├─ SESSION_INFO.txt
   ├─ FRAME_INDEX.csv
   ├─ EXCLUDED_FRAMES.csv
   ├─ TRACK_INFO/
   ├─ INTERACTIONS/
   └─ NON_INTERACTION/
```

이미지는 가능한 경우 hard link로 연결한다. `dataset/`, `dataset_v2/`, `training/`은 Git에서 제외된다.

## 시간 동기화

화면 capture, UDP telemetry, CSP render callback과 GPU present는 서로 다른 시점에 실행된다. 수집 중에 이들을 한 행으로 강제로 묶지 않고 독립 raw stream으로 저장한 뒤 camera frame을 기준으로 정합한다.

```text
telemetry timestamp_ac/currentTime
                 +
image graphics iCurrentTime/lap count
                 ↓
         monotonic game time
                 ↓
       image 시각에 상태 보간
```

연속값은 선형 보간하고 yaw/pitch/roll은 sin/cos 공간에서 보간한다. 동일 game time의 인접 CSP render sample은 graphics packet과 CSP render frame 관계를 이용해 구분한다.

## 3D Bounding Box와 물리 검증

Bounding box는 이미지에 맞춰 수동 fitting한 값이 아니다. CSP가 같은 render callback에서 제공한 실제 camera pose, vehicle position/look/up과 visual AABB를 사용해 image plane에 투영한다.

![CSP same-render 3D bounding box, BEV and physics validation](docs/images/validation_bbox_and_physics.png)

위 화면은 다음 검증을 한 번에 보여준다.

- 실제 Assetto Corsa RGB와 상대차 3D bounding box
- 전체 트랙과 현재 위치
- Ego-centric BEV와 camera FOV
- 속도, control, yaw rate와 acceleration
- 위치미분 속도와 telemetry speed 비교
- Yaw와 진행방향 비교
- 종·횡가속도 물리 일관성
- 상대차 위치미분 속도 비교

독립 검증 도구는 별도 저장소에 있다.

- [Summer_USRG_Validation](https://github.com/songdavid0206/Summer_USRG_Validation)

```powershell
python -m pip install -e ..\Summer_USRG_Validation
av-check dataset/<run_id>
av-view dataset/<run_id> --frame 300
```

## 데이터 검증과 재가공

Raw run 구조 검사:

```powershell
python -m ac_dataset.validate dataset/<run_id>
```

기존 raw run을 최신 structured schema로 재export:

```powershell
python -m ac_dataset.structured_export dataset/<run_id>
```

Laguna Seca 분포 분석:

```powershell
python tools\analyze_lagunaseca_distribution.py
python tools\build_lagunaseca_viewer.py
start analysis\lagunaseca_distribution\run_viewer.html
```

분포 분석 결과는 `analysis/lagunaseca_distribution/`에 생성된다.

## TCP-AC 학습 데이터

현재 V4 cohort는 다음 데이터로 구성됐다.

| 구분 | Frame | Trajectory |
|---|---:|---:|
| 전체 | 209,098 | 440 |
| Train | 157,636 | 338 |
| Validation | 51,462 | 102 |

원본 session과 trajectory 단위로 split하며 train/validation overlap은 0이다. 미래 label은 같은 lap 또는 recovery episode 안에서만 조회한다.

V4 입력:

- RGB `3×256×512`
- Ego dynamics 6개
- 0~150 m의 ego-local racing-line point 11개
- Standing/flying phase를 포함한 command 6개
- 11개 target speed와 현재 speed error

V4 출력:

- Current steer/throttle/brake
- `0.1/0.25/0.5/1.0/1.5/2.0 s` future waypoint 6개
- 같은 horizon의 future control

Racing line은 track에 저장된 정적 route prior다. 미래 실제 ego waypoint나 future action을 inference input으로 사용하지 않는다.

## V4 준비와 학습

대용량 dataset과 prepared manifest는 저장소에 포함되지 않는다. 먼저 로컬 `dataset_v2`와 기존 normal prepared data가 필요하다.

```powershell
python scripts/experiments/prepare_v4.py `
  --dataset-v2 dataset_v2 `
  --normal-prepared training/tcp_lagunaseca_v3_dense `
  --output training/tcp_lagunaseca_v4

python scripts/experiments/audit_v4.py --prepared training/tcp_lagunaseca_v4
```

학습:

```powershell
python train.py `
  --prepared training/tcp_lagunaseca_v4 `
  --output training/tcp_lagunaseca_v4/checkpoints `
  --epochs 50 `
  --batch-size 32 `
  --workers 16 `
  --lr 1e-4 `
  --weight-decay 1e-5 `
  --imagenet-pretrained `
  --balanced-sampler `
  --amp `
  --early-stopping-patience 8
```

마지막 epoch의 `last.pt`가 아니라 validation loss가 가장 낮은 `best.pt`를 사용한다.

## Offline 평가

```powershell
python evaluate.py `
  --prepared training/tcp_lagunaseca_v4 `
  --checkpoint training/tcp_lagunaseca_v4/checkpoints/best.pt `
  --output training/tcp_lagunaseca_v4/evaluation
```

평가는 control MAE, horizon별 waypoint error, source/phase별 error와 track-position error를 저장한다. Offline 정확도는 closed-loop 안정성을 보장하지 않으므로 바로 차량을 제어하지 않고 shadow를 먼저 실행한다.

## Read-only shadow

Shadow runner는 실제 RGB와 telemetry로 inference하지만 차량에 command를 보내지 않는다.

```powershell
python shadow.py `
  --prepared training/tcp_lagunaseca_v4 `
  --checkpoint training/tcp_lagunaseca_v4/checkpoints/best.pt `
  --output training/tcp_lagunaseca_v4/shadow `
  --duration 60 `
  --hz 40
```

결과:

```text
training/tcp_lagunaseca_v4/shadow/<timestamp>/
├─ config.json
├─ first_frame.jpg
├─ shadow.jsonl
└─ summary.json
```

## vJoy 설정

Closed loop에서는 vJoy device 1의 X/Y/Z를 steer/throttle/brake로 사용한다. 저장소에 vJoy 설치 파일은 포함하지 않는다. vJoy를 별도로 설치한 뒤 Content Manager의 `Settings > Assetto Corsa > Controls`에서 input method를 wheel로 바꾸고 축을 매핑한다.

Bridge 실행:

```powershell
python vjoy_bridge.py
```

축 확인:

```powershell
python vjoy_axis_test.py --steer 0.10 --duration 2 --verify-telemetry --confirm-axis-test
python vjoy_axis_test.py --throttle 0.20 --duration 2 --verify-telemetry --confirm-axis-test
python vjoy_axis_test.py --brake 0.20 --duration 2 --verify-telemetry --confirm-axis-test
```

Bridge는 command가 0.15초 이상 끊기거나 F8을 누르면 `steer=0`, `throttle=0`, `brake=1`을 latch한다. 실제 주행 전에 F8을 반드시 시험한다.

## 최종 Hybrid closed loop

현재 안정 기준 명령은 다음과 같다.

```powershell
python closed_loop.py `
  --prepared training/tcp_lagunaseca_v4 `
  --checkpoint training/tcp_lagunaseca_v4/checkpoints/best.pt `
  --output training/tcp_lagunaseca_v4/closed_loop `
  --arm-delay 8 `
  --duration 60 `
  --hz 40 `
  --realtime-lite `
  --lateral-controller waypoint-pure-pursuit `
  --lateral-wheelbase-m 3.5 `
  --lateral-lookahead-base-m 8 `
  --lateral-lookahead-speed-gain-s 0.35 `
  --lateral-lookahead-min-m 10 `
  --lateral-lookahead-max-m 35 `
  --lateral-steer-gain 4.5 `
  --lateral-steer-rate 2.5 `
  --lateral-high-speed-steer-limit 0.45 `
  --lateral-cross-track-gain 0.55 `
  --lateral-cross-track-softening-mps 6 `
  --longitudinal-controller profile-pid `
  --longitudinal-profile-prepared training/tcp_lagunaseca_v4 `
  --longitudinal-profile-quantile 0.50 `
  --longitudinal-preview-decel-mps2 14 `
  --longitudinal-kp 0.15 `
  --longitudinal-ki 0.01 `
  --longitudinal-kd 0 `
  --longitudinal-target-scale 0.85 `
  --speed-profile-guard `
  --speed-profile-quantile 0.50 `
  --steer-limit 1.0 `
  --throttle-limit 1.0 `
  --brake-limit 1.0 `
  --max-speed-mps 120 `
  --max-racing-line-deviation-m 10 `
  --stop-on-track-out `
  --save-frames-hz 0 `
  --audio-cues `
  --confirm-steer-map-calibrated `
  --confirm-vjoy-mapped
```

처음부터 60초를 실행하지 말고 다음 순서를 권장한다.

1. 정지 상태에서 F8 확인
2. 낮은 제한으로 10~20초 실행
3. Steering sign과 pedal mapping 확인
4. 60초 실행
5. 로그 확인 후 120초 이상으로 연장

## 제어 구조

### 횡방향

모델의 direct steer 출력은 사용하지 않는다.

```text
V4 predicted waypoints
        ↓
speed-aware pure pursuit
        +
Stanley-style cross-track correction
        ↓
steer rate/limit/map
        ↓
vJoy steering
```

Stanley-style 항은 학습 steer를 보조하는 것이 아니라 pure-pursuit steering의 racing-line 복귀를 보조한다.

### 종방향

모델의 direct throttle/brake 출력도 사용하지 않는다.

```text
normal train expert speed profile
        ↓
0~150 m braking preview
        ↓
PI/PID
        ↓
speed guard + slip throttle cut
        ↓
throttle / brake
```

Model command phase와 speed-profile phase는 분리돼 있다. 모델은 학습과 동일한 lap phase를 받고, speed profile만 출발 완료 후 flying phase로 latch된다.

## V4~V8 실험 요약

| 버전 | 목적 | 결과 |
|---|---|---|
| V4 | Normal/recovery/DAgger와 dense planning | Waypoint planner를 최종 구조에 사용 |
| V5 | Predicted plan을 소비하는 control head | Target-speed observational shortcut 발견 |
| V6 | Counterfactual causal supervision | Causal gate 통과 후에도 실제 braking onset이 늦음 |
| V7 | 50 ms control-label alignment | Delay만으로 off-policy drift를 해결하지 못함 |
| V8 | Single-session 장기 수렴 | Offline은 개선됐지만 direct steering/pedal 폐루프 불안정 지속 |

V8의 future-speed prediction은 held-out 동일 session에서 낮은 error를 기록했다. 그러나 target-speed 입력은 외부 expert profile이며, 좋은 speed plan이 있어도 direct pedal head가 braking을 제때 실행하지 못한 사례가 있었다.

## 로그

각 closed-loop run은 다음을 저장한다.

```text
training/tcp_lagunaseca_v4/closed_loop/<timestamp>/
├─ config.json
├─ closed_loop.jsonl
├─ summary.json
└─ frames/                 # 요청한 경우만
```

주요 확인값:

- `effective_hz`, `deadline_misses`, pipeline timing
- `pred_waypoints_m`
- pure-pursuit lookahead와 curvature
- cross-track correction
- nearest racing-line deviation
- spline-indexed diagnostic deviation
- profile/PID target과 limiting lookahead
- speed guard와 slip throttle cut
- model lap phase와 speed-profile lap phase
- 실제 sent steer/throttle/brake

시각화 도구:

```powershell
python tools/visualize_closed_loop.py --help
python tools/scrub_waypoints.py --help
python tools/viz_waypoints_offline.py --help
python tools/visualize_racing_line_alignment.py --help
```

## 저장소 구조

```text
Summer_USRG/
├─ src/ac_dataset/
│  ├─ recorder/             # Raw 수집과 동기화
│  ├─ control/              # vJoy bridge, client, calibration
│  └─ tcp/                  # Prepare, model, train, eval, live control
├─ configs/                 # Ferrari SF15-T steering calibration
├─ analysis/                # Dataset 분포 분석
├─ docs/images/             # README용 검증 이미지
├─ docs/experiments/        # V4~V8 실험 절차와 원인 감사 기록
├─ scripts/experiments/     # 버전별 prepare/train/evaluate 실행 진입점
├─ tests/                   # Unit tests
├─ tools/                   # Export, analysis, installer 보조 코드
├─ SETUP.md                 # Assetto Corsa/CSP 설치
└─ README.md
```

## 테스트

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Pytest를 사용하는 경우:

```powershell
python -m pytest -q
```

게임 연결이 필요한 live test와 offline unit test는 분리돼 있다.

## 데이터와 Git 정책

다음 항목은 Git에 올리지 않는다.

- `dataset/`
- `dataset_v2/`
- `training/`
- Checkpoint와 TensorBoard log
- 로컬 실행 영상과 임시 frame
- vJoy 설치 실행 파일
- 개인 연구 정리 문서

공개 checkpoint가 필요하면 Git commit이 아니라 GitHub Release 또는 별도 model storage를 사용한다.

## 알려진 한계

- 최종 모델은 Laguna Seca와 Ferrari SF15-T 중심이다.
- Static racing-line prior와 track localization이 필요하다.
- Camera-only E2E가 아니라 map-conditioned planner다.
- Tire temperature, wheel slip과 downforce를 model input으로 직접 사용하지 않는다.
- V4 waypoint uncertainty를 별도로 추정하지 않는다.
- 안정 대표 run은 확보했지만 동일 초기조건의 반복 성공률 통계가 더 필요하다.

## 다음 연구

다음 학습은 V9 pure conditional E2E를 별도 실험으로 진행할 예정이다.

- 최근 4~8 frame을 사용하는 temporal input
- Previous action과 physical steering state 입력
- 실제 actuator delay에 맞춘 label alignment
- Homogeneous expert policy 중심 dataset
- Turn-in, apex, counter-steer와 braking transition balancing
- Future waypoint와 future speed auxiliary task 유지
- Direct steer/throttle/brake 재학습
- On-policy recovery/DAgger 수집
- 최종 평가는 offline MAE가 아니라 반복 closed-loop completion으로 수행

현재 V4 hybrid controller를 고정 baseline으로 두고, 새 learned controller가 동일 초기조건에서 완주율과 deviation을 개선할 때만 실제 개선으로 판단한다.

## 참고 프로젝트

- [Assetto Corsa Gym](https://github.com/dasGringuen/assetto_corsa_gym)
- [TCP: Trajectory-guided Control Prediction](https://github.com/OpenDriveLab/TCP)
- [Summer_USRG_Validation](https://github.com/songdavid0206/Summer_USRG_Validation)

본 저장소는 Assetto Corsa Gym의 환경 구성과 TCP의 trajectory/control 공동 학습 아이디어를 참고했지만, 데이터 수집·렌더 정합·검증·모델 입력·closed-loop controller는 본 연구 환경에 맞게 별도로 구현했다.
