# Summer_USRG

Assetto Corsa에서 사람의 주행 화면과 차량 상태를 함께 수집하고, imitation learning 모델을 학습해 실제 게임에서 시험하기 위한 프로젝트다.

현재 기준 환경은 **Laguna Seca / Ferrari SF15-T / 40 Hz**다. 단순히 화면과 조작값만 저장하는 데서 끝내지 않고, AC game time을 기준으로 telemetry와 render state를 맞추고, 학습 전 검증부터 shadow inference와 vJoy closed loop까지 한 저장소에서 다룬다.

현재 closed loop는 모델이 미래 waypoint를 예측하고, 실제 조향과 페달은 고전 제어기가 맡는 hybrid 구성을 기본으로 사용한다.

```text
RGB + ego state + racing line + target-speed features
                         |
                    TCP-AC V4
                         |
              predicted waypoints (6)
                  /              \
     pure pursuit + CTE       expert speed profile
            steering          + preview PI/PID
                              throttle / brake
```

모델이 보는 racing line과 target speed는 주행 후에만 알 수 있는 정답 라벨이 아니다. 같은 트랙에서 실행 중에도 사용할 수 있는 경로·속도 계획 입력이다. 다만 Laguna Seca에 고정된 map prior이므로 다른 트랙에 그대로 일반화되는 구조는 아니다.

## 들어 있는 것

- DXCAM/MSS/WGC 화면 캡처
- `sensors_par` ego/opponent telemetry 수집
- AC shared memory와 CSP render bridge 기록
- Camera frame 기준 telemetry 보간 및 동기화
- Racing line, track edge, 차량 제원 추출
- Same-render 3D bounding box 생성
- Raw run과 `dataset_v2` structured export
- Normal/recovery/DAgger 학습 데이터 준비
- TCP 계열 모델 학습·평가·shadow inference
- vJoy closed loop와 watchdog
- Waypoint pure pursuit, cross-track correction, speed-profile PI/PID, slip recovery

데이터셋, 학습 결과와 checkpoint는 용량 때문에 Git에 포함하지 않는다.

## 준비물

- Windows 10/11
- Assetto Corsa
- Content Manager
- Custom Shaders Patch(CSP)
- `sensors_par`
- Python 3.9 이상
- 학습과 실시간 추론용 NVIDIA GPU 권장
- Closed loop를 실행할 경우 vJoy 2.1.9.1

게임, CSP app, telemetry server와 camera bridge 설정은 [SETUP.md](SETUP.md)에 정리되어 있다. 수집과 offline 학습만 할 때는 vJoy가 필요 없다.

## 설치

PowerShell에서:

```powershell
git clone https://github.com/songdavid0206/Summer_USRG.git
cd Summer_USRG
python -m pip install --upgrade pip
python -m pip install -e ".[capture,train]"
```

학습 기능이 필요 없다면 다음만 설치해도 된다.

```powershell
python -m pip install -e ".[capture]"
```

설치 확인:

```powershell
python -c "import ac_dataset; print('ac_dataset import OK')"
python -m unittest discover -s tests -p "test_*.py"
```

## 데이터 수집

게임을 주행 화면으로 전환한 뒤 collector를 실행한다. `--capture-region`은 실제 AC 창 위치에 맞춰 바꿔야 한다.

Solo 7 lap:

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

Multi 7 lap:

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

정상 종료하면 raw run build, quality check와 structured export가 이어서 실행된다. 자동 export가 필요 없으면 `--no-structured-export`를 붙인다.

수집 전에 확인할 점:

- `--laps`는 Content Manager의 race length를 바꾸지 않는다. 게임 쪽 lap 수도 충분히 설정해야 한다.
- 다른 창이나 overlay가 AC 화면을 가리면 그대로 이미지에 들어간다.
- Retire, pit 이동, teleport가 발생한 run은 마지막 구간과 제외 목록을 확인한다.
- 연결 상태에는 ego UDP 2345, opponents TCP 2346, AC shared memory, CSP bridge가 표시되어야 한다.

### 저장 구조

Raw run은 원본 자료다.

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

학습과 배포에는 structured dataset을 사용한다.

```text
dataset_v2/
├─ solo/<session>/
│  ├─ FRAME_INDEX.csv
│  ├─ TRACK_INFO/
│  └─ lap_XX/{CAMERA,EGO_INFO}
└─ multi/<session>/
   ├─ FRAME_INDEX.csv
   ├─ EXCLUDED_FRAMES.csv
   ├─ TRACK_INFO/
   ├─ INTERACTIONS/
   └─ NON_INTERACTION/
```

가능한 경우 이미지는 hard link로 연결한다. `dataset/`, `dataset_v2/`, `training/`은 `.gitignore` 대상이다.

## 동기화와 데이터 검증

화면 capture, UDP telemetry와 CSP render callback은 서로 다른 주기로 들어온다. 원본 stream을 따로 보존하고, image의 AC game time을 기준으로 상태를 보간해 한 frame으로 만든다. 위치와 연속값은 선형 보간하고 각도는 sin/cos 공간에서 처리한다.

3D bounding box도 RGB 검출 결과에 맞춰 만든 값이 아니다. 같은 render tick에서 기록한 camera pose, 상대차 position/forward/up과 visual AABB를 image plane으로 투영한다.

![CSP same-render 3D bounding box, BEV and physics validation](docs/images/validation_bbox_and_physics.png)

별도 검증 저장소에서는 이미지 투영과 물리 채널을 함께 확인할 수 있다.

- [Summer_USRG_Validation](https://github.com/songdavid0206/Summer_USRG_Validation)

```powershell
git clone https://github.com/songdavid0206/Summer_USRG_Validation.git
cd Summer_USRG_Validation
python -m pip install -e .
av-check ..\Summer_USRG\dataset\<run_id>
av-view  ..\Summer_USRG\dataset\<run_id> --frame 300
```

분포 분석 결과와 lap viewer는 [`analysis/lagunaseca_distribution`](analysis/lagunaseca_distribution)에 있다.

## TCP 학습

현재 closed-loop 기준 모델은 V4다. V5~V8은 입력과 loss 구성을 바꿔 본 실험 버전이며, 결과와 원인 분석은 [`docs/experiments`](docs/experiments)에 남겨 두었다. 특히 후속 버전의 direct steering/control head는 offline metric과 달리 실제 인계에서 불안정했기 때문에 기본 제어기로 사용하지 않는다.

### V4 데이터 준비

```powershell
python scripts/experiments/prepare_v4.py `
  --dataset-v2 dataset_v2 `
  --normal-prepared training/tcp_lagunaseca `
  --output training/tcp_lagunaseca_v4
```

Prepared dataset에는 split manifest, normalization 통계, racing-line anchor, target-speed profile과 six-waypoint supervision이 들어간다. Split은 frame 무작위 분할이 아니라 run 단위로 유지한다.

### 학습

```powershell
python train.py `
  --prepared training/tcp_lagunaseca_v4 `
  --output training/tcp_lagunaseca_v4 `
  --epochs 60 `
  --batch-size 32 `
  --workers 4 `
  --amp
```

루트의 `train.py`는 짧은 실행 wrapper다. 모델과 학습 loop의 실제 구현은 [`src/ac_dataset/tcp/train.py`](src/ac_dataset/tcp/train.py)와 같은 package 내부에 있다. `evaluate.py`, `shadow.py`, `closed_loop.py`도 같은 방식이다.

### 평가

```powershell
python evaluate.py `
  --prepared training/tcp_lagunaseca_v4 `
  --checkpoint training/tcp_lagunaseca_v4/checkpoints/best.pt `
  --output training/tcp_lagunaseca_v4/evaluation
```

Offline loss만으로 closed-loop 안정성을 판단하지 않는다. Waypoint 오차, 구간별 편향과 실제 인계 결과를 함께 본다.

### Shadow inference

Shadow mode는 모델 출력을 기록하지만 게임 조작은 보내지 않는다. 처음 checkpoint를 시험할 때는 항상 이것부터 실행한다.

```powershell
python shadow.py `
  --prepared training/tcp_lagunaseca_v4 `
  --checkpoint training/tcp_lagunaseca_v4/checkpoints/best.pt `
  --output training/tcp_lagunaseca_v4/shadow `
  --device cuda `
  --hz 40 `
  --duration 60
```

## Closed loop

Closed loop는 모델의 waypoint 판단과 고전 제어를 결합한다.

- Lateral: predicted waypoint → speed/curvature adaptive pure pursuit → cross-track correction
- Longitudinal: normal expert speed profile → braking-distance preview → PI/PID
- Recovery: deviation/slip이 커지면 목표 속도, throttle과 brake를 보수적으로 제한
- Guard: 속도 초과, track-out, telemetry timeout, F8 emergency brake

실행 전에 vJoy axis mapping과 steering calibration을 완료해야 한다. 처음에는 낮은 속도와 짧은 duration으로 시험하고, 사람이 즉시 F8을 누를 수 있는 상태를 유지한다.

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
  --longitudinal-controller profile-pid `
  --longitudinal-profile-prepared training/tcp_lagunaseca_v4 `
  --longitudinal-profile-quantile 0.50 `
  --longitudinal-preview-decel-mps2 14 `
  --longitudinal-kp 0.12 `
  --longitudinal-ki 0.015 `
  --longitudinal-kd 0 `
  --speed-profile-guard `
  --speed-profile-quantile 0.50 `
  --lateral-recovery `
  --max-racing-line-deviation-m 30 `
  --stop-on-track-out `
  --save-frames-hz 0 `
  --audio-cues `
  --confirm-steer-map-calibrated `
  --confirm-vjoy-mapped
```

차량과 vJoy 설정에 따라 제어 파라미터는 다시 맞춰야 한다. 전체 옵션은 `python closed_loop.py --help`에서 확인한다. `--duration`은 양수면 해당 시간만 실행하고, 0 이하면 수동 중지까지 계속 실행한다.

## 저장소 구조

```text
Summer_USRG/
├─ src/ac_dataset/          수집, export, 학습, 제어 구현
├─ csp_apps/                CSP camera/render bridge
├─ configs/                 차량별 steering calibration
├─ scripts/experiments/     V4~V8 실험 entrypoint와 audit
├─ docs/experiments/        버전별 실험 기록
├─ tools/                   분석과 시각화 도구
├─ tests/                   unit tests
├─ analysis/                Laguna Seca 분포 분석 결과
├─ train.py                 학습 wrapper
├─ evaluate.py              평가 wrapper
├─ shadow.py                read-only live inference wrapper
└─ closed_loop.py           guarded vJoy 실행 wrapper
```

버전별 파일이 루트에 흩어지지 않도록 실험용 wrapper는 `scripts/experiments`, 설명은 `docs/experiments`, 재사용 도구는 `tools`에 나눠 두었다.

## 자주 쓰는 명령

```powershell
# Raw run 검사
ac-validate dataset/<run_id>

# Raw/processed frame 확인
ac-view dataset/<run_id>

# Structured export 다시 생성
ac-export-v2 dataset/<run_id>

# 테스트
python -m unittest discover -s tests -p "test_*.py"
```

각 명령의 최신 옵션은 `--help`로 확인하는 것이 가장 정확하다.

## 참고

- [TCP: Trajectory-guided Control Prediction for End-to-end Autonomous Driving](https://arxiv.org/abs/2207.07601)
- [Assetto Corsa Python documentation](https://assettocorsamods.net/threads/documents-for-ac-phyton-api.59/)
- [Custom Shaders Patch](https://acstuff.club/patch/)

이 저장소는 연구와 실험용이다. 실제 도로 주행을 위한 시스템이 아니다.
