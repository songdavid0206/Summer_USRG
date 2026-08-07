# Summer_USRG — Assetto Corsa Driving Dataset Pipeline

Assetto Corsa의 주행 화면, ego telemetry, 운전 조작값, 상대차 상태와 충돌 GT를
동일한 시간축에 정렬하여 모방학습용 데이터셋으로 만드는 Windows 기반 파이프라인이다.

현재 실험 범위는 **Laguna Seca · Ferrari SF15-T · 40 Hz**이며, solo 정상 주행과
multi 상호작용 주행을 함께 수집한다. 대용량 원본 데이터는 Git에 올리지 않고 코드,
스키마, 테스트와 분포 분석 결과만 버전 관리한다.

## 저장소 구성

| 저장소 | 역할 |
|---|---|
| **Summer_USRG** (현재 저장소) | 수집, 시간 동기화, 가공, structured export, 자동 품질 검사, 분포 분석 |
| [Summer_USRG_Validation](https://github.com/songdavid0206/Summer_USRG_Validation) | 원본 run을 읽기 전용으로 열어 3D bbox, BEV, 물리 일관성을 독립 검증 |

데이터 흐름은 다음과 같다.

```text
Assetto Corsa
  ├─ 화면(dxcam)
  ├─ sensors_par ego/opponent UDP
  ├─ AC shared memory
  └─ CSP camera-pose bridge
          ↓
dataset/<run_id>/                 # 재가공 가능한 원본
          ↓ build / quality / export
dataset_v2/solo 또는 multi/      # 학습용 structured dataset
          ↓
분포 분석 → manifest/loader → imitation-learning baseline → closed-loop 평가
```

## 현재 구현 상태

- 화면 40 Hz, ego telemetry 100 Hz, 상대차 telemetry 약 40~55 Hz 수집
- AC 게임시간 기반 이미지·telemetry 보간 정합
- CSP 실제 렌더 카메라 pose와 same-render 상대차 6DoF/AABB 기록
- solo 완전 랩 자동 보존 및 목표 랩 도달 시 자동 종료
- multi 상대차 quality/retirement 처리
- 물리 callback 기반 차량·장벽 접촉 GT
- 추월, 추월 시도, 추월당함, side-by-side, close-follow 자동 라벨
- interaction union과 non-interaction의 중복 없는 완전 분할
- Laguna Seca run별·lap별 속도/레이싱라인 오차 분석 뷰어

## 빠른 시작

### 1. 설치

Python 3.9 이상에서:

```powershell
git clone https://github.com/songdavid0206/Summer_USRG.git
cd Summer_USRG
python -m pip install -e ".[capture]"
```

게임, Content Manager, CSP, `sensors_par`, 카메라 브리지 설치는
[SETUP.md](SETUP.md)를 먼저 따른다. Windows가 unsigned launcher를 차단할 수 있으므로
아래 예시처럼 `python -m` 실행을 권장한다.

### 2. 수집

게임 세션에 진입하고 다음 연결이 모두 준비된 상태에서 collector를 실행한다.

```text
telemetry 연결 OK
opponents 연결 OK                  # multi 필수
AC Graphics 공유메모리 = 연결
CSP 실제 카메라 pose = 연결
```

현재 사용 중인 multi 7랩 명령:

```powershell
python -m ac_dataset.collect --session-mode multi --laps 7 `
  --capture-backend dxcam `
  --capture-region 0,80,1920,945 `
  --telemetry-hz 100 --image-hz 40 `
  --jpg-quality 80
```

Solo도 `--session-mode solo`만 바꿔 같은 방식으로 실행한다.

```powershell
python -m ac_dataset.collect --session-mode solo --laps 7 `
  --capture-backend dxcam `
  --capture-region 0,80,1920,945 `
  --telemetry-hz 100 --image-hz 40 `
  --jpg-quality 80
```

주의사항:

- `--laps N`은 Content Manager의 레이스 길이를 변경하지 않는다.
- 중간 랩에서 시작해도 완전한 N랩을 보장하려면 게임 세션에서 결승선을 `N+1`번
  통과할 수 있게 설정한다.
- `dxcam`은 모니터 영역을 캡처하므로 수집 중 다른 창을 게임 위에 띄우지 않는다.
- 정지 출발은 지속 움직임을 확인한 뒤 움직이기 직전의 clean frame 하나부터 보존한다.
- 정상 종료 시 raw build, quality 검사와 `dataset_v2` export가 자동 실행된다.
- 목표 랩 전에 ego가 리타이어하면 `Ctrl+C`로 종료하고 해당 run은 학습 후보에서 제외한다.

주요 옵션:

| 옵션 | 의미 | 현재 권장값 |
|---|---|---|
| `--session-mode` | `solo` 또는 `multi` | 실험에 맞게 지정 |
| `--laps` | 보존할 완전 랩 수 | `7` |
| `--capture-backend` | 화면 캡처 방식 | `dxcam` |
| `--capture-region` | 모니터 캡처 영역 | `0,80,1920,945` |
| `--telemetry-hz` | ego 기록 목표 주파수 | `100` |
| `--image-hz` | 이미지 목표 주파수 | `40` |
| `--jpg-quality` | JPEG 품질 | `80` |
| `--onboard-camera` | `cameras.ini` 카메라 번호 | 기본 `0` |
| `--no-structured-export` | raw까지만 만들기 | 필요할 때만 사용 |

## 데이터 구조

### 원본 run

```text
dataset/<run_id>/
├─ camera/                    # 이미지와 실제 카메라 pose
├─ ego_state/                 # 프레임 정렬 ego 상태/action
├─ opponent/                  # 상대차별 정렬 상태
├─ processed/                 # 품질 검사·interaction 계산 feature
├─ track/                     # racing line과 트랙 형상
├─ image_index.parquet
├─ telemetry_raw.parquet
├─ opponents_raw.parquet
└─ meta.json
```

원본은 재수집 없이 exporter와 라벨을 고칠 수 있는 기준 자료이므로 보존한다.

### Structured dataset v3

```text
dataset_v2/
├─ solo/<session>/
│  ├─ SESSION_INFO.txt
│  ├─ SCHEMA.txt
│  ├─ FRAME_INDEX.csv
│  ├─ TRACK_INFO/
│  ├─ lap_01/{LAP_INFO.txt,CAMERA,EGO_INFO}
│  ├─ lap_02/{...}
│  └─ ...
└─ multi/<session>/
   ├─ SESSION_INFO.txt
   ├─ SCHEMA.txt
   ├─ FRAME_INDEX.csv
   ├─ EXCLUDED_FRAMES.csv
   ├─ TRACK_INFO/
   ├─ INTERACTION_UNION_INDEX.csv
   ├─ PARTITION_INFO.txt
   ├─ INTERACTIONS/
   │  ├─ EVENTS/{overtake,overtaken_by,*_attempt,car_contact,barrier_contact}/
   │  └─ STATES/{side_by_side,close_follow}/
   └─ NON_INTERACTION/
      ├─ 0001/{FRAME_INDEX.csv,SEGMENT_INFO.txt,CAMERA,EGO_INFO,OBJECT_INFO}
      ├─ 0002/{...}
      └─ ...
```

Multi의 핵심 불변식:

```text
INTERACTION_UNION_INDEX ∩ NON_INTERACTION = ∅
INTERACTION_UNION_INDEX ∪ NON_INTERACTION = 전체 clean frame
```

따라서 여러 interaction clip에 중복 포함된 프레임은 union index에서 한 번만 세며,
`NON_INTERACTION`의 번호 폴더는 interaction 사이의 연속적인 25 ms 구간이다.
`EXCLUDED_FRAMES.csv`에는 pre-motion desktop, sync 실패, camera pose 누락 또는
50 ms 초과 pose residual 등 학습에서 제외할 이유가 기록된다.

이미지는 가능한 경우 hard link로 연결하므로 interaction과 non-interaction export가
원본 이미지 용량을 물리적으로 중복 사용하지 않는다.

## 라벨과 interaction

`EGO_INFO/<ms>.txt`에는 pose, 속도, 가속도와 throttle/brake/steer가 들어간다.
`OBJECT_INFO/<ms>.txt`에는 유효한 모든 상대차가 포함되며 `visible`로 카메라 내부 여부를
구분한다. `track_id`는 세션 안에서만 안정적인 ID다.

자동 interaction:

| 그룹 | 라벨 | 기준 |
|---|---|---|
| EVENTS | `overtake`, `overtaken_by` | 차체 전체 통과와 clearance 지속 |
| EVENTS | `overtake_attempt`, `overtaken_by_attempt` | 중심 순서는 바뀌었지만 완전 통과가 유지되지 않음 |
| EVENTS | `car_contact` | CSP `onCarCollision` 물리 callback |
| EVENTS | `barrier_contact` | 높이 기준을 통과한 장벽 접촉 callback |
| STATES | `side_by_side` | 일정 시간 근접 병렬 주행 |
| STATES | `close_follow` | 일정 시간 근접 추종 |

접촉은 거리로 추정하지 않는다. collision callback의 차량 ID는 dataset opponent ID로
정규화한 뒤 거리·상대속도·접촉 위치 같은 부가 라벨을 연결한다.

## 검증과 재가공

원본 run 자동 검사:

```powershell
python -m ac_dataset.validate dataset/<run_id>
```

기존 원본을 최신 structured schema로 다시 export:

```powershell
python -m ac_dataset.structured_export dataset/<run_id>
```

독립적인 시각·물리 검증:

```powershell
python -m pip install -e ..\ac_validation
av-check dataset/<run_id>
av-view dataset/<run_id> --frame 300
```

자세한 사용법은
[Summer_USRG_Validation](https://github.com/songdavid0206/Summer_USRG_Validation)을 참고한다.

## Laguna Seca 분포 분석

현재 선택된 solo 11개와 multi 12개 run을 분석한다.

```powershell
python tools\analyze_lagunaseca_distribution.py
python tools\build_lagunaseca_viewer.py
start analysis\lagunaseca_distribution\run_viewer.html
```

- [분포 분석 보고서](analysis/lagunaseca_distribution/REPORT.md)
- [run/lap 인터랙티브 뷰어](analysis/lagunaseca_distribution/run_viewer.html)

현재 스냅샷:

- 총 23 runs, 154 complete laps, 440,344 frames
- solo 195,147 frames / multi 245,197 frames
- Laguna Seca 정상 주행은 첫 baseline을 학습하기에 충분한 트랙 커버리지를 확보
- 추가 bulk 수집보다 RGB 최종 QC, run 단위 split, loader와 closed-loop 평가가 우선

분포 그림은 이미지와 동기화된 숫자 라벨을 분석한다. RGB의 검은 화면, 심한 blur,
게임 외 화면과 perceptual duplicate 검사는 학습 manifest 생성 전에 별도로 수행해야 한다.

## 테스트

게임 없이 단위 테스트:

```powershell
python -m unittest discover -s tests -v
```

또는 pytest가 설치돼 있다면:

```powershell
python -m pytest -q
```

## 데이터와 Git 정책

- `dataset/`, `dataset_v2/`는 대용량이므로 `.gitignore`에서 제외한다.
- 코드, schema, 분석 스크립트, 작은 CSV 요약과 시각화 결과만 Git에 올린다.
- 원본 run은 로컬/외부 저장장치에 보존하고, 수정된 exporter로 파생 데이터를 재생성한다.
- 실험 비교는 프레임 무작위 분할이 아니라 run 단위 분할을 사용한다.

## 다음 단계

1. RGB 이미지 최종 QC와 제외 manifest 생성
2. run 단위 train/validation split
3. 이미지·ego state·action/trajectory 데이터 로더
4. camera-only imitation-learning baseline
5. Assetto Corsa closed-loop inference
6. 완주율, 랩타임, 충돌, 트랙 이탈과 라인 오차 기반 실패 분석

## Laguna Seca solo TCP baseline

수집이 끝난 Laguna Seca solo run만 사용해 TCP-style baseline을 준비한다. 원본 TCP의
trajectory/control 이중 branch는 유지하고, CARLA Roach teacher 전용 value/feature
distillation은 사용하지 않는다. 미래 waypoint는 40 Hz 기준 0.5, 1.0, 1.5, 2.0초의
실제 ego 위치이며, target point는 현재 진행 위치에서 AI racing line을 따라 50 m 앞이다.
고속 레이싱 분포에 맞춰 waypoint와 target point는 50 m, speed 입력과 speed-head 정답은
90 m/s로 나눈다. 두 스케일은 checkpoint의 `normalization` 메타데이터에도 저장한다.

```powershell
python -m pip install -e ".[train]"
python -m ac_dataset.tcp.prepare
python train.py --batch-size 32 --epochs 50 --imagenet-pretrained
python evaluate.py --checkpoint training/tcp_lagunaseca/checkpoints/best.pt
```

standing/flying lap phase를 구분해 학습한 checkpoint의 전체 validation offline 평가는
prepared dataset과 checkpoint를 모두 phase 버전으로 맞춰 실행한다.

```powershell
python evaluate.py `
  --prepared training/tcp_lagunaseca_phase `
  --checkpoint training/tcp_lagunaseca_phase/checkpoints/best.pt `
  --output training/tcp_lagunaseca_phase/evaluation
```

빠른 실행 확인에는 위 명령 끝에 `--max-batches 10`을 추가한다. 실제 평가에는 마지막
epoch의 `last.pt`가 아니라 validation loss가 가장 낮은 `best.pt`를 사용한다.

`training/`은 대용량 원본 dataset, prepared manifest, checkpoint와 평가 산출물이 들어가는
로컬 작업 디렉터리이므로 Git에서 제외한다. 따라서 모델 weight는 이 저장소에 포함되지
않으며, 다른 컴퓨터에서 추론하려면 별도로 전달받은 checkpoint를 위 경로에 두거나
`--checkpoint`에 실제 위치를 지정해야 한다. 공개 배포가 필요하면 일반 Git 커밋 대신
GitHub Release asset 또는 Git LFS를 사용한다.

준비 결과는 `training/tcp_lagunaseca/`에 생성된다.

- `manifest.parquet`: 이미지·조작·pose를 합친 학습 index
- `split.json`: Laguna Seca solo 9-run train / 2-run validation 분할
- `rgb_qc_samples.parquet`: 약 1 Hz RGB 자동 QC 표본
- `contact_sheet.jpg`: run별 육안 확인 표본
- `summary.json`: 세션·프레임·QC 요약

별도 offline test split은 만들지 않는다. 최종 평가는 Laguna Seca closed-loop 주행으로
수행한다.

학습 후 `evaluate.py`는 전체 validation run을 대상으로 current/future control과 horizon별
waypoint 오차를 계산하고 `training/tcp_lagunaseca/evaluation/`에 loss curve, track error map,
RGB+ego-BEV 정성 표본, `predictions.parquet`, `metrics.json`, `REPORT.md`를 생성한다.

### Read-only live shadow inference

폐루프 제어를 연결하기 전에 모델 예측과 운전자의 실제 입력을 실시간으로 비교한다. 아래
명령은 화면과 telemetry를 읽기만 하며 vJoy/CSP/controller로 값을 보내는 코드가 없다.

```powershell
python shadow.py --duration 60
```

기본값은 학습 때와 같은 `0,80,1920,945` 화면 영역, 40 Hz, `best.pt`, Laguna Seca AI
racing line 50 m lookahead이다. AC 세션에 진입하고 `sensors_par`의 ego UDP server가 켜진
상태에서 실행한다. 실행 후 5초간 화면 전환 시간을 주고, 차량 속도가 1 m/s 이상으로
0.25초간 유지될 때 자동으로 기록을 시작한다. 즉시 중단하려면 `Ctrl+C`를 누른다.
CUDA에서는 학습의 PIL bilinear resize와 수치적으로 동등한 antialiased bilinear resize 및
ImageNet 정규화를 GPU에서 실행한다. CPU device를 선택하면 기존 PIL 전처리를 사용한다.

결과는 `training/tcp_lagunaseca/shadow/<실행시각>/`에 저장된다.

- `config.json`: checkpoint, capture, normalization 및 `actuation_enabled=false` 안전 설정
- `first_frame.jpg`: 기록이 실제 시작된 첫 RGB crop(캡처 내용·구간 확인용)
- `shadow.jsonl`: 매 프레임 예측 steer/signed_accel, 디코드한 throttle/brake, 실제 운전자
  입력, 두 값의 차이, waypoint/future action, inference/loop 시간
- `summary.json`: 처리 프레임 수, 실효 Hz, 누락·반복 프레임 수

`signed_accel = throttle - brake`이므로 모델 출력이 양수면 throttle, 음수면 brake로
해석한다. 이 baseline은 throttle과 brake를 동시에 밟는 상태를 표현하지 못한다.
추가 시각 디버깅이 필요하면 `--save-frame-every 40`으로 약 1초마다 JPEG를 `frames/`에
저장할 수 있다. JPEG 저장 지연이 성능 측정에 섞이므로 순수 timing run에서는 생략한다.

### Guarded vJoy closed-loop test

vJoy 2.1.9.1 device 1의 X/Y/Z를 각각 steer/gas/brake로 사용한다. Content Manager의
`Settings > Assetto Corsa > Controls`에서 input method를 wheel로 바꾸고 세 축을 먼저
매핑해야 한다. 모델에는 gear 출력이 없으므로 첫 테스트에서는 automatic shifting을 켠다.

독립 watchdog bridge를 별도 터미널에서 먼저 실행한다.

```powershell
python vjoy_bridge.py
```

bridge가 vJoy를 소유하며 F8을 누르거나 command packet이 0.15초 끊기면 steer=0,
throttle=0, brake=1을 latch한다. latch 해제는 bridge 재시작으로만 가능하다.

Content Manager에서 각 binding 칸을 선택한 뒤 필요한 축을 움직인다.

```powershell
python vjoy_axis_test.py --steer 0.5 --duration 5 --confirm-axis-test
python vjoy_axis_test.py --throttle 0.5 --duration 5 --confirm-axis-test
python vjoy_axis_test.py --brake 0.5 --duration 5 --confirm-axis-test
```

트랙 화면에서 실제 telemetry 규약까지 확인한다.

```powershell
python vjoy_axis_test.py --steer 0.10 --duration 2 --verify-telemetry --confirm-axis-test
python vjoy_axis_test.py --throttle 0.20 --duration 2 --verify-telemetry --confirm-axis-test
python vjoy_axis_test.py --brake 0.20 --duration 2 --verify-telemetry --confirm-axis-test
```

`telemetry-during`이 각각 steerNorm +0.10, throttle 0.20, brake 0.20 근처인지 확인한다.
steer만 반대 부호라면 closed-loop에 `--steer-sign -1`을 사용한다. pedal이 반대로 움직이면
Content Manager에서 해당 축의 invert 설정을 바꾼다.

첫 모델 제어는 다음 제한 profile로만 실행한다.

```powershell
python closed_loop.py --duration 10 --confirm-vjoy-mapped
```

기본 제한은 steer +/-0.20, throttle <=0.30, speed <=20m/s이며 track-out 시 bridge
emergency를 latch한다. F8을 먼저 실제로 시험하고, 축 부호 확인 전에는 제한을 풀지 않는다.
