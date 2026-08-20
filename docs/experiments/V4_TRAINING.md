# TCP-AC v4 학습 구조와 실행 방법

## 검증된 데이터

- 전체: 209,098 frames / 440 trajectories
- train: 157,636 frames / 338 trajectories
- validation: 51,462 frames / 102 trajectories
- train/validation trajectory 중복: 0
- train/validation 원본 session 중복: 0 (recovery/DAgger도 세션 통째 holdout)
- 제외:
  - heavyclouds DAgger `episode_0001`: session index에는 있으나 실제 폴더가 없음
  - midclouds DAgger `episode_0050`: 전 프레임에 CSP 설정 사이드바가 보임
- 대표 이미지 440장: 모두 RGB 1920x945
- future mask 오류: 0
- 학습/실시간 목표속도 최대 차이: `9.42e-7 m/s`
- 동일 AC preset 별칭 통합: `3_clear→clear`, `5_light_clouds→scattered_clouds`, `2_light_fog→mist`

원본 `solo`, `solo_recovery`, `solo_Dagger`의 디렉터리와 CSV 열이 서로 달라도,
prepare 단계에서 공통 manifest로 변환한다. 미래 정답은 반드시 같은
`trajectory_id`(normal lap 또는 recovery episode) 안에서만 조회한다.

## v3와 v4 차이

| 항목 | v3 | v4 |
|---|---|---|
| 데이터 | 정상 solo lap | 정상 solo + synthetic recovery + DAgger |
| split | 정상 session 단위 | 모든 source를 원본 session 단위로 완전 분리 |
| measurement 입력 | ego 6 + line 22 + command 6 = 34 | 기존 34 + 목표속도 12 = 46 |
| 목표속도 12 | 없음 | line 11지점 속도 + 현재 목표속도 오차 1 |
| 미래 label | 6 horizon이 모두 있어야 sample 사용 | 없는 horizon만 mask, episode 경계 연결 금지 |
| sampling | 자연 분포 | normal 70%, synthetic 12%, DAgger 18% 목표 |
| 극단 recovery | 별도 처리 없음 | extreme 합계 1%로 제한 |
| supervisor pedal | 구분 없음 | 개입 프레임 throttle/brake loss 0.5배 |
| 이미지 | brightness만 | brightness/contrast/color + 약한 blur |
| encoder 시작 | 기본 random | ImageNet pretrained 기본 |
| weight decay | `1e-7` | `1e-5` |
| 정밀도 | FP32 | CUDA AMP(FP16 연산 + FP32 loss scaling) |

학습 이미지는 원본을 보존한 채 먼저 모델 입력 크기(512x256)로 BILINEAR
resize한 후 train-only 색상 augmentation을 적용한다. 공유 live 전처리는
변경하지 않는다. 300-batch worker sweep 결과 이 장비에서는 workers=16이
가장 빨랐고(0.257초/batch), 출력 shape와 finite loss가 정상임을 확인했다.
| speed 보조 loss | 0.05 | 기본 0 (속도 head는 호환용으로 유지) |
| 모니터링 | standing/flying | standing/flying + normal/recovery/DAgger |

목표속도는 미래 실제 차량 속도가 아니다. `AI_RACING_LINE.csv`를 현재
`spline_position`으로 조회한 계획값이며, live에서도 똑같이 계산되므로 정답 누출이 아니다.

## v4 입력과 출력

입력:

- RGB 이미지: `3 x 256 x 512`, ImageNet 정규화
- ego: 현재 속도, 종/횡속도, yaw rate, 종/횡가속도 = 6
- racing line: 0/5/10/20/30/40/50/75/100/125/150m의 local 좌표 = 22
- command: standing/flying을 포함한 one-hot = 6
- target speed: 위 11지점 계획속도 + `(현재 지점 계획속도 - 현재속도)` = 12

출력은 v3와 동일하다:

- 현재 `steer/throttle/brake` = 3
- 0.1/0.25/0.5/1.0/1.5/2.0초 waypoint = `6 x 2`
- 같은 6개 horizon의 미래 control = `6 x 3`
- speed head = 1 (v4에서는 학습 loss에 사용하지 않는 호환/진단 출력)

## 명령어

prepare와 감사는 이미 완료되어 있다. 다시 만들거나 검증하려면:

```powershell
python scripts/experiments/prepare_v4.py --dataset-v2 dataset_v2 --normal-prepared training/tcp_lagunaseca_v3_dense --output training/tcp_lagunaseca_v4
python scripts/experiments/audit_v4.py --prepared training/tcp_lagunaseca_v4
```

본 학습:

```powershell
python train.py --prepared training/tcp_lagunaseca_v4 --output training/tcp_lagunaseca_v4/checkpoints --epochs 50 --batch-size 32 --workers 16 --lr 1e-4 --weight-decay 1e-5 --imagenet-pretrained --balanced-sampler --amp --progress-mode quarters --early-stopping-patience 8
```

`best.pt`를 offline 평가한 뒤 shadow, 짧은 guarded closed-loop 순서로 진행한다.
v3 체크포인트는 target-speed 폭 0으로 자동 인식하므로 기존 평가/주행 명령도 유지된다.
