# 설치 및 수집 환경 설정

새 Windows PC에서 Assetto Corsa 데이터 수집 환경을 재현하는 문서다. 처음 설치한다면 위에서부터
차례대로 진행한다. 데이터 형식과 내부 동기화 원리는 [README](README.md)를 참고한다.

## 목차

- [1. 필요한 구성요소](#1-필요한-구성요소)
- [2. Assetto Corsa와 Content Manager](#2-assetto-corsa와-content-manager)
- [3. sensors_par 플러그인](#3-sensors_par-플러그인)
- [4. CSP 렌더 pose 브리지](#4-csp-렌더-pose-브리지)
- [5. Python 패키지 설치](#5-python-패키지-설치)
- [6. Content Manager 세션 설정](#6-content-manager-세션-설정)
- [7. 화면과 카메라 설정](#7-화면과-카메라-설정)
- [8. 수집 실행](#8-수집-실행)
- [9. 정상 동작 확인](#9-정상-동작-확인)
- [10. 문제 해결](#10-문제-해결)

## 1. 필요한 구성요소

| 구성요소 | 용도 | 필수 여부 |
|---|---|---|
| Assetto Corsa (Steam) | 시뮬레이터 | 수집 시 필수 |
| Content Manager | 세션·차량·트랙·CSP 설정 | 권장 |
| Custom Shaders Patch (CSP) | Lua 앱과 실제 렌더 pose 제공 | 필수 |
| `sensors_par` | ego/opponent telemetry 서버 | 필수 |
| AC Dataset Camera Pose Bridge v3 | 실제 카메라·same-render 차량 상태·충돌 GT 기록 | 필수 |
| Python 3.9 이상 | 수집·정렬·검증 CLI | 필수 |
| `dxcam` | Windows 고속 화면 캡처 | 권장, 없으면 mss fallback |

> Git 저장소에는 게임과 Content Manager가 포함되지 않는다. AC·Content Manager·CSP·`sensors_par`는
> PC에 별도로 설치해야 한다. vJoy는 제어를 전송하지 않는 현재 수집 파이프라인에는 필요 없다.

## 2. Assetto Corsa와 Content Manager

1. Steam에서 Assetto Corsa를 설치한다.
2. [Content Manager 공식 페이지](https://acstuff.club/app/)에서 내려받아 AC 설치 경로를 연결한다.
3. Content Manager의 `Settings > Custom Shaders Patch`에서 CSP를 설치하고 활성화한다.
   CSP의 공개 소스·설정 자료는 [ac-custom-shaders-patch](https://github.com/ac-custom-shaders-patch)에서
   확인할 수 있다.

기본 Steam 경로는 다음과 같다.

```text
C:\Program Files (x86)\Steam\steamapps\common\assettocorsa
```

다른 Steam Library를 사용한다면 이후 모든 경로를 실제 설치 위치로 바꾼다.

## 3. `sensors_par` 플러그인

기반 플러그인은
[`dasGringuen/assetto_corsa_gym`](https://github.com/dasGringuen/assetto_corsa_gym)의
Assetto Corsa Python 앱을 사용한다.

### 3.1 파일 설치

```text
sensors_par
  → <AC>\apps\python\sensors_par

windows-libs의 DLLs/Lib 내용
  → <AC>\system\x64
```

Content Manager의 `Settings > Assetto Corsa > Apps` 또는 `Python Apps`에서 `sensors_par`를 켠다.

### 3.2 이 프로젝트에서 사용한 패치

원본 `sensors_par`에 다음 변경이 적용돼 있어야 한다.

| 파일 | 변경 | 목적 |
|---|---|---|
| `sensors_par.py` | `encodings.idna` 사전 로드 | AC 내장 Python의 socket encoding 초기화 실패 방지 |
| `sensors_par.py` | `SO_REUSEADDR`, 서버 전체 예외 로그 | 세션 재시작 시 2346 bind 실패 진단 |
| `sensors_par.py` | opponent loop `time.sleep(0.008)` | 약 50 Hz 상대차 송신 |
| `structures.py` | `steerNorm=info.physics.steerAngle` | IL용 정규화 조향 라벨 수집 |
| `structures.py` | ego tyre contact points | 좌표·카메라 검증 보조값 |
| `structures.py` | opponent velocity/body pose | 상대차 속도·차체 방향·휠베이스/윤거 기록 |

수정 전 파일은 `.bak`으로 보관한다. 플러그인을 다시 설치하거나 덮어썼다면 패치도 다시 적용해야
하며, 변경 후에는 트랙에서 나갔다가 재진입해 Python 앱을 다시 로드한다.

플러그인이 여는 포트:

- UDP `2345`: ego state
- TCP `2346`: opponents
- TCP `2347`: management

## 4. CSP 렌더 pose 브리지

저장소의 다음 폴더를 AC Lua 앱 폴더로 복사한다.

```text
Summer_USRG\csp_apps\ac_camera_pose_bridge
  → <AC>\apps\lua\ac_camera_pose_bridge
```

복사 후 트랙에 진입하고 오른쪽 앱 사이드바에서 **AC Dataset Camera Pose**를 실행한다.

브리지 v3가 매 CSP `script.update` 및 물리 충돌 callback에서 기록하는 값:

- 실제 렌더 카메라 `position / forward / up / FOV`
- 전체 차량 `position / look / up`
- 차량 visual AABB
- render frame, game/lap time, 차량 state timestamp
- 자차 충돌 시작, 충돌 상대(차량/벽), 차체 기준 접촉 위치, 상대속도, 손상 구역

이미지 픽셀이나 이미지 검출 결과는 브리지 입력으로 사용하지 않는다.

## 5. Python 패키지 설치

PowerShell에서 다음을 실행한다.

```powershell
git clone https://github.com/songdavid0206/Summer_USRG.git
git clone https://github.com/songdavid0206/Summer_USRG_Validation.git

cd Summer_USRG
python -m pip install -e ".[capture]"

cd ..\Summer_USRG_Validation
python -m pip install -e .
```

설치 확인:

```powershell
collect --help
ac-validate --help
av-view --help
av-check --help
```

## 6. Content Manager 세션 설정

사용자가 첨부한 화면은 `DRIVE > single > QUICK`이다. 데이터 수집에는 **RACE** 세션을 사용한다.

### 클릭 순서

1. 상단 `DRIVE`
2. `single`
3. `QUICK`
4. 차량과 트랙 선택
5. 중앙 세션 탭에서 `RACE`
6. `Grid type = Same car`
7. `Opponents = 5`
8. `Starting position = Last`
9. 오른쪽 아래 `Go!`

현재 재현 예시:

| 항목 | 화면 예시 | 필수 여부 |
|---|---|---|
| Car | Ferrari SF15-T | 원하는 학습 대상에 맞게 변경 가능 |
| Track | Laguna Seca | 트랙별로 별도 run 권장 |
| Session | RACE | 상대차 수집 시 필수 |
| Race duration | 1 lap | 테스트용; 장시간 수집 시 조절 |
| Grid type | Same car | 차종 통일 실험에 권장 |
| Opponents | 5 | 현재 검증 기준값; 변경 가능 |
| Starting position | Last | 추월 장면 확보에 권장 |
| Strength | `72.5 ± 2.5%` | 난이도/속도 분포에 따라 조절 |
| Aggression | `98.3 ± 1.8%` | 근접·추월 상황 확보용 예시 |
| Weather | Random | 다양성 확보용; 재현 실험은 고정값 사용 |
| Track condition | Green | 화면 예시이며 필수값 아님 |
| Penalties | ON | 선택 |
| Jump-start penalty | Disabled | 수집 편의용 예시 |

강도·공격성·날씨는 파이프라인 필수 설정이 아니다. 실험 재현이 목적이면 값을 고정하고
`meta.json` 외에 실험 노트에도 함께 남긴다.

### 사람 주행과 AC 내장 AI

- 사람 주행: 트랙 진입 후 직접 운전한다.
- AI 전문가 주행: 주행 화면에서 `Ctrl+C`로 AC 내장 AI를 켠 뒤 수집한다.
- 게임 창의 `Ctrl+C`는 AI 토글이고, PowerShell의 `Ctrl+C`는 수집 종료다.

## 7. 화면과 카메라 설정

### 7.1 화면

현재 검증된 조합:

| 항목 | 값 |
|---|---|
| 게임 렌더 해상도 | `1920 × 1080` |
| 캡처 영역 | `0,80,1920,945` |
| 이미지 출력 | JPG |
| 목표 image rate | `40 Hz` |
| 목표 telemetry rate | `100 Hz` |

HUD·앱 창·세션 배너는 가능한 한 끈다. 현재 capture region은 위 80px와 아래 55px를 제외해
순수 주행 화면을 `1920×945`로 저장한다. 검은 화면이 캡처되면 독점 전체화면 대신 borderless
windowed로 바꾼다.

### 7.2 카메라 선택

1. 트랙 진입 후 `F6`으로 온보드 카메라를 순환한다.
2. 수집에 사용할 시점을 CAMERA_0으로 맞춘다.
3. 한 run 안에서는 같은 시점을 유지한다.
4. 실행 명령에는 `--onboard-camera 0`을 넣는다.

`--onboard-camera 0`은 차종·FOV metadata와 구 run fallback을 지정한다. 새 run의 실제 extrinsic은
CSP 브리지가 매 프레임 직접 저장하므로 카메라가 움직여도 pose가 함께 기록된다. F1/F5/F6으로
시점을 바꾸는 것도 기록되지만, E2E 학습 데이터의 분포가 섞이므로 한 run 중 전환은 권장하지 않는다.

### 7.3 카메라 흔들림 설정

Content Manager 버전에 따라 메뉴명이 조금 다를 수 있지만, `Settings > Assetto Corsa > View & UI`
부근에서 다음 값을 사용한다.

```ini
; Documents\Assetto Corsa\cfg\camera_onboard.ini
[MODE]
IS_WORLD_ALIGNED=1
FOV=56

[GFORCES]
MIX=0.00000,0.00000,0.00000

[SHAKE]
SCALE=0.00000,0.00000,0.00000
```

중요한 주의점:

- CAMERA_0 자체 수직 FOV는 차종 `cameras.ini`의 `60°`이며, 실제 CSP FOV가 우선 사용된다.
- `Lock onboard camera to horizon(IS_WORLD_ALIGNED=1)`은 현재 F6 CAMERA_0에서 완전한 roll 고정으로
  작동하지 않았다. 실제 run에서 카메라 roll과 차체 roll의 상관이 `0.99985`였다.
- 따라서 lock 값을 bbox 계산에 가정하지 않는다. CSP가 기록한 **실제 렌더 카메라 up/forward**를 쓴다.
- G-force와 Shake는 실제로 0으로 두어 불필요한 카메라 변동을 줄인다.

## 8. 수집 실행

AC가 트랙 주행 화면에 있고 두 앱이 활성화된 상태에서, `Summer_USRG` 루트에서 실행한다.

```powershell
collect --capture-region 0,80,1920,945 --telemetry-hz 100 --image-hz 40 --onboard-camera 0
```

주행이 끝나면 `Alt+Tab`으로 PowerShell에 포커스를 옮긴 뒤 `Ctrl+C`를 누른다. 종료 후 자동으로
정렬, 트랙 추출, processed 피처 생성, 40 Hz 균일 격자 export가 실행된다.

검증:

```powershell
ac-validate dataset/<run_id>
av-check dataset/<run_id>
av-view dataset/<run_id> --frame 0
```

Validation 저장소에서 실행한다면 run 경로를 `../ac_dataset/dataset/<run_id>`처럼 조정한다.

## 9. 정상 동작 확인

수집 시작 시 반드시 확인할 로그:

```text
[telemetry] 연결 OK
[opponents] 연결 OK
[capture] backend=dxcam
[capture] AC Graphics 공유메모리 = 연결
[capture] CSP 실제 카메라 pose = 연결
```

종료 시 확인할 항목:

- `actual camera pose: N/N (100.0%)`
- `render car states: ... / N frames`
- telemetry 실제 rate가 목표 100 Hz에 근접
- image 실제 rate가 목표 40 Hz에 근접
- export의 `중복 0`
- 소수 `빈칸`은 순간 캡처 stall이며 로그에 개수가 표시됨

`av-view` 시작 로그의 최신 경로 지문:

```text
[view/v14] 투영 카메라 = CSP 실제 렌더 pose
[view] 상대차 3D pose = CSP same-render full 6DoF + visual AABB
```

## 10. 문제 해결

| 증상 | 확인할 것 |
|---|---|
| `ego_server 연결 실패` | 트랙 주행 화면인지, sensors_par가 활성화됐는지 확인 |
| opponents 2346 연결 실패 | RACE에 상대차가 있는지, idna/SO_REUSEADDR 패치와 세션 재진입 확인 |
| CSP pose 연결 실패 | Lua 앱 복사 경로와 AC Dataset Camera Pose 활성화 확인 |
| 카메라 pose coverage < 100% | CSP 앱이 run 도중 꺼졌거나 shared-memory ring을 읽지 못한 것 |
| 검은 캡처 | borderless windowed 사용, region 좌표와 주 모니터 확인 |
| 이미지 Hz 저하 | 다른 대량 I/O 중단, dxcam 설치, 40 Hz 사용 |
| bbox가 카메라 흔들림과 같이 움직임 | 정상: 실제 렌더 pose를 사용하므로 이미지와 함께 움직여야 함 |
| 멀리 있는 박스가 꼬여 보임 | 작은 두 차량 박스/숨은 모서리가 겹친 wireframe일 수 있음; frame·car_id를 분리 확인 |
| `pitch+roll=OFF` 로그 | CSP pose에 telemetry 자세를 중복 적용하지 않는다는 뜻; 실제 카메라 roll이 0이라는 뜻 아님 |

설정이 바뀌면 짧은 10~20초 run을 먼저 수집해 coverage와 bbox를 확인한 후 본 수집을 시작한다.
