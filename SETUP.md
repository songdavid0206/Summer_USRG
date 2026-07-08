# SETUP & 실행 체크리스트

이 문서 하나로 **게임 환경 준비 → 실행 → 게임에 맞춰 확정할 것**을 다 본다.
파이프라인 설계/코드 설명은 [README.md](README.md) 참고.

---

## 0. 큰 그림

```
[준비] AC + 플러그인 설치 + Python 환경
   ↓
[실행] python collect.py  →  주행  →  Ctrl+C
   ↓
[결과] dataset/<run_id>/samples.parquet  (학습용 데이터셋)
```

- **telemetry(차량 상태/조작값/상대차)** = AC 플러그인에서 받음
- **이미지** = 우리가 직접 화면 캡처(dxcam/mss)
- 둘을 같은 시계로 모아 → sync로 병합

---

## 1. 게임/플러그인 설치 (출처: 레포 INSTALL.md)

> 경로는 기본 Steam 설치 기준. 다르면 본인 경로로 바꿀 것.

### 1-1. 플러그인 복사
- `assetto_corsa_gym\AssettoCorsaPlugin\plugins\sensors_par`
  → `C:\Program Files (x86)\Steam\steamapps\common\assettocorsa\apps\python\sensor_par`

### 1-2. 설정/라이브러리 복사 (`assetto_corsa_gym\AssettoCorsaPlugin\windows-libs` 에서)
- `Vjoy.ini` → `C:\Users\%user%\Documents\Assetto Corsa\cfg\controllers\savedsetups`
- `WASD.ini` → `C:\Users\%user%\Documents\Assetto Corsa\cfg\controllers\savedsetups`
- DLL/라이브러리 폴더 → `C:\Program Files (x86)\Steam\steamapps\common\assettocorsa\system\x64`

### 1-3. vJoy 설치
- SourceForge에서 vJoy 다운로드/설치 (AC에 명령을 보내기 위한 가상 조이스틱)
- ※ 우리는 **사람이 직접 주행**하며 제어를 보내지 않으므로, vJoy가 사람 입력과
  충돌하지 않는지 확인 필요 (아래 3번 TODO).

### 1-4. AC에서 플러그인 활성화
- **Options > General > UI Modules** → `sensor_par` 켜기

### 1-5. 컨트롤 설정
- **Options > Controls** → 입력 장치로 vJoy(+WASD) 로드

---

## 2. Python 환경

### 2-1. 화면 캡처용 (우리 파이프라인, Python 3.9+)
```bash
pip install numpy pandas pillow pyarrow      # 저장/동기화
pip install dxcam                            # 화면 캡처(Windows 권장) — 안 되면 pip install mss
pip install omegaconf                        # config.yml 로드 (실제 수집 시)
```

### 2-2. 레포 플러그인/캡처 연동용 (레포 기준)
- AC 내장 Python은 **3.3.5** (수정 불가) → 별도로 `C:\Python33\` 설치(PATH 포함) + `pywin32`
- 레포 캡처 프로세스용 3.9+ 환경에 `mss`, `pygetwindow`, `opencv-python`
- `<AC설치>/apps/python/sensor_par/config.py` 에서 Python 3.9+ 경로 지정(비표준 위치일 때)

> 참고: 우리는 이미지 캡처를 레포 방식(shared memory) 대신 **직접** 하므로,
> 최소한 telemetry만 받으려면 1번 플러그인 + `config.yml`이면 됨.
> 레포의 screen capture 세팅(2-2 일부)은 우리가 안 써도 무방.

---

## 3. 실행

### 지금 (게임 없이 가능)
```bash
# 저장/동기화 로직 검증
python tests/test_offline.py

# 화면 캡처 단독 점검 (실제 화면 한 장 저장)
python recorder/screen_capture.py --save shot.png
python recorder/screen_capture.py --region 0,0,640,480 --save shot.png   # 영역 지정
```

### 게임 붙인 뒤 (진짜 수집)
```bash
# AC 실행 + 플러그인 켜고 주행 준비 상태에서:
python collect.py --config <레포>/config.yml --out ./dataset \
    --telemetry-hz 50 --image-hz 20 --capture-backend auto
# 주행 → Ctrl+C → 자동 flush + sync → dataset/<run_id>/samples.parquet
```
- 게임 창만 캡처: `--capture-region L,T,W,H` (예: `0,0,1280,720`)
- 나중에 raw만 다시 병합: `python recorder/sync.py dataset/<run_id>`

---

## 4. 게임에 맞춰 확정할 것 (코드 TODO 총정리)

| # | 항목 | 위치 | 할 일 |
|---|---|---|---|
| 1 | **조작값 키명** | `schema.py` `ACTION_MAP` | state 실제 키가 `accStatus`/`gas` 중 뭔지, brake 스케일(0~1?) 확인 후 수정 |
| 2 | **상대차 추출** | `collect.py` `extract_opponents()` | AC에서 상대차 뽑히면 `OPPONENT_FIELDS`에 맞춰 채움. 안 되면 비움(파이프라인 정상 동작) |
| 3 | **제어 충돌** | `collect.py` 상단 주석 | 레포 `step()`이 tick 진행에 응답을 요구하면 vJoy 끄기/record 모드 확인 |
| 4 | **AC 화면 모드** | `screen_capture.py` 주석 | **borderless windowed**로 실행 (전체화면 독점은 캡처가 검게 나올 수 있음) |
| 5 | **HUD 처리** | `screen_capture.py` 주석 | HUD 끄거나 `--capture-region`으로 crop (속도계 찍히면 편법 학습 위험) |
| 6 | **컬럼 확정** | 첫 수집 후 `telemetry_raw.parquet` | 컬럼 찍어서 `EGO_KEYS_HINT` 키들이 실제로 나오는지 확인(일부는 config로 켜야 나올 수 있음) |

---

## 5. 조교님께 확인할 것

- 이미지 뷰: cockpit / hood / chase / custom 어느 시점?
- 저장 주기: image 몇 Hz / telemetry 몇 Hz?
- 상대차 정보 수준: 위치·속도·거리만? 주변 N대만?
- 최종 포맷: image folder + parquet 로 OK인지?

---

## 6. 출력물 구조 (수집 성공 시)

```
dataset/<run_id>/
    images/000000.png ...        # 캡처 프레임
    image_index.parquet          # frame ↔ t ↔ image_path
    telemetry_raw.parquet        # tick별 ego + steer/throttle/brake
    opponents_raw.parquet        # (t, opp_id, ...)  ※ 상대차 있을 때만
    samples.parquet              # ← 최종 데이터셋 (image ↔ ego ↔ action)
    meta.json                    # 수집 조건 요약(영수증)
```
