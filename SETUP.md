# SETUP — 새 컴퓨터에서 세팅하기

이 문서 하나로 **새 PC에서 처음부터 세팅 → 실행**까지 따라 할 수 있다.
파이프라인 설계/필드 설명은 [README.md](README.md) 참고.

---

## 0. 먼저 — 뭘 하려는지에 따라 필요한 게 다르다

| 하려는 것 | 게임 환경 필요? | 필요한 것 |
|---|---|---|
| **데이터 수집** (`collect`) — 직접 주행하며 뽑기 | ✅ 필요 | 1번(게임) + 2번(코드) 전부 |
| **데이터 처리/분석만** (`ac-build`, `ac-view`) — 이미 뽑은 raw 가공 | ❌ 불필요 | 2번(코드)만 + 데이터 파일 복사 |

> **git으로 오는 건 우리 코드뿐**이다. AC 게임·Content Manager·CSP·플러그인은 그 PC에 **직접 설치**해야 한다.

---

## 1. 게임 환경 (수집할 때만)

경로는 기본 Steam 설치 기준. 다르면 본인 경로로 바꿀 것.
AC 설치 위치 예: `C:\Program Files (x86)\Steam\steamapps\common\assettocorsa`

### 1-1. Assetto Corsa 설치
- Steam에서 AC 구매/설치.

### 1-2. Content Manager + CSP 설치
- **Content Manager** 다운로드/설치 (AC 실행 런처).
- **CSP (Custom Shaders Patch)** 를 Content Manager로 설치.
  - ⚠️ **CSP 필수** — 우리 플러그인이 `ac.ext_*` 확장 함수를 쓰기 때문. CSP 없으면 플러그인이 안 뜬다.

### 1-3. 플러그인 + 라이브러리 복사 (출처: `dasGringuen/assetto_corsa_gym`)
- 플러그인 폴더 `sensors_par` (약 17개 파일)
  → `...\assettocorsa\apps\python\sensors_par\`
- windows-libs 의 `DLLs` + `Lib` 폴더 내용
  → `...\assettocorsa\system\x64\`
  - ⚠️ 이게 있어야 플러그인의 **소켓 통신(UDP/TCP)** 이 된다. 빠뜨리면 서버가 안 열림.
- ※ vJoy는 **필요 없음.** 우리는 제어를 보내지 않고 state를 **읽기만** 하므로 (사람 주행 비개입).

### 1-4. 플러그인 소스 패치 (⚠️ 새로 설치할 때마다 재적용)
설치한 `apps\python\sensors_par\sensors_par.py` 를 아래처럼 수정해야 **상대차 서버(2346)** 가 정상 동작한다.
(안 하면 `LookupError: unknown encoding: idna` 로 상대차 서버가 조용히 죽음)

1. 모듈 상단 `sys.path` 설정 직후에 idna 코덱 사전 로드 추가:
   ```python
   import encodings.idna
   "0.0.0.0".encode("idna")
   ```
2. `opponents_server_task` 전체를 `try/except` + `logger.exception(...)` 으로 감싸기
3. 소켓에 `SO_REUSEADDR` 설정
4. 상대차 송신 루프에 `time.sleep(0.04)` 추가

> 패치는 **세션 재로드**(트랙 메뉴 나갔다 재진입) 시 적용된다. `.bak` 백업을 떠두면 편함.

### 1-5. 플러그인 활성화
- AC/Content Manager: **Settings > Assetto Corsa > Apps (또는 UI Modules)** 에서 `sensors_par` 켜기.

### (참고) 플러그인이 여는 서버
- `ego_server` **UDP 2345** — 자차 state(약 90필드) 스트리밍
- `opponents_server` **TCP 2346** — 상대차 (상대 있는 세션에서만 열림)
- `mgmt` **TCP 2347**

---

## 2. Python + 우리 코드

### 2-1. Python 설치
- Python **3.9 이상** 설치 (설치 시 "Add to PATH" 체크).
- 확인: 터미널에서 `python --version`

### 2-2. 코드 받고 설치
```bash
git clone https://github.com/songdavid0206/Summer_USRG.git
cd Summer_USRG
pip install -e .              # 필요한 라이브러리 자동 설치 + collect/ac-build/ac-view 명령 생성
pip install -e ".[capture]"   # (선택) Windows 빠른 캡처 백엔드 dxcam 까지
```
- 설치되는 라이브러리: numpy, pandas, pyarrow, pillow, mss (+선택 dxcam)
- 설치 후 `collect`, `ac-build`, `ac-view` 명령을 어느 폴더에서든 쓸 수 있다.

---

## 3. 실행

### 데이터 수집 (게임 + 플러그인 + CSP 켜고, 트랙에 진입한 상태)
```bash
cd <프로젝트 루트>            # ★ dataset/ 폴더가 보이는 곳에서 실행 (src 안 아님!)
collect                       # 기본 telemetry 50Hz / image 20Hz, 주 모니터 전체 캡처
# 주행 → Alt+Tab 으로 터미널 이동 → Ctrl+C 로 종료
#   (게임이 키보드를 잡으므로 Alt+Tab 필수)
```
- 종료 시 자동으로 `camera/ ego_state/ opponent/` 폴더 + parquet + csv 생성.
- 게임 창만 캡처(속도↑): `collect --capture-region L,T,W,H` (예: `0,0,1280,720`)
- ⚠️ **실행 위치 주의**: `collect`는 "현재 터미널 폴더"의 `dataset/`에 저장한다.
  꼭 `dataset/`가 보이는 프로젝트 루트에서 실행할 것. (`src/`에서 하면 엉뚱한 곳에 생김)

### 이미 뽑은 raw를 (재)정렬 / 확인 (게임 불필요)
```bash
ac-build dataset/<run_id>                                   # raw → 폴더 구조 재생성
ac-view dataset/<run_id>/ego_state/ego_state.parquet --csv  # 미리보기/CSV
python tests/test_offline.py                                # 게임 없이 로직 검증
```

---

## 4. 결과물 구조

```
dataset/<run_id>/                     # run_id = 수집 시각
    camera/000000.png ... + camera_index.parquet/csv    # 이미지 + 인덱스
    ego_state/ego_state.parquet/csv                     # 자차 state + steer/throttle/brake
    opponent/opp_<id>/opp_<id>.parquet/csv              # 상대차별 (상대 있을 때만)
    telemetry_raw / opponents_raw / image_index / meta.json   # raw 소스(백업)
```
camera / ego_state / 각 opp 폴더는 **행 개수가 같고** row i 끼리 1:1 대응. (필드 설명은 README 참고)

---

## 5. 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `state 수신 실패` (collect 시작 시) | 아직 트랙에 진입 안 함. **주행 화면(트랙 위)** 에서 실행. 플러그인/CSP 켜졌는지 확인 |
| 상대차 폴더가 안 생김 | 혼자 주행이면 정상. 상대차 원하면 **RACE 모드로 AI** 넣기 |
| 상대차 서버(2346) 안 열림 | 1-4 플러그인 패치(idna) 재적용했는지 확인. 세션 재진입 후 재시도 |
| 캡처가 검게 나옴 | AC를 **borderless windowed**(테두리 없는 창)로 실행. 전체화면 독점은 캡처 실패 가능 |
| 수집 속도가 느림(~5Hz) | dxcam 전체모니터 캡처 병목. `--capture-region`으로 게임창만 잡기 |
| `dataset/`가 엉뚱한 곳에 생김 | `src/`에서 실행함. 프로젝트 루트(`dataset/` 보이는 곳)에서 실행 |

---

## 6. 참고 / 남은 개선

- **HUD**: 속도계·맵이 이미지에 찍히면 학습이 편법으로 흐를 수 있음 → 게임 HUD 끄거나 region으로 crop.
- **영상 세팅**: borderless windowed + FPS 제한(예: 50) 권장.
- **수집 속도**: 목표 50/20Hz 대비 실측 ~5Hz (화면캡처 병목). region 지정 또는 캡처 스레드 분리로 개선 여지.
- **상대차 필드 확장**: 플러그인 `Opponent.update()` 에 채널 추가 가능(속도/기어/RPM 등). 단 타차 조작값은 AC가 주지 않아 불가.
