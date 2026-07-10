"""acd - Assetto Corsa data.acd 복호화 + 차량 제원 읽기
======================================================

AC 차량 데이터(steer lock, 충돌박스 치수 등)는 `<car>/data.acd` 안에 암호화돼 있다.
암호는 폴더명으로 만든 키를 쓰는 단순 ROT(바이트 덧셈). Content Manager 없이 직접 푼다.

알고리즘 출처: 0danny/AssettoTools ACDBackend/ACDEncryption.cs (공개).
- 키 = "octet1-...-octet8" 문자열(폴더명 소문자에서 8개 정수 유도, 하이픈으로 join).
- 복호화: plain[i] = (cipher_byte[i] - key[i % len(key)]) mod 256.
- 컨테이너: [int32 -1111][int32 version] 다음 엔트리 반복 =
  [int32 이름길이][이름][int32 데이터길이 N][N*4 바이트] — 각 원본 바이트가 4바이트 필드에
  들어가 있어(첫 바이트만 유효) 실제 크기의 4배.

주의: 위 키 알고리즘은 일부 폴더명에서 정수 나눗셈 경계 때문에 octet 하나가 ±1 어긋날 수 있다.
그래서 복호 결과가 '전부 텍스트'가 되도록 위치별 키를 미세보정한다(ini/lut 는 전부 ASCII).
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

_TEXT_OK = set(range(32, 127)) | {9, 10, 13}


def _setup_key(folder_name: str) -> str:
    f = folder_name.lower()
    o = [ord(c) for c in f]
    n = len(f)

    def b(v: int) -> int:
        return (v % 256 + 256) % 256

    k1 = b(sum(o))
    x = 0
    for i in range(0, n - 1, 2):
        x = x * o[i] - o[i + 1]
    k2 = b(x)
    x = 0
    for j in range(1, n - 3, 3):
        x = (x * o[j]) // (o[j + 1] + 27) - 27 - o[j - 1]
    k3 = b(x)
    x = 5763
    for kk in range(1, n):
        x -= o[kk]
    k4 = b(x)
    x = 66
    for l in range(1, n - 4, 4):
        x = (o[l] + 15) * x * (o[l - 1] + 15) + 22
    k5 = b(x)
    x = 101
    for m in range(0, n - 2, 2):
        x -= o[m]
    k6 = b(x)
    x = 171
    for nn in range(0, n - 2, 2):
        x %= o[nn]
    k7 = b(x)
    x = 171
    for i in range(0, n - 1):
        x = x // o[i] + o[i + 1]
    k8 = b(x)
    return "-".join(str(v) for v in (k1, k2, k3, k4, k5, k6, k7, k8))


def _parse_container(raw: bytes) -> dict[str, bytes]:
    """data.acd 컨테이너 -> {엔트리이름: 암호문 바이트열(4바이트당 첫 바이트만)}."""
    off = 0
    if struct.unpack_from("<i", raw, 0)[0] == -1111:
        off = 8  # [-1111][version]
    out: dict[str, bytes] = {}
    while off < len(raw):
        (nlen,) = struct.unpack_from("<i", raw, off); off += 4
        if not (0 < nlen < 500):
            break
        name = raw[off:off + nlen].decode("ascii", "replace"); off += nlen
        (dlen,) = struct.unpack_from("<i", raw, off); off += 4
        out[name] = bytes(raw[off + i] for i in range(0, dlen * 4, 4))
        off += dlen * 4
    return out


def unpack_acd(car_dir: str | Path) -> dict[str, str]:
    """<car_dir>/data.acd 를 풀어 {파일이름: 텍스트내용} 으로 반환.

    폴더명으로 키를 만들고, 복호 결과가 전부 텍스트가 되도록 위치별로 미세보정한다.
    """
    car_dir = Path(car_dir)
    raw = (car_dir / "data.acd").read_bytes()
    entries = _parse_container(raw)
    key = _setup_key(car_dir.name)
    K0 = [ord(c) for c in key]
    L = len(K0)

    # 위치 j(mod L) 마다, 모든 엔트리 바이트를 텍스트로 만드는 키 후보 중
    # 원래 알고리즘 값에 가장 가까운 것을 고른다.
    cols: list[list[int]] = [[] for _ in range(L)]
    for d in entries.values():
        for i, c in enumerate(d):
            cols[i % L].append(c)
    key_bytes = []
    for j in range(L):
        goods = [k for k in range(256)
                 if all(((c - k) % 256) in _TEXT_OK for c in cols[j])]
        if goods:
            key_bytes.append(min(goods, key=lambda k: min(abs(k - K0[j]),
                                                          256 - abs(k - K0[j]))))
        else:
            key_bytes.append(K0[j])

    out: dict[str, str] = {}
    for name, d in entries.items():
        out[name] = bytes((c - key_bytes[i % L]) % 256
                          for i, c in enumerate(d)).decode("utf-8", "replace")
    return out


def _ini_get(text: str, section: str, key: str) -> str | None:
    """아주 단순한 ini 값 읽기(주석 ; 제거). 값 문자열 반환."""
    cur = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("["):
            cur = line.strip("[]").upper()
        elif cur == section.upper() and "=" in line:
            k, v = line.split("=", 1)
            if k.strip().upper() == key.upper():
                return v.split(";")[0].strip()
    return None


def _floats(s: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", s)]


def read_car_specs(car_dir: str | Path) -> dict:
    """차량 제원을 게임 파일에서 직접 읽는다.

    반환: {steer_lock(도), dimension=(length,width,height)[m], mass, source}
    dimension 은 colliders.ini 박스들의 합집합 bounding box.
      축: SIZE/CENTRE = (x=좌우/폭, y=상하/높이, z=앞뒤/길이).
    """
    car_dir = Path(car_dir)
    files = unpack_acd(car_dir)
    car = files.get("car.ini", "")
    col = files.get("colliders.ini", "")

    steer_lock = None
    v = _ini_get(car, "CONTROLS", "STEER_LOCK")
    if v is not None:
        steer_lock = float(_floats(v)[0])

    mass = None
    v = _ini_get(car, "BASIC", "TOTALMASS")
    if v is not None:
        mass = float(_floats(v)[0])

    dimension = None
    lo = [1e9, 1e9, 1e9]
    hi = [-1e9, -1e9, -1e9]
    cur: dict[str, list[float]] = {}
    got = False
    for line in col.splitlines():
        line = line.strip()
        if line.startswith("[COLLIDER"):
            cur = {}
        elif "=" in line:
            k, val = line.split("=", 1)
            k = k.strip().upper()
            if k in ("CENTRE", "SIZE"):
                nums = _floats(val)
                if len(nums) >= 3:
                    cur[k] = nums[:3]
            if "CENTRE" in cur and "SIZE" in cur:
                c, s = cur["CENTRE"], cur["SIZE"]
                for i in range(3):
                    lo[i] = min(lo[i], c[i] - s[i] / 2)
                    hi[i] = max(hi[i], c[i] + s[i] / 2)
                got = True
                cur = {}
    if got:
        # (length=z, width=x, height=y)
        dimension = (hi[2] - lo[2], hi[0] - lo[0], hi[1] - lo[1])

    return {
        "car_model": car_dir.name,
        "steer_lock": steer_lock,
        "dimension": dimension,
        "mass": mass,
        "source": "data.acd (car.ini + colliders.ini)",
    }


def _default_cars_dir() -> Path:
    return Path(r"C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/content/cars")


def read_car_specs_by_model(model: str, cars_dir: str | Path | None = None) -> dict | None:
    """차종 이름으로 제원 읽기. 폴더 없거나 실패하면 None."""
    base = Path(cars_dir) if cars_dir else _default_cars_dir()
    car_dir = base / model
    if not (car_dir / "data.acd").exists():
        return None
    try:
        return read_car_specs(car_dir)
    except Exception:
        return None


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="AC data.acd 에서 차량 제원 읽기")
    ap.add_argument("car", help="차종 이름(예: ks_ferrari_f2004) 또는 차 폴더 경로")
    args = ap.parse_args()
    p = Path(args.car)
    specs = read_car_specs(p) if p.exists() else read_car_specs_by_model(args.car)
    print(json.dumps(specs, indent=2, ensure_ascii=False))
