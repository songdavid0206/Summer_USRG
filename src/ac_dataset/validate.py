"""validate - 수집된 run 의 자동 검증 (ac-validate)
===================================================

run 하나를 받아 여러 검사를 돌리고 PASS/FAIL 로 보고한다. 매번 손으로 확인하던 것을
테스트된 코드로 박아, 조용한 퇴행(예: 신규 필드가 필터에 걸려 사라지는 일)을 잡는다.

지금은 **1층: 구조 검사**만 있다. 이후 물리 일관성(2층), 재투영(3층), 시간오프셋(4층)이 추가된다.

  ac-validate dataset/<run_id>

결과는 콘솔 + `<run>/validation.json` 에 저장. 하나라도 FAIL 이면 exit code 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class Check:
    """검사 결과 하나. name / passed / detail."""
    __slots__ = ("name", "passed", "detail")

    def __init__(self, name: str, passed: bool, detail: str = ""):
        self.name = name
        self.passed = passed
        self.detail = detail

    def to_dict(self) -> dict:
        return {"name": self.name, "pass": self.passed, "detail": self.detail}


# --------------------------------------------------------------------------- 1층: 구조
def _check_structure(run_dir: Path, dt_tol_ms: float = 50.0) -> list[Check]:
    checks: list[Check] = []
    cam = pd.read_parquet(run_dir / "camera" / "camera_index.parquet")
    ego = pd.read_parquet(run_dir / "ego_state" / "ego_state.parquet")
    n = len(cam)

    # 1) camera / ego_state / opp_* 행 개수가 전부 같은가
    lengths = {"camera": len(cam), "ego_state": len(ego)}
    opp_files = sorted((run_dir / "opponent").glob("opp_*/opp_*.parquet"))
    for f in opp_files:
        lengths[f.parent.name] = len(pd.read_parquet(f))
    all_same = len(set(lengths.values())) == 1
    checks.append(Check("row_counts_match", all_same,
                        "모든 스트림 %d행" % n if all_same else "불일치: %s" % lengths))

    # 2) jpg 파일 수 = 행 수
    n_jpg = len(list((run_dir / "camera").glob("*.jpg")))
    checks.append(Check("jpg_count_matches", n_jpg == n,
                        "jpg %d / 행 %d" % (n_jpg, n)))

    # 3) 정렬 오차 dt 가 임계 이내 (카메라 프레임에 가장 가까운 telemetry)
    if "dt" in ego.columns:
        dt_max = float(ego["dt"].abs().max()) * 1e3
        checks.append(Check("alignment_dt_ok", dt_max <= dt_tol_ms,
                            "dt 최대 %.4f ms (임계 %.1f)" % (dt_max, dt_tol_ms)))

    # 4) ego_state / processed 에 NaN 없음 (opp 는 미매칭 프레임에 NaN 가능 → present 로 거름)
    ego_nan = int(ego.isna().sum().sum())
    checks.append(Check("ego_no_nan", ego_nan == 0, "NaN %d개" % ego_nan))

    proc = run_dir / "processed"
    if (proc / "ego_features.parquet").exists():
        ef_nan = int(pd.read_parquet(proc / "ego_features.parquet").isna().sum().sum())
        checks.append(Check("ego_features_no_nan", ef_nan == 0, "NaN %d개" % ef_nan))
    # 상대차 피처: present=True 인 행에 NaN 이 없어야 한다
    opp_feat = sorted((proc / "opponent").glob("opp_*_features.parquet"))
    bad = []
    for f in opp_feat:
        o = pd.read_parquet(f)
        if "present" in o.columns:
            sub = o[o["present"].astype(bool)]
        else:
            sub = o
        if int(sub.isna().sum().sum()) > 0:
            bad.append(f.stem)
    checks.append(Check("opp_features_no_nan_when_present", not bad,
                        "OK" if not bad else "NaN 있는 파일: %s" % bad))

    # 5) camera 프레임 시간 단조증가
    t = cam["t_img"].to_numpy()
    mono = bool(np.all(np.diff(t) > 0))
    checks.append(Check("time_monotonic", mono,
                        "단조증가" if mono else "역행/정체 프레임 있음"))

    return checks


def _check_export(run_dir: Path) -> list[Check]:
    """export 산출물(dataset/<run>_<track>_<hz>hz)이 있으면 검사."""
    checks: list[Check] = []
    exports = sorted(run_dir.parent.glob(run_dir.name + "_*"))
    exports = [e for e in exports if (e / "EGO_INFO").is_dir()]
    if not exports:
        return checks
    E = exports[-1]

    def stems(sub, ext):
        return {p.stem for p in (E / sub).glob("*." + ext)}
    cam_ms, ego_ms, obj_ms = stems("CAMERA", "jpg"), stems("EGO_INFO", "txt"), stems("OBJECT_INFO", "txt")

    # 6) 세 폴더 파일명(ms) 집합 일치
    same = cam_ms == ego_ms == obj_ms
    checks.append(Check("export_filenames_match", same,
                        "%d 프레임 일치" % len(ego_ms) if same
                        else "불일치: CAM %d / EGO %d / OBJ %d" % (len(cam_ms), len(ego_ms), len(obj_ms))))

    # 7) ms 파일명 중복 없음 (정수 반올림 충돌)
    ms_ints = sorted(int(s) for s in ego_ms)
    dup = len(ms_ints) - len(set(ms_ints))
    checks.append(Check("export_ms_unique", dup == 0, "중복 %d개" % dup))

    # 8) 리타이어한 상대차가 OBJECT_INFO 에서 빠졌는가
    #    (processed 의 retired 플래그가 True 인 프레임에 그 track_id 가 없어야 함)
    proc = run_dir / "processed" / "opponent"
    cam = pd.read_parquet(run_dir / "camera" / "camera_index.parquet")
    ms_of = {i: int(round(cam["t_img"].iloc[i] * 1000)) for i in range(len(cam))}
    leak = 0
    for f in sorted(proc.glob("opp_*_features.parquet")):
        o = pd.read_parquet(f)
        if "retired" not in o.columns:
            continue
        oid = int(o["opp_id"].iloc[0]) if "opp_id" in o.columns else int(f.stem.split("_")[1])
        for i in np.where(o["retired"].to_numpy())[0]:
            p = E / "OBJECT_INFO" / ("%d.txt" % ms_of[int(i)])
            if not p.exists():
                continue
            ids = [int(l.rsplit(",", 1)[1]) for l in p.read_text().strip().splitlines()[1:]]
            if oid in ids:
                leak += 1
    checks.append(Check("no_retired_ghost_in_export", leak == 0,
                        "유령차 등장 %d회" % leak))

    return checks


def validate_run(run_dir: str | Path, dt_tol_ms: float = 1.0) -> dict:
    run_dir = Path(run_dir)
    checks: list[Check] = []
    checks += _check_structure(run_dir, dt_tol_ms=dt_tol_ms)
    checks += _check_export(run_dir)

    n_pass = sum(c.passed for c in checks)
    n_fail = len(checks) - n_pass
    result = {
        "run_id": run_dir.name,
        "passed": n_fail == 0,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "checks": [c.to_dict() for c in checks],
    }
    (run_dir / "validation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== validate %s ===" % run_dir.name)
    for c in checks:
        print("  [%s] %-34s %s" % ("PASS" if c.passed else "FAIL", c.name, c.detail))
    print("  -> %d PASS / %d FAIL" % (n_pass, n_fail))
    return result


def main():
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="수집 run 자동 검증")
    ap.add_argument("run_dir", help="dataset/<run_id>")
    ap.add_argument("--dt-tol-ms", type=float, default=50.0,
                    help="정렬 dt 허용 오차(ms). build_dataset 의 merge tolerance 와 맞춤")
    args = ap.parse_args()
    res = validate_run(args.run_dir, dt_tol_ms=args.dt_tol_ms)
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
