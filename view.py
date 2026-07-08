"""
parquet 뷰어 - 데이터셋 내용을 터미널에 보거나 CSV로 내보내기
============================================================

사용법:
  python view.py                                # dataset 폴더의 '가장 최근' 수집 자동 선택
  python view.py --csv                          # 최근 수집을 .csv 로 저장 (엑셀로 열기)
  python view.py --all                          # 최근 수집 전체 행
  python view.py dataset/<run_id>               # 특정 run 폴더 지정
  python view.py dataset/<run_id>/samples.parquet   # 특정 파일 지정
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def resolve_path(arg: str | None) -> Path:
    """인자 해석. 파일이면 그대로, 폴더면 samples.parquet, 없으면 dataset 최근 run 자동."""
    if arg:
        p = Path(arg)
        if p.is_file():
            return p
        if p.is_dir():
            direct = p / "samples.parquet"
            if direct.exists():
                return direct
            # dataset 루트를 준 경우: 하위 run 중 최근 것
            root = p
        else:
            raise SystemExit(f"경로 없음: {p}")
    else:
        root = Path("dataset")             # 인자 없으면 기본 dataset 폴더
        if not root.is_dir():
            raise SystemExit("dataset 폴더가 없어요. 먼저 collect.py 를 돌리세요.")

    runs = sorted([d for d in root.iterdir() if (d / "samples.parquet").exists()])
    if not runs:
        raise SystemExit(f"{root} 안에 samples.parquet 가진 수집 폴더가 없어요.")
    latest = runs[-1]                      # 이름이 날짜_시각이라 정렬하면 마지막이 최근
    print(f"[auto] 가장 최근 수집: {latest.name}")
    return latest / "samples.parquet"


def main():
    ap = argparse.ArgumentParser(description="parquet 뷰어")
    ap.add_argument("path", nargs="?", default=None,
                    help="parquet 파일/run 폴더 (생략 시 dataset 최근 수집 자동)")
    ap.add_argument("--all", action="store_true", help="전체 행 출력")
    ap.add_argument("--rows", type=int, default=10, help="미리보기 행 수 (기본 10)")
    ap.add_argument("--csv", action="store_true", help="같은 경로에 .csv 로 저장")
    args = ap.parse_args()

    p = resolve_path(args.path)
    if not p.exists():
        raise SystemExit(f"파일 없음: {p}")

    df = pd.read_parquet(p)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print(f"[file] {p}")
    print(f"[shape] {df.shape[0]}행 x {df.shape[1]}열")
    print(f"[columns] {list(df.columns)}\n")

    print(df if args.all else df.head(args.rows))

    if args.csv:
        out = p.with_suffix(".csv")
        df.to_csv(out, index=False, encoding="utf-8-sig")   # 엑셀 한글 안 깨지게
        print(f"\n[csv 저장] {out}  (엑셀로 열기 가능)")


if __name__ == "__main__":
    main()
