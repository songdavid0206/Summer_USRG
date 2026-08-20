"""Hard gates for the V8 single-session convergence experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_v5 import TCPDatasetV5
from .profile_v5 import sha256_file


def audit_v8(prepared: Path) -> dict[str, object]:
    table = pd.read_parquet(prepared / "manifest.parquet")
    split = json.loads((prepared / "split.json").read_text(encoding="utf-8"))
    schema = json.loads((prepared / "schema.json").read_text(encoding="utf-8"))
    quality = json.loads(
        (prepared / "quality_exclusions.json").read_text(encoding="utf-8"))
    train_groups, val_groups = set(split["train_groups"]), set(split["val_groups"])
    train = table[table.trajectory_id.isin(train_groups)]
    val = table[table.trajectory_id.isin(val_groups)]
    artifact_hashes_valid = all(
        sha256_file(prepared / name) == expected
        for name, expected in schema["artifacts"].items())
    standing = set(table.loc[table.lap_phase.eq("standing"), "trajectory_id"])
    required = {
        "one_session": table.session.nunique() == 1,
        "normal_only": set(table.source_type.astype(str)) == {"normal"},
        "exact_laps": sorted(table.lap.astype(int).unique()) == list(range(1, 8)),
        "train_laps": sorted(train.lap.astype(int).unique()) == [1, 2, 3, 4, 5],
        "val_laps": sorted(val.lap.astype(int).unique()) == [6, 7],
        "five_two_groups": len(train_groups) == 5 and len(val_groups) == 2,
        "zero_overlap": not train_groups & val_groups,
        "standing_train_only": bool(standing) and standing <= train_groups and not standing & val_groups,
        "expected_counts": int(train.eligible.sum()) == 13918 and int(val.eligible.sum()) == 5503,
        "quality_exclusions_applied": (
            quality.get("source_rows") == 19424 and
            quality.get("retained_rows") == 19421 and
            quality.get("excluded_rows") == 3 and
            not table.duplicated(["trajectory_id", "source_index"]).any()
        ),
        "artifact_hashes_valid": artifact_hashes_valid,
    }
    train_set = TCPDatasetV5(prepared, "train", augment=False)
    val_set = TCPDatasetV5(prepared, "val", augment=False)
    standing_item = train_set[int(np.flatnonzero(
        train_set.samples.lap_phase.eq("standing").to_numpy())[0])]
    flying_item = train_set[int(np.flatnonzero(
        train_set.samples.lap_phase.eq("flying").to_numpy())[0])]
    command_contract = {
        "standing": standing_item["command"].numpy().tolist(),
        "flying": flying_item["command"].numpy().tolist(),
    }
    required["shared_model_phase_contract"] = (
        command_contract["standing"] == [0, 0, 0, 1, 1, 0] and
        command_contract["flying"] == [0, 0, 0, 1, 0, 1])
    result = {
        "passed": all(required.values()), "checks": required,
        "session": str(table.session.iloc[0]), "rows": int(len(table)),
        "train_samples": len(train_set), "val_samples": len(val_set),
        "train_laps": sorted(train.lap.astype(int).unique().tolist()),
        "val_laps": sorted(val.lap.astype(int).unique().tolist()),
        "command_contract": command_contract,
        "quality_exclusions": quality,
    }
    if not result["passed"]:
        raise RuntimeError(f"V8 audit failed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit TCP-AC V8")
    parser.add_argument("--prepared", type=Path, default=Path("training/tcp_lagunaseca_v8"))
    args = parser.parse_args()
    print(json.dumps(audit_v8(args.prepared), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
