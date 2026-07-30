"""Repair legacy dataset_v2 contact labels affected by CSP's +1 car-ID encoding."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .structured_export import _opponent_tables, _write_kv


def _read_kv(path: Path) -> dict[str, str]:
    return {
        key: value
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line
        for key, value in [line.split("=", 1)]
    }


def repair_session(session: Path, dataset_root: Path, apply: bool = False) -> dict:
    session_info_path = session / "SESSION_INFO.txt"
    if not session_info_path.exists():
        return {"session": str(session), "status": "not_managed"}
    session_info = _read_kv(session_info_path)
    run_id = session_info.get("run_id", "")
    run_dir = dataset_root / run_id
    if not run_dir.exists():
        return {"session": str(session), "run_id": run_id, "status": "raw_missing"}

    opponents = _opponent_tables(run_dir)
    ego_velocity = pd.read_parquet(
        run_dir / "processed" / "ego_features.parquet",
        columns=["t_img", "speed", "yaw"])
    collision_path = run_dir / "camera" / "collision_events.parquet"
    collisions = pd.read_parquet(collision_path)
    collisions = {
        int(row["collision_event_seq"]): row
        for _, row in collisions.iterrows()
        if int(row.get("collision_with", 0)) > 0
    }
    info_paths = sorted(
        (session / "INTERACTIONS").rglob("car_contact*/**/INTERACTION_INFO.txt"))
    if not info_paths:
        info_paths = sorted(
            (session / "INTERACTIONS").rglob("car_contact_*/INTERACTION_INFO.txt"))

    repairs = []
    for info_path in info_paths:
        info = _read_kv(info_path)
        try:
            seq = int(info["collision_event_seq"])
        except (KeyError, ValueError):
            continue
        collision = collisions.get(seq)
        if collision is None:
            raise RuntimeError(
                f"{session.name}: no corrected contact for collision_event_seq={seq}")
        raw_id = int(collision["collision_with"])
        opponent_id = raw_id - 1
        table = opponents.get(opponent_id)
        residual_ms = float("nan")
        minimum_distance = float("nan")
        mapping_valid = False
        if table is not None and not table.empty:
            event_time = float(collision["t_img"])
            nearest = int((table["t_img"] - event_time).abs().argmin())
            residual_ms = (
                abs(float(table["t_img"].iloc[nearest]) - event_time) * 1000.0)
            minimum_distance = float(table["distance"].iloc[nearest])
            mapping_valid = bool(
                residual_ms <= 50.0 and np.isfinite(minimum_distance)
                and minimum_distance <= 8.0)
        raw_relative_speed = float(collision["collision_relative_speed_mps"])
        relative_speed = raw_relative_speed
        relative_speed_source = "csp_raw_legacy_unverified"
        if (mapping_valid and table is not None
                and {"opp_vx", "opp_vy"} <= set(table.columns)):
            ego_nearest = int(
                (ego_velocity["t_img"] - float(collision["t_img"])).abs().argmin())
            opponent_nearest = int(
                (table["t_img"] - float(collision["t_img"])).abs().argmin())
            ego_speed = float(ego_velocity["speed"].iloc[ego_nearest])
            ego_yaw = float(ego_velocity["yaw"].iloc[ego_nearest])
            ego_v = np.array([
                ego_speed * np.cos(ego_yaw),
                ego_speed * np.sin(ego_yaw),
            ])
            opponent_v = table[
                ["opp_vx", "opp_vy"]
            ].iloc[opponent_nearest].to_numpy(float)
            if np.isfinite(ego_v).all() and np.isfinite(opponent_v).all():
                relative_speed = float(np.linalg.norm(ego_v - opponent_v))
                relative_speed_source = (
                    "recomputed_from_mapped_ego_opponent_velocity_v1")
        old_opponent = info.get("opponent_id", "")
        info.update({
            "opponent_id": opponent_id,
            "minimum_distance_m": minimum_distance if mapping_valid else float("nan"),
            "detection_method": "csp_onCarCollision_id_plus_one_decoded_v3",
            "collision_with_raw": raw_id,
            "collision_id_decode": "car_id=collision_with-1",
            "contact_time_residual_ms": residual_ms,
            "contact_mapping_valid": mapping_valid,
            "collision_relative_speed_mps": relative_speed,
            "collision_relative_speed_raw_mps": raw_relative_speed,
            "collision_relative_speed_source": relative_speed_source,
            "primary_opponent_id": opponent_id,
            "participant_ids": [opponent_id],
        })
        new_name = re.sub(
            r"_opp\d+$", f"_opp{opponent_id:02d}", info_path.parent.name)
        repairs.append({
            "seq": seq,
            "old_opponent_id": old_opponent,
            "new_opponent_id": opponent_id,
            "minimum_distance_m": minimum_distance,
            "mapping_valid": mapping_valid,
            "relative_speed_mps": relative_speed,
        })
        if apply:
            _write_kv(info_path, info)
            if new_name != info_path.parent.name:
                destination = info_path.parent.with_name(new_name)
                if destination.exists():
                    raise FileExistsError(destination)
                info_path.parent.rename(destination)

    if apply and repairs:
        session_info["collision_id_mapping"] = "car_id=collision_with-1"
        session_info["collision_id_mapping_repaired"] = True
        session_info["collision_id_mapping_repaired_contacts"] = len(repairs)
        _write_kv(session_info_path, session_info)
    return {
        "session": str(session),
        "run_id": run_id,
        "status": "repaired" if apply else "dry_run",
        "contacts": repairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sessions", nargs="+")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    for value in args.sessions:
        result = repair_session(
            Path(value), Path(args.dataset_root), apply=args.apply)
        print(result)


if __name__ == "__main__":
    main()
