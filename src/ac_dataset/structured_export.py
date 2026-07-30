"""Session-oriented TXT export for solo laps and multi-car interactions.

The raw run remains immutable.  This module builds a replaceable ``dataset_v2``
view with the layout agreed for collection and training:

* solo: one folder per complete lap;
* multi: interaction clips plus numbered, contiguous non-interaction segments.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .export_frames import _grid_slots, _link_or_copy, export_run


SCHEMA_VERSION = 3
MANAGED_BY = "ac_dataset.structured_export"


@dataclass(frozen=True)
class Interaction:
    interaction_type: str
    opponent_id: int | None
    event_time: float
    detected_start: float
    detected_end: float
    minimum_distance_m: float
    success: bool | None = None
    detection_method: str = "heuristic_v1"
    collision_event_seq: int | None = None
    collision_with_raw: int | None = None
    collision_id_decode: str | None = None
    contact_time_residual_ms: float | None = None
    contact_mapping_valid: bool | None = None
    collision_depth_m: float | None = None
    collision_position_forward: float | None = None
    collision_position_right: float | None = None
    collision_position_up: float | None = None
    collision_relative_speed_mps: float | None = None
    collision_relative_speed_raw_mps: float | None = None
    collision_relative_speed_source: str | None = None
    collision_damage_0: float | None = None
    collision_damage_1: float | None = None
    collision_damage_2: float | None = None
    collision_damage_3: float | None = None
    collision_damage_4: float | None = None


def _scalar(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.10g}"
    return str(value)


def _write_kv(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{key}={_scalar(value)}\n" for key, value in values.items()),
                    encoding="utf-8")


def _track_name(run_dir: Path) -> str:
    path = run_dir / "track" / "track.json"
    if not path.exists():
        return "unknown"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("track_name") or "unknown")
    except (OSError, ValueError):
        return "unknown"


def _export_track_info(run_dir: Path, session: Path) -> dict:
    """Export reusable racing-line and explicit left/right boundary geometry."""
    source_dir = run_dir / "track"
    line_path = source_dir / "racing_line.parquet"
    target = session / "TRACK_INFO"
    target.mkdir(parents=True, exist_ok=True)
    source_meta_path = source_dir / "track.json"
    source_meta = {}
    if source_meta_path.exists():
        try:
            source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            source_meta = {}

    if not line_path.exists():
        info = {
            "schema_version": SCHEMA_VERSION,
            "available": False,
            "track_name": source_meta.get("track_name", "unknown"),
            "reason": "source racing_line.parquet is missing",
        }
        _write_kv(target / "TRACK_INFO.txt", info)
        return info

    line = pd.read_parquet(line_path).copy()
    required = {"x", "y", "z", "side_left", "side_right"}
    missing = sorted(required.difference(line.columns))
    if missing:
        raise ValueError(f"track racing line is missing columns: {missing}")

    xyz = line[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce")
    xy = xyz[["x", "y"]].to_numpy(float)
    tangent = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    tangent_norm = np.linalg.norm(tangent, axis=1)
    tangent_norm[tangent_norm < 1e-9] = 1.0
    left_normal = np.column_stack(
        [-tangent[:, 1] / tangent_norm, tangent[:, 0] / tangent_norm])
    left_width = pd.to_numeric(line["side_left"], errors="coerce").to_numpy(float)
    right_width = pd.to_numeric(line["side_right"], errors="coerce").to_numpy(float)

    if "dist" in line:
        distance = pd.to_numeric(line["dist"], errors="coerce").to_numpy(float)
    else:
        segments = np.linalg.norm(np.diff(xy, axis=0, prepend=xy[:1]), axis=1)
        distance = np.cumsum(segments)
        line["dist"] = distance
    distance_end = float(np.nanmax(distance)) if len(distance) else 0.0
    normalized = distance / distance_end if distance_end > 0 else np.zeros(len(line))
    line.insert(1, "normalized_position", normalized)

    left_xy = xy + left_normal * left_width[:, None]
    right_xy = xy - left_normal * right_width[:, None]

    def boundary_frame(points: np.ndarray, width: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({
            "i": line["i"] if "i" in line else np.arange(len(line)),
            "normalized_position": normalized,
            "dist_m": distance,
            "x": points[:, 0],
            "y": points[:, 1],
            "z": xyz["z"].to_numpy(float),
            "width_from_racing_line_m": width,
        })

    left = boundary_frame(left_xy, left_width)
    right = boundary_frame(right_xy, right_width)
    line.to_csv(target / "AI_RACING_LINE.csv", index=False)
    line.to_parquet(target / "AI_RACING_LINE.parquet", index=False)
    left.to_csv(target / "LEFT_BOUNDARY.csv", index=False)
    left.to_parquet(target / "LEFT_BOUNDARY.parquet", index=False)
    right.to_csv(target / "RIGHT_BOUNDARY.csv", index=False)
    right.to_parquet(target / "RIGHT_BOUNDARY.parquet", index=False)

    closing_segment = (
        float(np.linalg.norm(xy[0] - xy[-1])) if len(xy) > 1 else 0.0)
    info = {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "track_name": source_meta.get("track_name", "unknown"),
        "points": len(line),
        "lap_length_m": distance_end + closing_segment,
        "coordinate_frame": source_meta.get(
            "coord_frame", "ego world_position x,y,z"),
        "boundary_method": (
            "AI racing-line point plus/minus local left normal multiplied by "
            "side_left/side_right"),
        "racing_line_has_speed_profile": "ai_speed_kmh" in line.columns,
        "source": source_meta.get("source", str(line_path)),
    }
    _write_kv(target / "TRACK_INFO.txt", info)
    return info


def _slug(value: object, fallback: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^0-9A-Za-z가-힣._-]+", "-", text)
    text = text.strip("._-")
    return text or fallback


def _compact_model_name(value: object, fallback: str) -> str:
    """Drop the Kunos `ks_` prefix and join words for folder names."""
    text = str(value or "").strip()
    text = re.sub(r"^ks[\s_-]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\s_-]+", "", text)
    text = re.sub(r"[^0-9A-Za-z가-힣.]+", "", text)
    return text or fallback


def _vehicle_name(meta: dict) -> str:
    """Return the ego vehicle model recorded in the race configuration."""
    cars = meta.get("race_cars")
    if isinstance(cars, list):
        ego = next((car for car in cars
                    if isinstance(car, dict) and int(car.get("car_id", -1)) == 0), None)
        if ego and ego.get("model"):
            return _compact_model_name(ego["model"], "unknowncar")
    for key in ("car_model", "vehicle_model", "car", "vehicle"):
        if meta.get(key):
            return _compact_model_name(meta[key], "unknowncar")
    return "unknowncar"


def _weather_folder_name(value: object) -> str:
    """Normalize Content Manager weather preset names for session folders."""
    text = str(value or "").strip().lower()
    text = re.sub(r"^\d+[\s_-]*", "", text)
    compact = re.sub(r"[^0-9a-z]+", "", text)
    if "mist" in compact or compact == "fog":
        return "mist"
    if "scatteredcloud" in compact or "lightcloud" in compact:
        return "scatteredclouds"
    if "clear" in compact:
        return "clear"
    return compact or "unknownweather"


def _ai_setting_mean(meta: dict, key: str) -> float | None:
    values = []
    for car in meta.get("race_cars") or []:
        if not isinstance(car, dict) or int(car.get("car_id", -1)) == 0:
            continue
        try:
            values.append(float(car[key]))
        except (KeyError, TypeError, ValueError):
            continue
    return float(np.mean(values)) if values else None


def _folder_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _session_name(run_id: str, track: str, vehicle: str,
                  mode: str, hz: int, meta: dict | None = None) -> str:
    parts = [
        _slug(run_id, "unknown_time"),
        _compact_model_name(track, "unknowntrack"),
        _compact_model_name(vehicle, "unknowncar"),
    ]
    if meta is not None:
        parts.append(_weather_folder_name(meta.get("weather_name")))
    parts.extend([mode, f"{hz}hz"])
    if meta is not None and mode == "multi":
        strength = _ai_setting_mean(meta, "ai_level")
        aggression = _ai_setting_mean(meta, "ai_aggression")
        if strength is not None:
            parts.append(_folder_number(strength))
        if aggression is not None:
            parts.append(_folder_number(aggression))
    return "_".join(parts)


def _opponent_tables(run_dir: Path) -> dict[int, pd.DataFrame]:
    tables: dict[int, pd.DataFrame] = {}
    for path in sorted((run_dir / "processed" / "opponent").glob("opp_*_features.parquet")):
        oid = int(path.stem.split("_")[1])
        table = pd.read_parquet(path)
        if len(table) and bool(table.get("valid", pd.Series(True, index=table.index)).any()):
            tables[oid] = table
    return tables


def _contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b)) for a, b in edges.reshape(-1, 2)]


def _median_dt(t: np.ndarray) -> float:
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.median(dt)) if len(dt) else 0.025


def _has_sustained_clearance(clearance: np.ndarray, t: np.ndarray, event_time: float,
                             margin_m: float = 0.5, hold_seconds: float = 0.75,
                             horizon_seconds: float = 6.0) -> tuple[bool, float | None]:
    window = (t >= event_time) & (t <= event_time + horizon_seconds)
    for a, b in _contiguous_regions(window & np.isfinite(clearance) & (clearance >= margin_m)):
        if b - a >= 2 and t[b - 1] - t[a] >= hold_seconds:
            return True, float(t[a])
    return False, None


def _bbox_clearances(run_dir: Path, opponents: dict[int, pd.DataFrame]
                     ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return full-body pass clearances from same-render visual AABBs.

    Tuple values are ``(ego rear - opponent front, opponent rear - ego front)``
    along the leading vehicle's horizontal heading. Positive means the trailing
    edge has completely cleared the other vehicle's leading edge.
    """
    path = run_dir / "camera" / "render_car_state.parquet"
    if not path.exists():
        return {}
    cars = pd.read_parquet(path)
    required = {"frame", "car_id", "car_pos_x", "car_pos_y", "car_forward_x",
                "car_forward_y", "aabb_center_forward", "aabb_center_right",
                "aabb_length", "aabb_width"}
    if not required <= set(cars.columns):
        return {}

    def geometry(table: pd.DataFrame, frames: np.ndarray):
        rows = table.drop_duplicates("frame").set_index("frame").reindex(frames)
        forward = rows[["car_forward_x", "car_forward_y"]].to_numpy(float)
        norm = np.linalg.norm(forward, axis=1, keepdims=True)
        forward = np.divide(forward, norm, out=np.full_like(forward, np.nan), where=norm > 1e-6)
        right = np.column_stack([forward[:, 1], -forward[:, 0]])
        pos = rows[["car_pos_x", "car_pos_y"]].to_numpy(float)
        center = (pos + rows["aabb_center_forward"].to_numpy(float)[:, None] * forward
                  + rows["aabb_center_right"].to_numpy(float)[:, None] * right)
        return center, forward, right, rows["aabb_length"].to_numpy(float), \
            rows["aabb_width"].to_numpy(float)

    ego_rows = cars[cars["car_id"] == 0]
    if ego_rows.empty:
        return {}
    clearances: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for oid, opponent in opponents.items():
        opp_rows = cars[cars["car_id"] == oid]
        if opp_rows.empty:
            continue
        frames = opponent["frame"].to_numpy(int)
        ec, ef, er, el, ew = geometry(ego_rows, frames)
        oc, of, oright, ol, ow = geometry(opp_rows, frames)
        aligned = np.einsum("ij,ij->i", ef, of) >= math.cos(math.radians(45.0))

        def extent(forward, right, length, width, tangent):
            return (np.abs(np.einsum("ij,ij->i", forward, tangent)) * length / 2.0
                    + np.abs(np.einsum("ij,ij->i", right, tangent)) * width / 2.0)

        ego_extent_on_ego = extent(ef, er, el, ew, ef)
        opp_extent_on_ego = extent(of, oright, ol, ow, ef)
        ego_center_on_ego = np.einsum("ij,ij->i", ec, ef)
        opp_center_on_ego = np.einsum("ij,ij->i", oc, ef)
        ego_ahead = ((ego_center_on_ego - ego_extent_on_ego)
                     - (opp_center_on_ego + opp_extent_on_ego))

        opp_extent_on_opp = extent(of, oright, ol, ow, of)
        ego_extent_on_opp = extent(ef, er, el, ew, of)
        opp_center_on_opp = np.einsum("ij,ij->i", oc, of)
        ego_center_on_opp = np.einsum("ij,ij->i", ec, of)
        opp_ahead = ((opp_center_on_opp - opp_extent_on_opp)
                     - (ego_center_on_opp + ego_extent_on_opp))
        ego_ahead[~aligned] = np.nan
        opp_ahead[~aligned] = np.nan
        clearances[oid] = ego_ahead, opp_ahead
    return clearances


def _track_order_deltas(run_dir: Path,
                        opponents: dict[int, pd.DataFrame]) -> dict[int, np.ndarray]:
    """Signed along-track gap in metres; positive means the opponent is ahead."""
    line_path = run_dir / "track" / "racing_line.parquet"
    ego_path = run_dir / "processed" / "ego_features.parquet"
    if not line_path.exists() or not ego_path.exists():
        return {}
    from .processed_builder import _nearest_indices

    line = pd.read_parquet(line_path)
    ego = pd.read_parquet(
        ego_path, columns=["world_position_x", "world_position_y"])
    required = {"x", "y", "dist"}
    if not required <= set(line.columns):
        return {}
    lx = line["x"].to_numpy(float)
    ly = line["y"].to_numpy(float)
    distance = line["dist"].to_numpy(float)
    closing_segment = (
        float(np.hypot(lx[0] - lx[-1], ly[0] - ly[-1]))
        if len(line) > 1 else 0.0)
    lap_length = float(np.nanmax(distance)) + closing_segment
    if not np.isfinite(lap_length) or lap_length <= 0:
        return {}
    ego_index = _nearest_indices(
        ego["world_position_x"].to_numpy(float),
        ego["world_position_y"].to_numpy(float), lx, ly)
    ego_distance = distance[ego_index]

    deltas: dict[int, np.ndarray] = {}
    for oid, table in opponents.items():
        if not {"opp_x", "opp_y"} <= set(table.columns) or len(table) != len(ego):
            continue
        opponent_index = _nearest_indices(
            table["opp_x"].to_numpy(float),
            table["opp_y"].to_numpy(float), lx, ly)
        raw = distance[opponent_index] - ego_distance
        deltas[oid] = (raw + lap_length / 2.0) % lap_length - lap_length / 2.0
    return deltas


def _contact_interactions(run_dir: Path,
                          opponents: dict[int, pd.DataFrame]) -> list[Interaction]:
    """Load exact ego collision-start events recorded by the CSP physics callback."""
    path = run_dir / "camera" / "collision_events.parquet"
    if not path.exists():
        return []
    contacts = pd.read_parquet(path)
    if not {"t_img", "collision_with"} <= set(contacts.columns):
        return []
    if "collision_event_seq" in contacts:
        contacts = contacts.drop_duplicates("collision_event_seq", keep="first")

    ego_velocity = None
    ego_path = run_dir / "processed" / "ego_features.parquet"
    if ego_path.exists():
        columns = ["t_img", "speed", "yaw"]
        try:
            ego_velocity = pd.read_parquet(ego_path, columns=columns)
        except (KeyError, ValueError):
            ego_velocity = None

    events = []
    last_contact_time: dict[tuple[str, int | None], float] = {}
    for _, row in contacts.sort_values("t_img").iterrows():
        event_time = float(row["t_img"])
        collided_with = int(row["collision_with"])
        # CSP `car.collidedWith` reserves 0 for no car and encodes a contacted
        # car as its zero-based AC car index plus one. Opponent tables and
        # render-car GT use the actual zero-based car index.
        opponent_id = collided_with - 1 if collided_with > 0 else None
        if opponent_id is not None:
            interaction_type = "car_contact"
        else:
            # onCarCollision also fires when the floor/wheels touch track or
            # curbs. Keep those events in raw GT, but do not label them as a
            # wall/barrier interaction.
            position_up = row.get("collision_position_up")
            if (position_up is None or pd.isna(position_up)
                    or float(position_up) < BARRIER_CONTACT_MIN_UP_M):
                continue
            interaction_type = "barrier_contact"
        contact_key = (interaction_type, opponent_id)
        previous_time = last_contact_time.get(contact_key)
        last_contact_time[contact_key] = event_time
        if (previous_time is not None
                and event_time - previous_time <= CONTACT_CLUSTER_GAP_SECONDS):
            continue
        minimum_distance = float("nan")
        contact_time_residual_ms = None
        contact_mapping_valid = None
        if opponent_id in opponents:
            table = opponents[opponent_id]
            nearest = int((table["t_img"] - event_time).abs().argmin())
            contact_time_residual_ms = (
                abs(float(table["t_img"].iloc[nearest]) - event_time) * 1000.0)
            candidate_distance = float(table["distance"].iloc[nearest])
            contact_mapping_valid = bool(
                contact_time_residual_ms <= CONTACT_TIME_TOLERANCE_MS
                and np.isfinite(candidate_distance)
                and candidate_distance <= CONTACT_MAX_CENTER_DISTANCE_M)
            if contact_mapping_valid:
                minimum_distance = candidate_distance
        elif opponent_id is not None:
            contact_mapping_valid = False

        def optional_float(key: str) -> float | None:
            value = row.get(key)
            return float(value) if value is not None and pd.notna(value) else None

        raw_relative_speed = optional_float("collision_relative_speed_mps")
        relative_speed = raw_relative_speed
        relative_speed_source = (
            "csp_raw_legacy_unverified" if raw_relative_speed is not None else None)
        if (contact_mapping_valid and ego_velocity is not None
                and opponent_id in opponents):
            table = opponents[opponent_id]
            velocity_columns = {"opp_vx", "opp_vy"}
            if velocity_columns <= set(table.columns):
                ego_nearest = int(
                    (ego_velocity["t_img"] - event_time).abs().argmin())
                opponent_nearest = int(
                    (table["t_img"] - event_time).abs().argmin())
                ego_speed = float(ego_velocity["speed"].iloc[ego_nearest])
                ego_yaw = float(ego_velocity["yaw"].iloc[ego_nearest])
                ego_v = np.array([
                    ego_speed * math.cos(ego_yaw),
                    ego_speed * math.sin(ego_yaw),
                ])
                opponent_v = table[
                    ["opp_vx", "opp_vy"]
                ].iloc[opponent_nearest].to_numpy(float)
                if np.isfinite(ego_v).all() and np.isfinite(opponent_v).all():
                    relative_speed = float(np.linalg.norm(ego_v - opponent_v))
                    relative_speed_source = (
                        "recomputed_from_mapped_ego_opponent_velocity_v1")

        events.append(Interaction(
            interaction_type=interaction_type,
            opponent_id=opponent_id,
            event_time=event_time,
            detected_start=event_time,
            detected_end=event_time,
            minimum_distance_m=minimum_distance,
            success=None,
            detection_method="csp_onCarCollision_id_plus_one_decoded_v3",
            collision_event_seq=(int(row["collision_event_seq"])
                                 if "collision_event_seq" in row else None),
            collision_with_raw=collided_with,
            collision_id_decode=(
                "car_id=collision_with-1" if collided_with > 0
                else "collision_with=0 means non-car contact"),
            contact_time_residual_ms=contact_time_residual_ms,
            contact_mapping_valid=contact_mapping_valid,
            collision_depth_m=optional_float("collision_depth_m"),
            collision_position_forward=optional_float("collision_position_forward"),
            collision_position_right=optional_float("collision_position_right"),
            collision_position_up=optional_float("collision_position_up"),
            collision_relative_speed_mps=relative_speed,
            collision_relative_speed_raw_mps=raw_relative_speed,
            collision_relative_speed_source=relative_speed_source,
            collision_damage_0=optional_float("collision_damage_0"),
            collision_damage_1=optional_float("collision_damage_1"),
            collision_damage_2=optional_float("collision_damage_2"),
            collision_damage_3=optional_float("collision_damage_3"),
            collision_damage_4=optional_float("collision_damage_4"),
        ))
    return events


def detect_interactions(opponents: dict[int, pd.DataFrame],
                        bbox_clearances: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
                        track_order_deltas: dict[int, np.ndarray] | None = None,
                        ) -> list[Interaction]:
    """Detect reproducible interaction candidates from ego-frame opponent GT.

    Labels are intentionally conservative and include their detection method.
    Collision is not inferred: the current telemetry has no reliable contact
    signal, and close distance alone is not a defensible collision label.
    """
    events: list[Interaction] = []
    for oid, table in opponents.items():
        if not {"t_img", "opp_forward", "opp_lateral", "distance"} <= set(table.columns):
            continue
        t = table["t_img"].to_numpy(float)
        fwd_raw = table["opp_forward"].to_numpy(float)
        track_order = (
            track_order_deltas.get(oid)
            if track_order_deltas and oid in track_order_deltas else None)
        order_raw = (
            np.asarray(track_order, dtype=float)
            if track_order is not None and len(track_order) == len(table)
            else fwd_raw)
        lat = table["opp_lateral"].to_numpy(float)
        distance = table["distance"].to_numpy(float)
        valid = table.get("valid", pd.Series(True, index=table.index)).to_numpy(bool)
        dt = _median_dt(t)
        smooth_n = max(3, int(round(0.25 / dt)))
        order = pd.Series(order_raw).rolling(
            smooth_n, center=True, min_periods=1).median().to_numpy()

        # An overtake is a sustained longitudinal ordering change while the
        # vehicles are close laterally and spatially.  Positive means opponent ahead.
        # Debounce repeated crossings in the same direction, but do not hide a
        # genuine counter-pass.  The sustained body-clearance check below
        # already rejects ordering jitter when the lead changes back quickly.
        last_cross_time = {
            "overtake": -np.inf,
            "overtaken_by": -np.inf,
        }
        for i in range(1, len(table)):
            direction = None
            if order[i - 1] >= 0 > order[i]:
                direction = "overtake"
            elif order[i - 1] <= 0 < order[i]:
                direction = "overtaken_by"
            if (direction is None or not valid[i]
                    or t[i] - last_cross_time[direction] < 4.0):
                continue
            before = (t >= t[i] - 1.0) & (t <= t[i] - 0.35) & valid
            after = (t >= t[i] + 0.35) & (t <= t[i] + 1.0) & valid
            near = (t >= t[i] - 1.0) & (t <= t[i] + 1.0) & valid
            if not before.any() or not after.any() or not near.any():
                continue
            b = float(np.nanmedian(order[before]))
            a = float(np.nanmedian(order[after]))
            sign_ordered = ((b > 0.0 and a < 0.0) if direction == "overtake"
                            else (b < 0.0 and a > 0.0))
            strong_ordered = ((b >= 2.0 and a <= -2.0) if direction == "overtake"
                              else (b <= -2.0 and a >= 2.0))
            if (not sign_ordered or abs(lat[i]) > 6.0
                    or float(np.nanmin(distance[near])) > 12.0):
                continue
            if track_order is not None:
                clearance = (
                    -order_raw - 4.5 if direction == "overtake"
                    else order_raw - 4.5)
                detection_method = "track_progress_clearance_v1"
            elif bbox_clearances and oid in bbox_clearances:
                clearance = bbox_clearances[oid][0 if direction == "overtake" else 1]
                detection_method = "same_render_bbox_clearance_v1"
            else:
                # Conservative fallback when same-render AABBs are unavailable.
                # The default 4.5 m cars must clear each other by another 0.5 m.
                clearance = -fwd_raw - 4.5 if direction == "overtake" else fwd_raw - 4.5
                detection_method = "length_clearance_fallback_v1"
            success, completion_time = _has_sustained_clearance(clearance, t, float(t[i]))
            # Full-body clearance is definitive even for a slow pass whose
            # centres move by less than 2 m in the fixed before/after windows.
            # Keep the stricter displacement requirement for failed attempts
            # so ordinary side-by-side ordering jitter is not mislabeled.
            if not success and not strong_ordered:
                continue
            label = direction if success else f"{direction}_attempt"
            events.append(Interaction(
                label, oid, completion_time if success else float(t[i]),
                float(t[before][0]), float(min(t[-1], t[i] + 3.0)),
                float(np.nanmin(distance[near])), success,
                detection_method,
            ))
            last_cross_time[direction] = float(t[i])

        # Side-by-side and close-follow are interval interactions.  They are
        # retained even when an overtake also occurs, because labels may overlap.
        side = valid & (np.abs(fwd_raw) <= 5.0) & (np.abs(lat) <= 5.0) & (distance <= 7.5)
        for a, b in _contiguous_regions(side):
            if b - a < 2 or t[b - 1] - t[a] < 1.0:
                continue
            segment = slice(a, b)
            k = a + int(np.nanargmin(distance[segment]))
            events.append(Interaction(
                "side_by_side", oid, float(t[k]), float(t[a]), float(t[b - 1]),
                float(np.nanmin(distance[segment])), None,
            ))

        follow = valid & (fwd_raw >= 3.0) & (fwd_raw <= 20.0) & (np.abs(lat) <= 3.5)
        for a, b in _contiguous_regions(follow):
            if b - a < 2 or t[b - 1] - t[a] < 2.0:
                continue
            segment = slice(a, b)
            k = a + int(np.nanargmin(distance[segment]))
            events.append(Interaction(
                "close_follow", oid, float(t[k]), float(t[a]), float(t[b - 1]),
                float(np.nanmin(distance[segment])), None,
            ))

    return sorted(events, key=lambda event: (event.event_time,
                                              event.opponent_id or -1,
                                              event.interaction_type))


STATE_INTERACTIONS = {"side_by_side", "close_follow"}
BARRIER_CONTACT_MIN_UP_M = 0.20
CONTACT_CLUSTER_GAP_SECONDS = 0.5
CONTACT_TIME_TOLERANCE_MS = 50.0
CONTACT_MAX_CENTER_DISTANCE_M = 8.0


def _interaction_group(event: Interaction) -> str:
    return "STATES" if event.interaction_type in STATE_INTERACTIONS else "EVENTS"


def _reset_clock_coherent(spline: float, camera_game_ms: float,
                          ego_game_ms: float, camera_lap: int, ego_lap: int,
                          standing_start_lap: int | None = None) -> bool:
    """Reject a stale pre-reset lap clock, except on a real standing-start lap.

    In Practice, AC can keep a large current-lap clock when a stationary car
    crosses the line to begin its first retained lap.  That is valid data, not
    the one-render-tick lap-boundary lag this guard was designed to reject.
    """
    low_progress_with_large_clock = (
        np.isfinite(spline) and spline <= 0.02
        and (camera_game_ms > 5000.0 or ego_game_ms > 5000.0))
    if not low_progress_with_large_clock:
        return True
    return bool(
        standing_start_lap is not None
        and camera_lap == standing_start_lap
        and ego_lap == standing_start_lap)


def _export_index(run_dir: Path, grid_hz: float, only_valid: bool = True,
                  standing_start: bool = False) -> pd.DataFrame:
    cam = pd.read_parquet(run_dir / "camera" / "camera_index.parquet")
    ego = pd.read_parquet(run_dir / "processed" / "ego_features.parquet")
    ego_raw = pd.read_parquet(run_dir / "ego_state" / "ego_state.parquet")
    slots, _, _, _ = _grid_slots(cam["t_img"].to_numpy(float), grid_hz)
    valid = ego.get("valid", pd.Series(True, index=ego.index)).to_numpy(bool)
    standing_start_lap = None
    if standing_start and len(ego_raw):
        lap_column = "completedLaps" if "completedLaps" in ego_raw \
            else "LapCount" if "LapCount" in ego_raw else None
        if lap_column is not None:
            standing_start_lap = int(round(float(ego_raw[lap_column].iloc[0])))
    rows = []
    for ms, i in slots:
        if only_valid and not valid[i]:
            continue
        spline = (float(ego["NormalizedSplinePosition"].iloc[i])
                  if "NormalizedSplinePosition" in ego else float("nan"))
        camera_game_ms = (float(cam["game_time_ms"].iloc[i])
                          if "game_time_ms" in cam else float("nan"))
        ego_game_ms = (float(ego_raw["currentTime"].iloc[i])
                       if "currentTime" in ego_raw else float("nan"))
        camera_lap = (int(round(float(cam["completed_laps"].iloc[i])))
                      if "completed_laps" in cam else 0)
        ego_lap = (int(round(float(ego_raw["completedLaps"].iloc[i])))
                   if "completedLaps" in ego_raw else
                   int(round(float(ego["LapCount"].iloc[i])))
                   if "LapCount" in ego else 0)
        residual = abs(camera_game_ms - ego_game_ms)
        # AC can reset spline/lap time one render tick before its lap counters.
        # A low spline position paired with the previous lap's large clock is
        # an incoherent boundary row and must not enter a retained lap.
        reset_coherent = _reset_clock_coherent(
            spline, camera_game_ms, ego_game_ms, camera_lap, ego_lap,
            standing_start_lap=standing_start_lap)
        sync_ok = bool(
            np.isfinite(residual) and residual <= 50.0
            and camera_lap == ego_lap and reset_coherent)
        rows.append({
            "time_ms": int(ms),
            "source_index": int(i),
            "source_frame": int(cam["frame"].iloc[i]),
            "t_img": float(cam["t_img"].iloc[i]),
            "lap_count": int(round(float(ego["LapCount"].iloc[i]))) if "LapCount" in ego else 0,
            "spline_position": spline,
            "camera_game_time_ms": camera_game_ms,
            "ego_game_time_ms": ego_game_ms,
            "game_time_residual_ms": residual,
            "sync_ok": sync_ok,
        })
    return pd.DataFrame(rows)


def _link_frame(flat: Path, target: Path, source_time_ms: int,
                destination_time_ms: int | None = None,
                include_objects: bool = True) -> None:
    destination_time_ms = (source_time_ms if destination_time_ms is None
                           else destination_time_ms)
    subdirs = ("CAMERA", "EGO_INFO", "OBJECT_INFO") if include_objects \
        else ("CAMERA", "EGO_INFO")
    for subdir in subdirs:
        source_dir = flat / subdir
        matches = list(source_dir.glob(f"{source_time_ms}.*")) if subdir == "CAMERA" \
            else [source_dir / f"{source_time_ms}.txt"]
        for source in matches:
            if source.exists():
                destination = target / subdir / f"{destination_time_ms}{source.suffix}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                _link_or_copy(source, destination)


def _link_selection(flat: Path, target: Path, index: pd.DataFrame,
                    include_objects: bool = True) -> None:
    subdirs = ("CAMERA", "EGO_INFO", "OBJECT_INFO") if include_objects \
        else ("CAMERA", "EGO_INFO")
    for subdir in subdirs:
        (target / subdir).mkdir(parents=True, exist_ok=True)
    # Do not glob a directory containing tens of thousands of frames once per
    # output row. Build the camera lookup once; the previous O(n²) scan made a
    # seven-lap re-export take many minutes.
    camera_by_time = {
        source.stem: source for source in (flat / "CAMERA").iterdir()
        if source.is_file()
    }
    source_times = (index["source_time_ms"] if "source_time_ms" in index
                    else index["time_ms"])
    for source_ms, destination_ms in zip(
            source_times.astype(int), index["time_ms"].astype(int)):
        source_key = str(int(source_ms))
        destination_key = str(int(destination_ms))
        camera_source = camera_by_time.get(source_key)
        if camera_source is not None:
            _link_or_copy(
                camera_source,
                target / "CAMERA" / f"{destination_key}{camera_source.suffix}")
        for subdir in ("EGO_INFO", "OBJECT_INFO"):
            if subdir == "OBJECT_INFO" and not include_objects:
                continue
            source = flat / subdir / f"{source_key}.txt"
            if source.exists():
                _link_or_copy(
                    source, target / subdir / f"{destination_key}.txt")


def _interaction_padding(event: Interaction, seconds_before: float,
                         seconds_after: float) -> tuple[float, float]:
    pass_labels = {
        "overtake", "overtaken_by", "overtake_attempt", "overtaken_by_attempt"}
    if event.interaction_type in pass_labels:
        return max(seconds_before, 10.0), max(seconds_after, 4.0)
    return seconds_before, seconds_after


def _interaction_partition(index: pd.DataFrame, interactions: list[Interaction],
                           seconds_before: float,
                           seconds_after: float
                           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the unique interaction-frame union and its exact complement."""
    selected = np.zeros(len(index), dtype=bool)
    t = index["t_img"].to_numpy(float)
    for event in interactions:
        before, after = _interaction_padding(
            event, seconds_before, seconds_after)
        selected |= (
            (t >= event.event_time - before)
            & (t <= event.event_time + after)
        )
    return index.loc[selected].copy(), index.loc[~selected].copy()


def _interaction_records(
        interactions: list[Interaction]
        ) -> list[tuple[Interaction, str]]:
    """Assign stable per-type IDs in the same order used for clip folders."""
    counters: dict[str, int] = {}
    records = []
    for event in interactions:
        counters[event.interaction_type] = counters.get(event.interaction_type, 0) + 1
        event_id = (
            f"{event.interaction_type}_{counters[event.interaction_type]:04d}")
        records.append((event, event_id))
    return records


def _interaction_union_index(
        index: pd.DataFrame,
        records: list[tuple[Interaction, str]],
        seconds_before: float,
        seconds_after: float,
        ) -> pd.DataFrame:
    """Build one row per unique interaction frame with multi-label provenance."""
    t = index["t_img"].to_numpy(float)
    labels = [set() for _ in range(len(index))]
    event_ids = [set() for _ in range(len(index))]
    for event, event_id in records:
        before, after = _interaction_padding(
            event, seconds_before, seconds_after)
        selected = np.flatnonzero(
            (t >= event.event_time - before)
            & (t <= event.event_time + after))
        for position in selected:
            labels[position].add(event.interaction_type)
            event_ids[position].add(event_id)
    selected = np.array([bool(value) for value in labels], dtype=bool)
    union = index.loc[selected].copy()
    union["interaction_types"] = [
        ",".join(sorted(labels[position]))
        for position in np.flatnonzero(selected)]
    union["interaction_ids"] = [
        ",".join(sorted(event_ids[position]))
        for position in np.flatnonzero(selected)]
    return union


def _clean_multi_index(
        run_dir: Path,
        index: pd.DataFrame,
        meta: dict,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trim desktop pre-roll and reject known-bad synchronization/pose rows.

    The last good frame before sustained movement is retained as the single
    stationary launch frame. Original grid time is kept in ``source_time_ms``;
    exported ``time_ms`` is normalized to start at zero.
    """
    work = index.copy().reset_index(drop=True)
    reasons = [set() for _ in range(len(work))]

    if "sync_ok" in work:
        for position in np.flatnonzero(~work["sync_ok"].astype(bool).to_numpy()):
            reasons[position].add("sync_false")

    raw_index_path = run_dir / "image_index.parquet"
    if raw_index_path.exists() and len(work):
        raw = pd.read_parquet(raw_index_path)
        pose_columns = [
            column for column in
            ("frame", "cam_pose_seq", "cam_pos_x", "cam_pose_game_delta_ms")
            if column in raw.columns]
        if "frame" in pose_columns:
            pose = raw[pose_columns].rename(columns={"frame": "source_frame"})
            pose = pose.drop_duplicates("source_frame", keep="last")
            audit = work[["source_frame"]].merge(
                pose, on="source_frame", how="left")
            pose_presence_columns = [
                column for column in ("cam_pose_seq", "cam_pos_x")
                if column in audit.columns]
            if pose_presence_columns:
                missing = audit[pose_presence_columns].isna().any(axis=1).to_numpy()
                for position in np.flatnonzero(missing):
                    reasons[position].add("camera_pose_missing")
            if "cam_pose_game_delta_ms" in audit:
                delta = pd.to_numeric(
                    audit["cam_pose_game_delta_ms"], errors="coerce")
                excessive = delta.abs().gt(50.0).fillna(False).to_numpy()
                for position in np.flatnonzero(excessive):
                    reasons[position].add("pose_delta_gt_50ms")

    motion_time = meta.get("standing_start_motion_time_s")
    if (meta.get("start_mode") == "standing"
            and motion_time is not None and len(work)):
        good = np.array([not value for value in reasons], dtype=bool)
        before_motion = np.flatnonzero(
            good & (work["t_img"].to_numpy(float) < float(motion_time)))
        if len(before_motion):
            retained_start = int(before_motion[-1])
        else:
            after_motion = np.flatnonzero(good)
            retained_start = int(after_motion[0]) if len(after_motion) else 0
        for position in range(retained_start):
            reasons[position].add("pre_motion_desktop")

    excluded_mask = np.array([bool(value) for value in reasons], dtype=bool)
    excluded = work.loc[excluded_mask].copy()
    excluded["reasons"] = [
        ",".join(sorted(reasons[position]))
        for position in np.flatnonzero(excluded_mask)]

    clean = work.loc[~excluded_mask].copy()
    if not clean.empty:
        clean.insert(1, "source_time_ms", clean["time_ms"].astype(int))
        origin = int(clean["source_time_ms"].iloc[0])
        clean["time_ms"] = clean["source_time_ms"].astype(int) - origin
    return clean.reset_index(drop=True), excluded.reset_index(drop=True)


def _non_interaction_segments(
        without_interaction: pd.DataFrame,
        grid_hz: float,
        ) -> list[pd.DataFrame]:
    """Split the complement into maximal continuous model-ready sequences."""
    if without_interaction.empty:
        return []
    ordered = without_interaction.sort_values("time_ms").reset_index(drop=True)
    step_ms = int(round(1000.0 / float(grid_hz)))
    breaks = np.r_[
        True,
        np.diff(ordered["time_ms"].to_numpy(int)) != step_ms,
    ]
    groups = np.cumsum(breaks)
    return [
        group.reset_index(drop=True)
        for _, group in ordered.groupby(groups, sort=True)
    ]


def _write_schema(path: Path) -> None:
    path.write_text(
        "schema_version=3\n"
        "SOLO directories=CAMERA EGO_INFO; OBJECT_INFO is omitted\n"
        "FRAME_INDEX time_ms=milliseconds since the first retained clean frame\n"
        "FRAME_INDEX source_time_ms=milliseconds since collection start\n"
        "LAP_INFO start_type=standing_start for a retained launch lap; otherwise flying\n"
        "LAP_INFO motion_start_time_ms=movement onset on the normalized session timeline\n"
        "EGO_INFO format=key: whitespace-separated-values, one field per line\n"
        "EGO_INFO fields=position orientation velocity angular_velocity acceleration throttle brake steer_angle\n"
        "OBJECT_INFO format=CSV header followed by one row per valid opponent\n"
        "OBJECT_INFO identity=track_id is stable only inside one session\n"
        "OBJECT_INFO scope=all valid opponents; visible is 1 only inside the captured camera FOV\n"
        "MULTI directories=INTERACTIONS and NON_INTERACTION/0001,0002,...\n"
        "MULTI start=last clean frame before sustained standing-start movement\n"
        "MULTI excluded=pre-motion desktop, sync failure, missing pose, or pose delta over 50ms\n"
        "INTERACTION_UNION_INDEX one unique row per frame in any interaction clip window\n"
        "INTERACTION_UNION_INDEX multi-label fields=interaction_types interaction_ids\n"
        "NON_INTERACTION segments=maximal 25ms-contiguous complement of interaction union\n"
        "PARTITION invariant=INTERACTION_UNION_INDEX plus NON_INTERACTION is disjoint and complete\n"
        "EXCLUDED_FRAMES reasons=pre_motion_desktop sync_false camera_pose_missing pose_delta_gt_50ms\n"
        "INTERACTIONS groups=EVENTS for point events; STATES for sustained intervals\n"
        "CONTACT source=CSP onCarCollision physics callback; never inferred from distance\n",
        encoding="utf-8",
    )


def _complete_laps(index: pd.DataFrame,
                   standing_start_time: float | None = None
                   ) -> list[tuple[int, pd.DataFrame]]:
    laps: list[tuple[int, pd.DataFrame]] = []
    if index.empty:
        return laps
    # Capture may begin anywhere in a lap. Even a small spline position does
    # not prove that recording began exactly at the finish line. Only retain
    # laps opened by a finish-line crossing observed during this recording.
    first_source_lap = int(
        pd.to_numeric(index.sort_values("t_img")["lap_count"], errors="coerce")
        .dropna().iloc[0])
    if standing_start_time is not None:
        first_group = index[index["lap_count"] == first_source_lap].sort_values("t_img")
        if "sync_ok" in first_group:
            first_group = first_group[first_group["sync_ok"].astype(bool)]
        before_motion = np.flatnonzero(
            first_group["t_img"].to_numpy(float) < float(standing_start_time))
        start = int(before_motion[-1]) if len(before_motion) else 0
        progress = pd.to_numeric(
            first_group["spline_position"], errors="coerce").to_numpy(float)
        finishes = np.flatnonzero(progress >= 0.85)
        finish_candidates = finishes[finishes >= start]
        finish = int(finish_candidates[-1]) if len(finish_candidates) else -1
        standing_lap = (first_group.iloc[start:finish + 1]
                        if finish >= start else first_group.iloc[0:0])
        if len(standing_lap) > 1:
            laps.append((first_source_lap, standing_lap))
    for source_lap, group in index.groupby("lap_count", sort=True):
        if int(source_lap) == first_source_lap:
            continue
        group = group.sort_values("t_img")
        if "sync_ok" in group:
            group = group[group["sync_ok"].astype(bool)]
        progress = pd.to_numeric(group["spline_position"], errors="coerce")
        starts = np.flatnonzero(progress.to_numpy() <= 0.15)
        finishes = np.flatnonzero(progress.to_numpy() >= 0.85)
        # AC can reset spline progress one render tick before LapCount changes.
        # Trim those boundary ticks instead of rejecting an otherwise complete lap.
        if len(starts) and len(finishes):
            start = int(starts[0])
            finish_candidates = finishes[finishes > start]
            finish = int(finish_candidates[-1]) if len(finish_candidates) else -1
        else:
            start, finish = 0, -1
        trimmed = group.iloc[start:finish + 1] if finish >= start else group.iloc[0:0]
        trimmed_progress = pd.to_numeric(
            trimmed["spline_position"], errors="coerce").dropna()
        complete = (len(trimmed_progress) > 1
                    and float(trimmed_progress.iloc[0]) <= 0.15
                    and float(trimmed_progress.iloc[-1]) >= 0.85
                    and float(trimmed_progress.max() - trimmed_progress.min()) >= 0.8)
        if complete:
            laps.append((int(source_lap), trimmed))
    return laps


def _prepare_session_root(output_root: Path, mode: str, session_name: str) -> Path:
    root = (output_root / mode / session_name).resolve()
    mode_root = (output_root / mode).resolve()
    if mode_root not in root.parents:
        raise ValueError(f"unsafe session output path: {root}")
    marker = root / "SESSION_INFO.txt"
    if root.exists():
        if not marker.exists() or f"managed_by={MANAGED_BY}" not in marker.read_text(encoding="utf-8"):
            raise FileExistsError(f"refusing to replace unmanaged directory: {root}")
        def remove_readonly(func, path, _excinfo):
            os.chmod(path, stat.S_IWRITE)
            func(path)

        shutil.rmtree(root, onerror=remove_readonly)
    root.mkdir(parents=True)
    return root


def export_structured(run_dir: str | Path, output_root: str | Path | None = None,
                      mode: str = "auto", grid_hz: float | None = None,
                      seconds_before: float = 3.0, seconds_after: float = 3.0) -> Path:
    run_dir = Path(run_dir).resolve()
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    opponents = _opponent_tables(run_dir)
    if mode == "auto":
        configured = str(meta.get("session_mode", "auto")).lower()
        mode = configured if configured in {"solo", "multi"} else ("multi" if opponents else "solo")
    if mode not in {"solo", "multi"}:
        raise ValueError("mode must be auto, solo, or multi")
    if mode == "solo" and opponents:
        raise ValueError("solo mode requested, but valid opponent data exists")

    track = _track_name(run_dir)
    if grid_hz is None:
        grid_hz = float(meta.get("image_hz", meta.get("image_hz_actual", 40.0)))
    hz = round(grid_hz)
    vehicle = _vehicle_name(meta)
    session_name = _session_name(run_dir.name, track, vehicle, mode, hz, meta)
    output_root = Path(output_root).resolve() if output_root else run_dir.parent.parent / "dataset_v2"
    output_root.mkdir(parents=True, exist_ok=True)
    session = _prepare_session_root(output_root, mode, session_name)

    session_info = {
        "managed_by": MANAGED_BY,
        "schema_version": SCHEMA_VERSION,
        "run_id": run_dir.name,
        "mode": mode,
        "track": track,
        "vehicle": vehicle,
        "grid_hz": float(grid_hz),
        "source_run": str(run_dir),
        "opponent_count": len(opponents),
        "opponent_ids": sorted(opponents),
        "target_laps": meta.get("target_laps"),
        "race_laps": meta.get("race_laps"),
        "race_car_count": meta.get("race_car_count"),
        "race_ai_level": meta.get("race_ai_level"),
        "race_cars": meta.get("race_cars"),
        "weather_name": meta.get("weather_name"),
        "ambient_temperature_c": meta.get("ambient_temperature_c"),
        "road_temperature_c": meta.get("road_temperature_c"),
        "wind_speed_kmh_min": meta.get("wind_speed_kmh_min"),
        "wind_speed_kmh_max": meta.get("wind_speed_kmh_max"),
        "wind_direction_deg": meta.get("wind_direction_deg"),
        "sun_angle_deg": meta.get("sun_angle_deg"),
        "time_multiplier": meta.get("time_multiplier"),
        "weather_controller": meta.get("weather_controller"),
        "weather_type": meta.get("weather_type"),
        "dynamic_track": meta.get("dynamic_track"),
        "groove": meta.get("groove"),
        "ai_hints_source": meta.get("ai_hints_source"),
        "ai_hints": meta.get("ai_hints"),
        "capture_backend": meta.get("capture_backend"),
        "image_hz_target": meta.get("image_hz"),
        "image_hz_actual": meta.get("image_hz_actual"),
        "telemetry_hz_target": meta.get("telemetry_hz"),
        "telemetry_hz_actual": meta.get("telemetry_hz_actual"),
        "camera_pose_coverage": meta.get("camera_pose_coverage"),
        "collision_gt_available": int(meta.get("csp_bridge_version") or 0) >= 3,
        "interaction_seconds_before": seconds_before if mode == "multi" else None,
        "interaction_seconds_after": seconds_after if mode == "multi" else None,
    }
    track_info = _export_track_info(run_dir, session)
    session_info["track_info_available"] = bool(track_info.get("available"))
    _write_kv(session / "SESSION_INFO.txt", session_info)
    _write_schema(session / "SCHEMA.txt")

    index = _export_index(
        run_dir, float(grid_hz),
        standing_start=(meta.get("start_mode") == "standing"))

    with tempfile.TemporaryDirectory(prefix="ac_dataset_v2_", dir=output_root) as tmp:
        flat = Path(tmp) / "flat"
        export_run(run_dir, grid_hz=float(grid_hz), out_dir=flat,
                   include_all_objects=(mode == "multi"))
        flat_info_path = flat / "export_info.json"
        flat_info = {}
        if flat_info_path.exists():
            flat_info = json.loads(flat_info_path.read_text(encoding="utf-8"))
            session_info.update({
                "frames": flat_info.get("frames"),
                "hfov_deg": flat_info.get("hfov_deg"),
                "fov_source": flat_info.get("fov_source"),
                "object_scope": flat_info.get("object_scope"),
                "grid_gaps": flat_info.get("grid_gaps"),
                "grid_duplicates": flat_info.get("grid_duplicates"),
            })

        if mode == "solo":
            standing_start_time = (
                float(meta["standing_start_motion_time_s"])
                if meta.get("start_mode") == "standing"
                and meta.get("standing_start_motion_time_s") is not None
                else None)
            laps = _complete_laps(index, standing_start_time=standing_start_time)
            if not laps:
                raise RuntimeError("no complete solo laps found")
            time_origin_ms = int(laps[0][1]["time_ms"].iloc[0])
            normalized_laps: list[tuple[int, pd.DataFrame]] = []
            session_rows = []
            for number, (source_lap, frames) in enumerate(laps, start=1):
                frames = frames.copy()
                frames.insert(1, "source_time_ms", frames["time_ms"].astype(int))
                frames["time_ms"] = frames["source_time_ms"] - time_origin_ms
                frames.insert(1, "dataset_lap_number", number)
                normalized_laps.append((source_lap, frames))
                session_rows.append(frames)
            index = pd.concat(session_rows, ignore_index=True)
            index.to_csv(session / "FRAME_INDEX.csv", index=False, encoding="utf-8-sig")
            for number, (source_lap, frames) in enumerate(normalized_laps, start=1):
                target = session / f"lap_{number:02d}"
                _link_selection(flat, target, frames, include_objects=False)
                invalid = 0
                _write_kv(target / "LAP_INFO.txt", {
                    "schema_version": SCHEMA_VERSION,
                    "lap_number": number,
                    "source_lap_count": source_lap,
                    "start_type": (
                        "standing_start"
                        if number == 1 and standing_start_time is not None
                        else "flying"),
                    "motion_start_time_ms": (
                        int(round(standing_start_time * 1000.0)) - time_origin_ms
                        if number == 1 and standing_start_time is not None
                        else None),
                    "frames": len(frames),
                    "start_time_ms": int(frames["time_ms"].iloc[0]),
                    "end_time_ms": int(frames["time_ms"].iloc[-1]),
                    "duration_seconds": float(frames["t_img"].iloc[-1] - frames["t_img"].iloc[0]),
                    "invalid": invalid,
                })
            session_info["source_frames"] = flat_info.get("frames")
            session_info["frames"] = len(index)
            session_info["time_origin_source_ms"] = time_origin_ms
            session_info["complete_laps"] = len(normalized_laps)
        else:
            index, excluded = _clean_multi_index(run_dir, index, meta)
            if index.empty:
                raise RuntimeError("no clean multi frames remain after filtering")
            index.to_csv(session / "FRAME_INDEX.csv", index=False, encoding="utf-8-sig")
            excluded.to_csv(
                session / "EXCLUDED_FRAMES.csv", index=False, encoding="utf-8-sig")
            interactions = detect_interactions(
                opponents, _bbox_clearances(run_dir, opponents),
                _track_order_deltas(run_dir, opponents))
            interactions.extend(_contact_interactions(run_dir, opponents))
            interactions.sort(key=lambda event: (
                event.event_time, event.opponent_id or -1, event.interaction_type))
            records = _interaction_records(interactions)
            interaction_union = _interaction_union_index(
                index, records, seconds_before, seconds_after)
            interaction_union.to_csv(
                session / "INTERACTION_UNION_INDEX.csv",
                index=False, encoding="utf-8-sig")
            interaction_keys = set(
                interaction_union["source_frame"].astype(int).tolist())
            without_interaction = index.loc[
                ~index["source_frame"].astype(int).isin(interaction_keys)
            ].copy()
            segments = _non_interaction_segments(
                without_interaction, float(grid_hz))
            for number, segment in enumerate(segments, start=1):
                target = session / "NON_INTERACTION" / f"{number:04d}"
                _link_selection(flat, target, segment)
                segment.to_csv(
                    target / "FRAME_INDEX.csv",
                    index=False, encoding="utf-8-sig")
                _write_kv(target / "SEGMENT_INFO.txt", {
                    "schema_version": SCHEMA_VERSION,
                    "segment_number": number,
                    "frames": len(segment),
                    "start_time_ms": int(segment["time_ms"].iloc[0]),
                    "end_time_ms": int(segment["time_ms"].iloc[-1]),
                    "duration_seconds": (
                        float(segment["t_img"].iloc[-1]
                              - segment["t_img"].iloc[0])
                        if len(segment) > 1 else 0.0),
                })
            partition_info = {
                "schema_version": SCHEMA_VERSION,
                "partition_basis": (
                    "unique union of every exported interaction clip window"),
                "interaction_frames": len(interaction_union),
                "non_interaction_frames": len(without_interaction),
                "non_interaction_segments": len(segments),
                "full_frames": len(index),
                "disjoint": True,
                "complete": (
                    len(interaction_union) + len(without_interaction) == len(index)),
            }
            _write_kv(session / "PARTITION_INFO.txt", partition_info)
            counters: dict[str, int] = {}
            group_counters = {"EVENTS": 0, "STATES": 0}
            group_types: dict[str, dict[str, int]] = {"EVENTS": {}, "STATES": {}}
            for event, event_id in records:
                counters[event.interaction_type] = counters.get(event.interaction_type, 0) + 1
                group = _interaction_group(event)
                group_counters[group] += 1
                group_types[group][event.interaction_type] = \
                    group_types[group].get(event.interaction_type, 0) + 1
                suffix = (f"_opp{event.opponent_id:02d}"
                          if event.opponent_id is not None else "")
                target = (session / "INTERACTIONS" / group / event.interaction_type
                          / f"{event_id}{suffix}")
                clip_before, clip_after = _interaction_padding(
                    event, seconds_before, seconds_after)
                clip_start = max(
                    float(index["t_img"].min()), event.event_time - clip_before)
                clip_end = min(
                    float(index["t_img"].max()), event.event_time + clip_after)
                clip = index[(index["t_img"] >= clip_start) & (index["t_img"] <= clip_end)]
                _link_selection(flat, target, clip)
                event_frame = index.iloc[(index["t_img"] - event.event_time).abs().argmin()]
                info = asdict(event)
                info.update({
                    "schema_version": SCHEMA_VERSION,
                    "interaction_id": event_id,
                    "interaction_group": group.lower(),
                    "primary_opponent_id": event.opponent_id,
                    "participant_ids": ([event.opponent_id]
                                        if event.opponent_id is not None else []),
                    "lap": int(event_frame["lap_count"]),
                    "event_time_ms": int(event_frame["time_ms"]),
                    "clip_start_time_ms": int(clip["time_ms"].iloc[0]),
                    "clip_end_time_ms": int(clip["time_ms"].iloc[-1]),
                    "seconds_before": clip_before,
                    "seconds_after": clip_after,
                    "frames": len(clip),
                })
                _write_kv(target / "INTERACTION_INFO.txt", info)
            session_info["interaction_count"] = len(interactions)
            session_info["interaction_types"] = counters
            session_info["event_count"] = group_counters["EVENTS"]
            session_info["state_count"] = group_counters["STATES"]
            session_info["event_types"] = group_types["EVENTS"]
            session_info["state_types"] = group_types["STATES"]
            excluded_counts: dict[str, int] = {}
            for reason_list in excluded.get("reasons", pd.Series(dtype=str)):
                for reason in str(reason_list).split(","):
                    if reason:
                        excluded_counts[reason] = excluded_counts.get(reason, 0) + 1
            session_info["source_frames"] = flat_info.get("frames")
            session_info["frames"] = len(index)
            session_info["excluded_frames"] = len(excluded)
            session_info["excluded_reasons"] = excluded_counts
            session_info["time_origin_source_ms"] = int(
                index["source_time_ms"].iloc[0])
            motion_time = meta.get("standing_start_motion_time_s")
            session_info["motion_start_time_ms"] = (
                int(round(float(motion_time) * 1000.0))
                - int(index["source_time_ms"].iloc[0])
                if motion_time is not None else None)
            session_info["interaction_union_frames"] = len(interaction_union)
            session_info["non_interaction_frames"] = len(without_interaction)
            session_info["non_interaction_segments"] = len(segments)
            session_info["partition_complete"] = (
                len(interaction_union) + len(without_interaction) == len(index))
            session_info["full_materialized"] = False

    _write_kv(session / "SESSION_INFO.txt", session_info)
    print(f"[structured-export] {mode}: {session}")
    return session


def main() -> None:
    parser = argparse.ArgumentParser(description="run -> dataset_v2 solo laps / multi interactions")
    parser.add_argument("run_dir", help="dataset/<run_id>")
    parser.add_argument("--out", default=None, help="output root (default: project/dataset_v2)")
    parser.add_argument("--mode", choices=["auto", "solo", "multi"], default="auto")
    parser.add_argument("--grid-hz", type=float, default=None)
    parser.add_argument("--interaction-before", type=float, default=3.0)
    parser.add_argument("--interaction-after", type=float, default=3.0)
    args = parser.parse_args()
    export_structured(args.run_dir, output_root=args.out, mode=args.mode,
                      grid_hz=args.grid_hz, seconds_before=args.interaction_before,
                      seconds_after=args.interaction_after)


if __name__ == "__main__":
    main()
