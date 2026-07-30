"""Lean distribution audit for the selected Laguna Seca training sessions.

The script is read-only with respect to dataset_v2/dataset.  It writes compact
CSV summaries and PNG plots to analysis/lagunaseca_distribution.
"""

from __future__ import annotations

import ast
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "dataset_v2"
RAW = ROOT / "dataset"
OUT = ROOT / "analysis" / "lagunaseca_distribution"

SOLO_NAMES = [
    "20260723_153625_lagunaseca_ferrarisf15t_solo_40hz",
    "20260723_162740_lagunaseca_ferrarisf15t_solo_40hz",
    "20260723_172733_lagunaseca_ferrarisf15t_solo_40hz",
    "20260724_141917_lagunaseca_ferrarisf15t_solo_40hz",
    "20260724_145227_lagunaseca_ferrarisf15t_solo_40hz",
    "20260724_151110_lagunaseca_ferrarisf15t_solo_40hz",
    "20260724_154559_lagunaseca_ferrarisf15t_clear_solo_40hz",
    "20260724_163545_lagunaseca_ferrarisf15t_scatteredclouds_solo_40hz",
    "20260724_172544_lagunaseca_ferrarisf15t_solo_40hz",
    "20260727_153344_lagunaseca_ferrarisf15t_clear_solo_40hz_stollrevised",
    "20260727_154916_lagunaseca_ferrarisf15t_mist_solo_40hz",
]

MULTI_NAMES = [
    "20260728_160119_lagunaseca_ferrarisf15t_clear_multi_40hz_80_85",
    "20260728_170457_lagunaseca_ferrarisf15t_clear_multi_40hz_80_85",
    "20260728_172011_lagunaseca_ferrarisf15t_clear_multi_40hz_80_90",
    "20260728_175057_lagunaseca_ferrarisf15t_clear_multi_40hz_80_80",
    "20260728_192423_lagunaseca_ferrarisf15t_clear_multi_40hz_80_80",
    "20260729_152224_lagunaseca_ferrarisf15t_scatteredclouds_multi_40hz_80_80",
    "20260729_160514_lagunaseca_ferrarisf15t_scatteredclouds_multi_40hz_80_85",
    "20260729_162015_lagunaseca_ferrarisf15t_scatteredclouds_multi_40hz_80_90_almostperfect",
    "20260729_163521_lagunaseca_ferrarisf15t_mist_multi_40hz_80_80",
    "20260729_170906_lagunaseca_ferrarisf15t_mist_multi_40hz_80_85",
    "20260729_172927_lagunaseca_ferrarisf15t_mist_multi_40hz_80_90",
    "20260729_174933_lagunaseca_ferrarisf15t_mist_multi_40hz_80_85",
]


def parse_info(path: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if key in {
            "run_id",
            "mode",
            "track",
            "vehicle",
            "weather_name",
            "interaction_id",
            "interaction_type",
            "interaction_group",
        }:
            out[key] = value
            continue
        if not value:
            out[key] = ""
            continue
        try:
            out[key] = json.loads(value)
        except Exception:
            try:
                out[key] = ast.literal_eval(value)
            except Exception:
                out[key] = value
    return out


def weather_group(name: str, info: dict[str, object]) -> str:
    text = f"{name} {info.get('weather_name', '')}".lower()
    if "mist" in text or "fog" in text:
        return "mist"
    if "cloud" in text:
        return "scatteredclouds"
    return "clear"


def parse_ego_text(path: Path) -> tuple[float, ...]:
    vals: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        vals[key] = [float(x) for x in value.split()]
    position = vals["position"]
    velocity = vals["velocity"]
    return (
        position[0],
        position[1],
        position[2],
        math.sqrt(sum(v * v for v in velocity)),
        vals["steer_angle"][0],
        vals["throttle"][0],
        vals["brake"][0],
    )


def solo_lap_numbers(session: Path, index: pd.DataFrame) -> np.ndarray:
    if "dataset_lap_number" in index:
        return index["dataset_lap_number"].to_numpy(dtype=int)
    lap_numbers = np.zeros(len(index), dtype=int)
    times = index["time_ms"].to_numpy(dtype=int)
    for lap_dir in sorted(session.glob("lap_*")):
        if not lap_dir.is_dir():
            continue
        info = parse_info(lap_dir / "LAP_INFO.txt")
        number = int(info.get("lap_number", lap_dir.name.split("_")[-1]))
        start = int(info["start_time_ms"])
        end = int(info["end_time_ms"])
        lap_numbers[(times >= start) & (times <= end)] = number
    return lap_numbers


def structured_solo_ego(
    session: Path, index: pd.DataFrame, lap_numbers: np.ndarray
) -> pd.DataFrame:
    paths = [
        session / f"lap_{int(lap):02d}" / "EGO_INFO" / f"{int(time_ms)}.txt"
        for lap, time_ms in zip(lap_numbers, index["time_ms"])
    ]
    with ThreadPoolExecutor(max_workers=16) as pool:
        rows = list(pool.map(parse_ego_text, paths, chunksize=128))
    return pd.DataFrame(
        rows,
        columns=["x", "y", "z", "speed_ms", "steer", "throttle", "brake"],
    )


def raw_ego(run_id: str, index: pd.DataFrame) -> pd.DataFrame | None:
    run = RAW / run_id
    processed_path = run / "processed" / "ego_features.parquet"
    state_path = run / "ego_state" / "ego_state.parquet"
    if not processed_path.exists() or not state_path.exists():
        return None
    ego = pd.read_parquet(
        processed_path,
        columns=[
            "speed",
            "world_position_x",
            "world_position_y",
            "world_position_z",
        ],
    )
    state = pd.read_parquet(
        state_path, columns=["steerNorm", "throttle", "brake"]
    )
    take = index["source_index"].to_numpy(dtype=int)
    return pd.DataFrame(
        {
            "x": ego["world_position_x"].to_numpy()[take],
            "y": ego["world_position_y"].to_numpy()[take],
            "z": ego["world_position_z"].to_numpy()[take],
            "speed_ms": ego["speed"].to_numpy()[take],
            "steer": state["steerNorm"].to_numpy()[take],
            "throttle": state["throttle"].to_numpy()[take],
            "brake": state["brake"].to_numpy()[take],
        }
    )


def reference_geometry(session: Path) -> pd.DataFrame:
    path = session / "TRACK_INFO" / "AI_RACING_LINE.parquet"
    if not path.exists():
        path = (
            V2
            / "solo"
            / "20260724_154559_lagunaseca_ferrarisf15t_clear_solo_40hz"
            / "TRACK_INFO"
            / "AI_RACING_LINE.parquet"
        )
    return pd.read_parquet(path)


def add_lateral_offset(data: pd.DataFrame, ref: pd.DataFrame) -> None:
    progress = np.mod(data["spline"].to_numpy(float), 1.0)
    rp = ref["normalized_position"].to_numpy(float)
    rx = ref["x"].to_numpy(float)
    ry = ref["y"].to_numpy(float)
    # Include a wrapped endpoint so interpolation near the finish line is stable.
    rp2 = np.r_[rp, 1.0]
    rx2 = np.r_[rx, rx[0]]
    ry2 = np.r_[ry, ry[0]]
    ref_x = np.interp(progress, rp2, rx2)
    ref_y = np.interp(progress, rp2, ry2)
    eps = 1.0 / max(len(ref), 1000)
    p0 = np.mod(progress - eps, 1.0)
    p1 = np.mod(progress + eps, 1.0)
    x0, x1 = np.interp(p0, rp2, rx2), np.interp(p1, rp2, rx2)
    y0, y1 = np.interp(p0, rp2, ry2), np.interp(p1, rp2, ry2)
    tx, ty = x1 - x0, y1 - y0
    norm = np.maximum(np.hypot(tx, ty), 1e-6)
    nx, ny = -ty / norm, tx / norm
    data["lateral_offset_m"] = (
        (data["x"].to_numpy() - ref_x) * nx
        + (data["y"].to_numpy() - ref_y) * ny
    )


def load_session(mode: str, name: str) -> tuple[pd.DataFrame, dict[str, object]]:
    session = V2 / mode / name
    info = parse_info(session / "SESSION_INFO.txt")
    index = pd.read_csv(session / "FRAME_INDEX.csv")
    run_id = str(info["run_id"])
    if mode == "solo":
        lap_numbers = solo_lap_numbers(session, index)
        # Early schema-v2 exports kept pre-first-complete-lap rows in the root
        # index even though the training material lives only in lap_* folders.
        retained = lap_numbers > 0
        index = index.loc[retained].reset_index(drop=True)
        lap_numbers = lap_numbers[retained]
    else:
        lap_numbers = index["lap_count"].to_numpy(dtype=int) + 1
    ego = raw_ego(run_id, index)
    source = "raw_parquet"
    if ego is None:
        if mode != "solo":
            raise RuntimeError(f"No raw parquet for multi session {name}")
        ego = structured_solo_ego(session, index, lap_numbers)
        source = "structured_text"

    data = pd.DataFrame(
        {
            "time_ms": index["time_ms"].astype(int),
            "source_index": index["source_index"].astype(int),
            "spline": index["spline_position"].astype(float),
            "lap": lap_numbers,
        }
    )
    for col in ego:
        data[col] = ego[col].to_numpy()
    data["speed_kmh"] = data["speed_ms"] * 3.6
    data["abs_steer"] = data["steer"].abs()
    data["mode"] = mode
    data["session"] = name
    data["run_id"] = run_id
    data["weather"] = weather_group(name, info)
    data["loader_source"] = source
    add_lateral_offset(data, reference_geometry(session))
    return data, info


def interaction_rows(session: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    root = session / "INTERACTIONS"
    for info_path in root.rglob("INTERACTION_INFO.txt"):
        info = parse_info(info_path)
        start = float(info.get("detected_start", np.nan))
        end = float(info.get("detected_end", np.nan))
        rows.append(
            {
                "session": session.name,
                "interaction_id": info.get("interaction_id", info_path.parent.name),
                "type": info.get("interaction_type", ""),
                "opponent_id": info.get("opponent_id", info.get("primary_opponent_id", "")),
                "duration_s": max(0.0, end - start) if np.isfinite(start + end) else 0.0,
                "minimum_distance_m": info.get("minimum_distance_m", np.nan),
                "success": info.get("success", ""),
            }
        )
    return rows


def opponent_histogram(name: str, info: dict[str, object]) -> tuple[np.ndarray, dict[str, object]]:
    run_id = str(info["run_id"])
    session = V2 / "multi" / name
    index = pd.read_csv(session / "FRAME_INDEX.csv")
    take = index["source_index"].to_numpy(dtype=int)
    xedges = np.linspace(-60, 80, 71)
    yedges = np.linspace(-15, 15, 61)
    hist = np.zeros((len(xedges) - 1, len(yedges) - 1), dtype=np.int64)
    total = close = front = rear = 0
    valid_opponents = []
    for path in sorted((RAW / run_id / "processed" / "opponent").glob("opp_*_features.parquet")):
        opp = pd.read_parquet(
            path, columns=["valid", "retired", "opp_forward", "opp_lateral", "distance"]
        ).iloc[take]
        mask = opp["valid"].fillna(False).to_numpy(bool)
        mask = mask & ~opp["retired"].fillna(False).to_numpy(bool)
        fwd = opp["opp_forward"].to_numpy(float)[mask]
        lat = opp["opp_lateral"].to_numpy(float)[mask]
        dist = opp["distance"].to_numpy(float)[mask]
        finite = np.isfinite(fwd) & np.isfinite(lat) & np.isfinite(dist)
        fwd, lat, dist = fwd[finite], lat[finite], dist[finite]
        hist += np.histogram2d(fwd, lat, bins=(xedges, yedges))[0].astype(np.int64)
        total += len(fwd)
        close += int(np.sum(dist <= 20.0))
        front += int(np.sum((fwd > 0) & (np.abs(lat) <= 5.0)))
        rear += int(np.sum((fwd < 0) & (np.abs(lat) <= 5.0)))
        valid_opponents.append(path.stem)
    stats = {
        "session": name,
        "valid_opponent_rows": total,
        "within_20m_rows": close,
        "within_20m_fraction": close / total if total else 0.0,
        "front_corridor_fraction": front / total if total else 0.0,
        "rear_corridor_fraction": rear / total if total else 0.0,
        "opponent_files": len(valid_opponents),
    }
    return hist, stats


def session_summary(data: pd.DataFrame, info: dict[str, object]) -> dict[str, object]:
    duration_s = (data["time_ms"].max() - data["time_ms"].min()) / 1000.0
    laps = (
        data["lap"].nunique()
        if data["mode"].iat[0] == "solo"
        else int(info.get("target_laps", data["lap"].nunique()))
    )
    return {
        "session": data["session"].iat[0],
        "run_id": data["run_id"].iat[0],
        "mode": data["mode"].iat[0],
        "weather": data["weather"].iat[0],
        "frames": len(data),
        "duration_s": duration_s,
        "laps": laps,
        "mean_speed_kmh": data["speed_kmh"].mean(),
        "p05_speed_kmh": data["speed_kmh"].quantile(0.05),
        "p95_speed_kmh": data["speed_kmh"].quantile(0.95),
        "brake_active_fraction": (data["brake"] > 0.05).mean(),
        "high_brake_fraction": (data["brake"] > 0.5).mean(),
        "high_steer_fraction": (data["abs_steer"] > 0.5).mean(),
        "mean_abs_lateral_offset_m": data["lateral_offset_m"].abs().mean(),
        "p95_abs_lateral_offset_m": data["lateral_offset_m"].abs().quantile(0.95),
        "grid_gaps": int(info.get("grid_gaps", 0)),
        "interaction_union_frames": int(info.get("interaction_union_frames", 0)),
        "interaction_fraction": (
            float(info.get("interaction_union_frames", 0)) / len(data)
            if len(data)
            else 0.0
        ),
        "loader_source": data["loader_source"].iat[0],
    }


def lap_summaries(data: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for lap, part in data.groupby("lap", sort=True):
        duration_s = (part["time_ms"].max() - part["time_ms"].min()) / 1000.0
        if duration_s < 30.0:
            continue
        rows.append(
            {
                "session": data["session"].iat[0],
                "mode": data["mode"].iat[0],
                "weather": data["weather"].iat[0],
                "lap": int(lap),
                "frames": len(part),
                "duration_s": duration_s,
                "mean_speed_kmh": part["speed_kmh"].mean(),
                "mean_abs_lateral_offset_m": part["lateral_offset_m"].abs().mean(),
                "p95_abs_lateral_offset_m": part["lateral_offset_m"].abs().quantile(0.95),
                "p99_abs_lateral_offset_m": part["lateral_offset_m"].abs().quantile(0.99),
            }
        )
    return rows


def lap_frame_profile(data: pd.DataFrame, bins: int = 200) -> pd.DataFrame:
    """Keep one real frame nearest each progress-bin center for every complete lap."""
    rows = []
    for lap, part in data.groupby("lap", sort=True):
        duration_s = (part["time_ms"].max() - part["time_ms"].min()) / 1000.0
        if duration_s < 30.0:
            continue
        work = part[
            ["session", "mode", "lap", "time_ms", "spline", "x", "y", "speed_kmh",
             "lateral_offset_m"]
        ].copy()
        work["track_bin"] = np.minimum((work["spline"] * bins).astype(int), bins - 1)
        centers = (work["track_bin"] + 0.5) / bins
        work["center_distance"] = (work["spline"] - centers).abs()
        take = work.groupby("track_bin", observed=True)["center_distance"].idxmin()
        sampled = work.loc[take].sort_values("track_bin").copy()
        sampled["abs_lateral_offset_m"] = sampled["lateral_offset_m"].abs()
        rows.append(
            sampled[
                ["session", "mode", "lap", "track_bin", "time_ms", "spline", "x", "y",
                 "speed_kmh", "abs_lateral_offset_m"]
            ]
        )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def binned_profile(data: pd.DataFrame, bins: int = 100) -> pd.DataFrame:
    work = data.copy()
    work["track_bin"] = np.minimum((work["spline"] * bins).astype(int), bins - 1)
    return (
        work.groupby(["mode", "session", "track_bin"], observed=True)
        .agg(
            frames=("speed_kmh", "size"),
            x=("x", "median"),
            y=("y", "median"),
            speed_kmh=("speed_kmh", "mean"),
            abs_steer=("abs_steer", "mean"),
            brake=("brake", "mean"),
            lateral_offset_m=("lateral_offset_m", "median"),
            abs_lateral_offset_m=("lateral_offset_m", lambda x: x.abs().mean()),
            p95_abs_lateral_offset_m=(
                "lateral_offset_m",
                lambda x: x.abs().quantile(0.95),
            ),
        )
        .reset_index()
    )


def short_run_label(name: str) -> str:
    parts = name.split("_")
    stamp = f"{parts[0][4:]}-{parts[1]}"
    weather = next(
        (x for x in ("clear", "scatteredclouds", "mist") if x in parts),
        "clear",
    )
    mode = "M" if "multi" in parts else "S"
    aggression = parts[-1] if mode == "M" and parts[-1].isdigit() else ""
    return f"{stamp} {mode} {weather[:5]}{('/A' + aggression) if aggression else ''}"


def heatmap_panel(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    colorbar_label: str,
) -> None:
    image = ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[0, 1, len(matrix), 0],
    )
    ax.set_title(title)
    ax.set_xlabel("Normalized track progress")
    ax.set_yticks(np.arange(len(matrix)) + 0.5)
    ax.set_yticklabels([short_run_label(x) for x in matrix.index], fontsize=8)
    plt.colorbar(image, ax=ax, label=colorbar_label, pad=0.01)


def save_run_level_plots(all_data: pd.DataFrame, profiles: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        2, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [11, 12]}
    )
    for ax, mode in zip(axes, ("solo", "multi")):
        matrix = (
            profiles[profiles["mode"] == mode]
            .pivot(index="session", columns="track_bin", values="speed_kmh")
            .reindex(SOLO_NAMES if mode == "solo" else MULTI_NAMES)
        )
        heatmap_panel(
            ax,
            matrix,
            f"{mode.upper()}: run-by-run speed over track position",
            "turbo",
            60,
            290,
            "Mean speed (km/h)",
        )
    fig.suptitle("Each row is one run; each column is 1% of Laguna Seca")
    fig.tight_layout()
    fig.savefig(OUT / "07_run_speed_heatmap.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(
        2, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [11, 12]}
    )
    for ax, mode in zip(axes, ("solo", "multi")):
        matrix = (
            profiles[profiles["mode"] == mode]
            .pivot(index="session", columns="track_bin", values="lateral_offset_m")
            .reindex(SOLO_NAMES if mode == "solo" else MULTI_NAMES)
        )
        heatmap_panel(
            ax,
            matrix,
            f"{mode.upper()}: median signed offset from AI racing line",
            "coolwarm",
            -4,
            4,
            "Signed lateral offset (m)",
        )
    fig.suptitle(
        "Blue/red are opposite sides of the AI line; white is near the AI line"
    )
    fig.tight_layout()
    fig.savefig(OUT / "08_run_lateral_offset_heatmap.png", dpi=190)
    plt.close(fig)

    ref = reference_geometry(V2 / "solo" / SOLO_NAMES[0])
    sessions = SOLO_NAMES + MULTI_NAMES
    fig, axes = plt.subplots(4, 6, figsize=(19, 15), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, sessions):
        part = all_data[all_data["session"] == name].iloc[::10]
        scatter = ax.scatter(
            part["x"],
            part["y"],
            c=part["lateral_offset_m"].abs().clip(0, 4),
            cmap="YlOrRd",
            vmin=0,
            vmax=4,
            s=1.1,
            alpha=0.72,
            rasterized=True,
        )
        ax.plot(
            ref["x"],
            ref["y"],
            color="black",
            linewidth=0.75,
            linestyle="--",
            alpha=0.9,
        )
        ax.set_title(short_run_label(name), fontsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[len(sessions):]:
        ax.axis("off")
    cbar = fig.colorbar(
        scatter,
        ax=axes.ravel().tolist(),
        fraction=0.012,
        pad=0.01,
        label="Absolute lateral error from AI line (m; clipped at 4 m)",
    )
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle(
        "AI racing line (black dashed) and absolute lateral error for every run"
    )
    fig.savefig(OUT / "09_run_trajectory_overlays.png", dpi=190)
    plt.close(fig)

    rows = []
    for name, part in all_data.groupby("session", sort=False):
        values = part["lateral_offset_m"].abs()
        rows.append(
            {
                "session": name,
                "label": short_run_label(name),
                "mode": part["mode"].iat[0],
                "mean": values.mean(),
                "p95": values.quantile(0.95),
                "p99": values.quantile(0.99),
            }
        )
    offsets = pd.DataFrame(rows)
    offsets.to_csv(OUT / "run_lateral_offset_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), sharex=True)
    for ax, mode in zip(axes, ("solo", "multi")):
        part = offsets[offsets["mode"] == mode].copy()
        y = np.arange(len(part))
        ax.barh(y, part["p99"], color="#d8d8d8", label="99th percentile")
        ax.barh(y, part["p95"], color="#e19a55", label="95th percentile")
        ax.barh(y, part["mean"], color="#356f9f", label="mean")
        ax.set_yticks(y)
        ax.set_yticklabels(part["label"], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Absolute offset from AI racing line (m)")
        ax.set_title(mode.upper())
        ax.grid(axis="x", alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Run-level AI racing-line deviation")
    fig.tight_layout()
    fig.savefig(OUT / "10_run_lateral_offset_summary.png", dpi=190)
    plt.close(fig)


def save_plots(
    all_data: pd.DataFrame,
    sessions: pd.DataFrame,
    profiles: pd.DataFrame,
    interactions: pd.DataFrame,
    opponent_hist: np.ndarray,
) -> None:
    colors = {"solo": "#2f6f9f", "multi": "#d05a3a"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    specs = [
        ("speed_kmh", np.linspace(0, 300, 61), "Speed (km/h)"),
        ("steer", np.linspace(-1, 1, 81), "Normalized steering"),
        ("throttle", np.linspace(0, 1, 51), "Throttle"),
        ("brake", np.linspace(0, 1, 51), "Brake"),
    ]
    for ax, (column, bins, label) in zip(axes.flat, specs):
        for mode in ("solo", "multi"):
            values = all_data.loc[all_data["mode"] == mode, column].dropna()
            ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.8,
                    label=mode, color=colors[mode])
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle("Laguna Seca control-state distributions")
    fig.tight_layout()
    fig.savefig(OUT / "01_control_distributions.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for mode in ("solo", "multi"):
        p = profiles[profiles["mode"] == mode]
        pivot = p.pivot(index="session", columns="track_bin", values="speed_kmh")
        x = (pivot.columns.to_numpy() + 0.5) / 100.0
        axes[0].plot(x, pivot.mean(axis=0), color=colors[mode], label=mode)
        axes[0].fill_between(
            x,
            pivot.quantile(0.1, axis=0),
            pivot.quantile(0.9, axis=0),
            color=colors[mode],
            alpha=0.18,
        )
        b = p.pivot(index="session", columns="track_bin", values="brake")
        axes[1].plot(x, b.mean(axis=0), color=colors[mode], label=mode)
    axes[0].set_ylabel("Mean speed (km/h)")
    axes[1].set_ylabel("Mean brake")
    axes[1].set_xlabel("Normalized track progress")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Track-position profiles (line: run mean, band: 10-90% runs)")
    fig.tight_layout()
    fig.savefig(OUT / "02_track_profiles.png", dpi=170)
    plt.close(fig)

    ref = reference_geometry(V2 / "solo" / SOLO_NAMES[0])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].plot(ref["x"], ref["y"], color="black", linewidth=0.8, alpha=0.5)
    h = axes[0].hexbin(
        all_data["x"], all_data["y"], gridsize=90, mincnt=1, bins="log", cmap="viridis"
    )
    fig.colorbar(h, ax=axes[0], label="log frame density")
    axes[0].set_title("Spatial frame density")
    axes[0].axis("equal")
    axes[0].set_xlabel("World x (m)")
    axes[0].set_ylabel("World y (m)")
    for mode in ("solo", "multi"):
        values = all_data.loc[all_data["mode"] == mode, "lateral_offset_m"]
        axes[1].hist(
            values.clip(-8, 8),
            bins=np.linspace(-8, 8, 81),
            density=True,
            histtype="step",
            linewidth=1.8,
            label=mode,
            color=colors[mode],
        )
    axes[1].set_title("Offset from AI racing line")
    axes[1].set_xlabel("Signed lateral offset (m; clipped)")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "03_spatial_coverage.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    weather = sessions.groupby(["mode", "weather"])["duration_s"].sum().unstack(fill_value=0) / 60
    weather.T.plot(kind="bar", ax=axes[0], color=[colors.get(c, "gray") for c in weather.index])
    axes[0].set_ylabel("Minutes")
    axes[0].set_title("Data duration by weather")
    axes[0].tick_params(axis="x", rotation=0)
    if len(interactions):
        counts = interactions.groupby("type").size().sort_values()
        counts.plot(kind="barh", ax=axes[1], color="#7b5ea7")
    axes[1].set_title("Interaction instances (multi)")
    axes[1].set_xlabel("Count")
    fig.tight_layout()
    fig.savefig(OUT / "04_balance_and_interactions.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    extent = [-15, 15, -60, 80]
    image = np.log1p(opponent_hist)
    im = ax.imshow(image, origin="lower", aspect="auto", extent=extent, cmap="magma")
    ax.axhline(0, color="white", linewidth=0.6, alpha=0.7)
    ax.axvline(0, color="white", linewidth=0.6, alpha=0.7)
    ax.set_xlabel("Opponent lateral position (m)")
    ax.set_ylabel("Opponent forward position (m)")
    ax.set_title("Valid opponent relative-position density (log1p counts)")
    fig.colorbar(im, ax=ax, label="log1p frame count")
    fig.tight_layout()
    fig.savefig(OUT / "05_opponent_relative_positions.png", dpi=170)
    plt.close(fig)

    solo = profiles[profiles["mode"] == "solo"].pivot(
        index="session", columns="track_bin", values="speed_kmh"
    )
    multi = profiles[profiles["mode"] == "multi"].pivot(
        index="session", columns="track_bin", values="speed_kmh"
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, matrix, title in (
        (axes[0], solo, "Solo run speed-profile correlation"),
        (axes[1], multi, "Multi run speed-profile correlation"),
    ):
        corr = matrix.T.corr().to_numpy()
        im = ax.imshow(corr, vmin=0.8, vmax=1.0, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Run index")
        ax.set_ylabel("Run index")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT / "06_run_redundancy.png", dpi=170)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    session_rows: list[dict[str, object]] = []
    lap_rows: list[dict[str, object]] = []
    interaction_data: list[dict[str, object]] = []
    data_parts: list[pd.DataFrame] = []
    infos: dict[str, dict[str, object]] = {}

    selected = [("solo", n) for n in SOLO_NAMES] + [("multi", n) for n in MULTI_NAMES]
    for number, (mode, name) in enumerate(selected, 1):
        print(f"[{number:02d}/{len(selected)}] loading {name}", flush=True)
        session = V2 / mode / name
        if not session.exists():
            raise FileNotFoundError(session)
        data, info = load_session(mode, name)
        infos[name] = info
        data_parts.append(data)
        session_rows.append(session_summary(data, info))
        lap_rows.extend(lap_summaries(data))
        if mode == "multi":
            interaction_data.extend(interaction_rows(session))

    all_data = pd.concat(data_parts, ignore_index=True)
    sessions = pd.DataFrame(session_rows)
    laps = pd.DataFrame(lap_rows)
    interactions = pd.DataFrame(interaction_data)
    profiles = binned_profile(all_data)
    lap_profiles = pd.concat(
        [lap_frame_profile(part) for part in data_parts],
        ignore_index=True,
    )

    hist = np.zeros((70, 60), dtype=np.int64)
    opponent_rows = []
    for name in MULTI_NAMES:
        print(f"[opponents] {name}", flush=True)
        part, stats = opponent_histogram(name, infos[name])
        hist += part
        opponent_rows.append(stats)
    opponents = pd.DataFrame(opponent_rows)

    sessions.to_csv(OUT / "session_summary.csv", index=False)
    laps.to_csv(OUT / "lap_summary.csv", index=False)
    profiles.to_csv(OUT / "track_bin_profiles.csv", index=False)
    lap_profiles.to_csv(OUT / "lap_frame_profiles.csv", index=False)
    interactions.to_csv(OUT / "interaction_summary.csv", index=False)
    opponents.to_csv(OUT / "opponent_summary.csv", index=False)

    distribution_rows = []
    for mode, part in all_data.groupby("mode"):
        distribution_rows.append(
            {
                "mode": mode,
                "frames": len(part),
                "speed_lt_20_fraction": (part["speed_kmh"] < 20).mean(),
                "speed_20_60_fraction": part["speed_kmh"].between(20, 60, inclusive="left").mean(),
                "speed_60_120_fraction": part["speed_kmh"].between(60, 120, inclusive="left").mean(),
                "speed_120_200_fraction": part["speed_kmh"].between(120, 200, inclusive="left").mean(),
                "speed_ge_200_fraction": (part["speed_kmh"] >= 200).mean(),
                "abs_steer_lt_005_fraction": (part["abs_steer"] < 0.05).mean(),
                "abs_steer_ge_05_fraction": (part["abs_steer"] >= 0.5).mean(),
                "brake_active_fraction": (part["brake"] > 0.05).mean(),
                "throttle_active_fraction": (part["throttle"] > 0.05).mean(),
                "throttle_brake_overlap_fraction": (
                    (part["throttle"] > 0.05) & (part["brake"] > 0.05)
                ).mean(),
                "abs_lateral_gt_2m_fraction": (part["lateral_offset_m"].abs() > 2).mean(),
                "abs_lateral_gt_4m_fraction": (part["lateral_offset_m"].abs() > 4).mean(),
                "abs_lateral_gt_6m_fraction": (part["lateral_offset_m"].abs() > 6).mean(),
            }
        )
    pd.DataFrame(distribution_rows).to_csv(
        OUT / "distribution_summary.csv", index=False
    )

    run_matrix = profiles.pivot(index=["mode", "session"], columns="track_bin", values="speed_kmh")
    run_matrix.T.corr().to_csv(OUT / "run_speed_profile_correlations.csv")

    compact = {
        "selected_solo_sessions": len(SOLO_NAMES),
        "selected_multi_sessions": len(MULTI_NAMES),
        "frames": int(len(all_data)),
        "minutes": float(len(all_data) / 40.0 / 60.0),
        "solo_frames": int((all_data["mode"] == "solo").sum()),
        "multi_frames": int((all_data["mode"] == "multi").sum()),
        "weather_minutes": {
            mode: {
                weather: float(count / (40 * 60))
                for weather, count in part.groupby("weather").size().items()
            }
            for mode, part in all_data.groupby("mode")
        },
        "interaction_instances": (
            interactions.groupby("type").size().to_dict() if len(interactions) else {}
        ),
        "median_solo_run_correlation": float(
            run_matrix.loc["solo"].T.corr().where(
                ~np.eye(len(SOLO_NAMES), dtype=bool)
            ).stack().median()
        ),
        "median_multi_run_correlation": float(
            run_matrix.loc["multi"].T.corr().where(
                ~np.eye(len(MULTI_NAMES), dtype=bool)
            ).stack().median()
        ),
    }
    (OUT / "summary.json").write_text(
        json.dumps(compact, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    save_plots(all_data, sessions, profiles, interactions, hist)
    save_run_level_plots(all_data, profiles)
    print(json.dumps(compact, indent=2, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
