"""Visualize CSV racing-line geometry against recorded AC trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def binned_percentiles(spline: np.ndarray, values: np.ndarray, bins: int = 200):
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    indices = np.clip(np.digitize(spline % 1.0, edges) - 1, 0, bins - 1)
    p50 = np.full(bins, np.nan)
    p95 = np.full(bins, np.nan)
    for index in range(bins):
        selected = values[indices == index]
        if len(selected):
            p50[index], p95[index] = np.percentile(selected, [50, 95])
    return centers, p50, p95


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--closed-loop", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = pd.read_parquet(args.prepared / "manifest.parquet")
    normal = manifest[manifest.source_type == "normal"].copy()
    line_path = Path(str(normal.iloc[0].session_path)) / "TRACK_INFO" / "AI_RACING_LINE.csv"
    line = pd.read_csv(line_path)
    normal["line_deviation_m"] = np.hypot(
        normal.line_0m_x - normal.pos_x, normal.line_0m_y - normal.pos_y)

    closed = pd.DataFrame()
    if args.closed_loop and args.closed_loop.exists():
        closed = pd.DataFrame(
            json.loads(row) for row in args.closed_loop.read_text(encoding="utf-8").splitlines()
            if row.strip())

    figure, axes = plt.subplots(1, 3, figsize=(21, 7))
    full, zoom, deviation = axes

    for number, (_, session) in enumerate(normal.groupby("session", sort=False)):
        sampled = session.iloc[::20]
        full.plot(sampled.pos_x, sampled.pos_y, color="tab:blue", alpha=.20,
                  linewidth=.7, label="recorded AC AI" if number == 0 else None)
    full.plot(line.x, line.y, color="black", linewidth=2.0, label="AI_RACING_LINE.csv")
    if not closed.empty:
        full.plot(closed.position_x, closed.position_y, color="tab:red", linewidth=2.0,
                  label="latest v4 closed-loop")
    full.set_title("Full-track geometry overlay")
    full.set_xlabel("world x (m)")
    full.set_ylabel("world y (m)")
    full.axis("equal")
    full.grid(alpha=.2)
    full.legend()

    sector_min, sector_max = .08, .15
    line_sector = line[(line.normalized_position >= sector_min)
                       & (line.normalized_position <= sector_max)]
    for number, (_, session) in enumerate(normal.groupby("session", sort=False)):
        sector = session[(session.spline_position >= sector_min)
                         & (session.spline_position <= sector_max)].iloc[::4]
        zoom.plot(sector.pos_x, sector.pos_y, color="tab:blue", alpha=.20,
                  linewidth=.8, label="recorded AC AI" if number == 0 else None)
    zoom.plot(line_sector.x, line_sector.y, color="black", linewidth=2.2,
              label="AI_RACING_LINE.csv")
    if not closed.empty:
        sector = closed[(closed.spline_position >= sector_min)
                        & (closed.spline_position <= sector_max)]
        zoom.plot(sector.position_x, sector.position_y, color="tab:red", linewidth=2.2,
                  label="latest v4 closed-loop")
        if len(sector):
            zoom.scatter(sector.position_x.iloc[-1], sector.position_y.iloc[-1],
                         color="red", marker="x", s=100, label="last logged point")
    zoom.set_title("First-corner sector (spline 0.08-0.15)")
    zoom.set_xlabel("world x (m)")
    zoom.set_ylabel("world y (m)")
    zoom.axis("equal")
    zoom.grid(alpha=.2)
    zoom.legend()

    centers, p50, p95 = binned_percentiles(
        normal.spline_position.to_numpy(float), normal.line_deviation_m.to_numpy(float))
    deviation.plot(centers, p50, color="tab:blue", label="AC AI median")
    deviation.plot(centers, p95, color="tab:blue", linestyle="--", label="AC AI p95")
    if not closed.empty:
        deviation.plot(closed.spline_position, closed.racing_line_deviation_m,
                       color="tab:red", alpha=.85, linewidth=1.5, label="v4 closed-loop")
    deviation.axhline(10.0, color="tab:red", linestyle=":", label="10 m stop limit")
    deviation.set_xlim(0, 1)
    deviation.set_ylim(bottom=0)
    deviation.set_title("Distance from CSV line by track progress")
    deviation.set_xlabel("spline position")
    deviation.set_ylabel("distance (m)")
    deviation.grid(alpha=.2)
    deviation.legend()

    figure.suptitle(
        "Laguna Seca: reference line vs actual trajectories\n"
        f"normal AI median={normal.line_deviation_m.median():.2f} m, "
        f"p95={normal.line_deviation_m.quantile(.95):.2f} m, "
        f"p99={normal.line_deviation_m.quantile(.99):.2f} m")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    print(f"[saved] {args.output}")
    print(f"[racing-line] {line_path}")


if __name__ == "__main__":
    main()
