"""Render closed-loop run diagnostics.

Reads a closed_loop.jsonl run and the prepared racing line, then writes:

  * track_map.png          - full lap: racing line + driven path colored by deviation
  * waypoints_<tag>.png     - per-corner bird's-eye with the model's predicted
                              waypoints (from the checkpoint) transformed into
                              world coordinates, overlaid on the racing line.

The per-corner view is the one that answers "does the model aim off-line, or is
it the controller that fails to track?": if the blue model waypoints curve back
onto the black racing line while the red driven path runs wide, the waypoints are
fine and the steering actuation is too weak; if the blue waypoints themselves run
wide with the car, the model is the cause.

Usage:
    python tools/visualize_closed_loop.py --prepared training/tcp_lagunaseca_v4 \
        [--run <run_dir>] [--corner 0.08 0.13 --corner 0.13 0.17]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ac_dataset.tcp.shadow import RacingLineTarget, discover_racing_line


def ego_to_world(local_xy: np.ndarray, origin: np.ndarray, yaw: float) -> np.ndarray:
    """Invert data.to_local: local=[lateral(right), forward] -> world xy."""
    local_xy = np.asarray(local_xy, float).reshape(-1, 2)
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    right = np.array([-np.sin(yaw), np.cos(yaw)])
    return origin + local_xy[:, 0:1] * right + local_xy[:, 1:2] * forward


def latest_run(output_root: Path) -> Path:
    runs = sorted(p for p in output_root.glob("*/") if (p / "closed_loop.jsonl").exists())
    if not runs:
        raise SystemExit(f"no closed_loop.jsonl under {output_root}")
    return runs[-1]


def load_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "closed_loop.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def track_map(rows, rl: RacingLineTarget, out: Path) -> None:
    px = np.array([r["position_x"] for r in rows])
    py = np.array([r["position_y"] for r in rows])
    dev = np.array([r["racing_line_deviation_m"] for r in rows])
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.plot(rl.xy[:, 0], rl.xy[:, 1], color="0.7", lw=1.0, label="racing line", zorder=1)
    sc = ax.scatter(px, py, c=dev, cmap="turbo", s=6, vmin=0, vmax=6,
                    label="driven path", zorder=2)
    worst = int(dev.argmax())
    ax.scatter([px[worst]], [py[worst]], s=90, facecolors="none",
               edgecolors="red", lw=2, zorder=3,
               label=f"max dev {dev[worst]:.1f}m @spline {rows[worst]['spline_position']:.3f}")
    fig.colorbar(sc, ax=ax, label="racing-line deviation (m)")
    ax.set_aspect("equal")
    ax.set_title("Closed-loop driven path vs racing line")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[wrote] {out}")


def corner_view(rows, rl: RacingLineTarget, lo: float, hi: float, out: Path,
                stride: int = 6) -> None:
    sel = [i for i, r in enumerate(rows)
           if lo <= (r["spline_position"] % 1.0) <= hi]
    if not sel:
        print(f"[skip] no frames in spline [{lo}, {hi}]")
        return
    # Racing-line segment for this spline window (plus margin).
    mask = (rl.norm >= max(0.0, lo - 0.01)) & (rl.norm <= min(1.0, hi + 0.01))
    line = rl.xy[mask]

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.plot(line[:, 0], line[:, 1], color="black", lw=2.2,
            label="racing line", zorder=1)
    # Split into contiguous passes so multi-lap windows don't draw a straight
    # connector between one lap's exit and the next lap's entry.
    passes, current = [], [sel[0]]
    for a, b in zip(sel, sel[1:]):
        if b - a == 1:
            current.append(b)
        else:
            passes.append(current)
            current = [b]
    passes.append(current)
    for k, seg in enumerate(passes):
        px = np.array([rows[i]["position_x"] for i in seg])
        py = np.array([rows[i]["position_y"] for i in seg])
        ax.plot(px, py, color="red", lw=1.6, alpha=0.9, zorder=2,
                label="driven path" if k == 0 else None)

    labelled = False
    for i in sel[::stride]:
        r = rows[i]
        origin = np.array([r["position_x"], r["position_y"]])
        yaw = float(r["yaw_rad"])
        wp = ego_to_world(np.array(r["pred_waypoints_m"]), origin, yaw)
        wp = np.vstack([origin, wp])  # start the polyline at the car
        ax.plot(wp[:, 0], wp[:, 1], color="tab:blue", lw=1.0, alpha=0.55,
                zorder=3, label="model waypoints" if not labelled else None)
        ax.scatter(wp[1:, 0], wp[1:, 1], color="tab:blue", s=8, alpha=0.55, zorder=3)
        labelled = True

    ax.scatter([rows[sel[0]]["position_x"]], [rows[sel[0]]["position_y"]],
               color="lime", s=60, zorder=4, label="entry")
    ax.set_aspect("equal")
    dmax = max(r["racing_line_deviation_m"] for r in (rows[i] for i in sel))
    ax.set_title(f"Corner spline [{lo:.2f}, {hi:.2f}]  max dev {dmax:.1f} m\n"
                 "blue = where the model wants to go; red = where the car went")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[wrote] {out}  ({len(sel)} frames)")


def animate_corner(rows, rl: RacingLineTarget, lo: float, hi: float, out: Path,
                   step: int = 2, fps: int = 15) -> None:
    """av-view style: drive through the corner, drawing the 6 model waypoints
    fanning forward from the car at every frame."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    sel = [i for i, r in enumerate(rows)
           if lo <= (r["spline_position"] % 1.0) <= hi]
    if not sel:
        print(f"[skip] no frames in spline [{lo}, {hi}]")
        return
    # First contiguous pass only (one lap through the corner).
    seg = [sel[0]]
    for a, b in zip(sel, sel[1:]):
        if b - a == 1:
            seg.append(b)
        else:
            break
    seg = seg[::step]

    mask = (rl.norm >= max(0.0, lo - 0.01)) & (rl.norm <= min(1.0, hi + 0.01))
    line = rl.xy[mask]
    px = np.array([rows[i]["position_x"] for i in seg])
    py = np.array([rows[i]["position_y"] for i in seg])
    xs = np.concatenate([line[:, 0], px]); ys = np.concatenate([line[:, 1], py])
    pad = 12.0

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(line[:, 0], line[:, 1], color="black", lw=2.2, label="racing line")
    trail, = ax.plot([], [], color="red", lw=1.6, label="driven path")
    wp_line, = ax.plot([], [], color="tab:blue", lw=1.4, marker="o", ms=4,
                       label="model waypoints (0.1-2.0s)")
    car = ax.scatter([], [], color="lime", s=70, zorder=5, label="car")
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    title = ax.set_title("")

    def frame(k):
        i = seg[k]
        r = rows[i]
        origin = np.array([r["position_x"], r["position_y"]])
        yaw = float(r["yaw_rad"])
        wp = ego_to_world(np.array(r["pred_waypoints_m"]), origin, yaw)
        wp = np.vstack([origin, wp])
        wp_line.set_data(wp[:, 0], wp[:, 1])
        trail.set_data(px[:k + 1], py[:k + 1])
        car.set_offsets([origin])
        title.set_text(f"spline {r['spline_position']:.3f}  "
                       f"{r['speed_mps'] * 3.6:.0f} km/h  dev {r['racing_line_deviation_m']:.1f} m")
        return wp_line, trail, car, title

    anim = FuncAnimation(fig, frame, frames=len(seg), blit=False)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[wrote] {out}  ({len(seg)} frames @ {fps}fps)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--run", type=Path, default=None,
                    help="run dir; default = latest under <prepared>/closed_loop")
    ap.add_argument("--corner", nargs=2, type=float, action="append", metavar=("LO", "HI"),
                    help="spline window to zoom (repeatable)")
    ap.add_argument("--stride", type=int, default=6,
                    help="draw model waypoints every Nth frame")
    ap.add_argument("--animate", action="store_true",
                    help="also render an av-view style GIF per corner")
    args = ap.parse_args()

    run_dir = args.run or latest_run(args.prepared / "closed_loop")
    rows = load_rows(run_dir)
    rl = RacingLineTarget.from_prepared(args.prepared, discover_racing_line(args.prepared))
    print(f"[run] {run_dir}  frames={len(rows)}")

    track_map(rows, rl, run_dir / "track_map.png")
    corners = args.corner or [[0.08, 0.13], [0.13, 0.17]]
    for lo, hi in corners:
        tag = f"{lo:.2f}_{hi:.2f}".replace(".", "")
        corner_view(rows, rl, lo, hi, run_dir / f"waypoints_{tag}.png", args.stride)
        if args.animate:
            animate_corner(rows, rl, lo, hi, run_dir / f"waypoints_{tag}.gif")


if __name__ == "__main__":
    main()
