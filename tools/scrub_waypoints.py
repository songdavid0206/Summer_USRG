"""Interactive closed-loop waypoint scrubber (av-view style).

Opens a window that follows the car through the run. Step frame-by-frame with
the keyboard and watch the model's 6 predicted waypoints (0.1-2.0 s ahead) sweep
forward over the AI racing line and track edges. No camera frames needed.

Keys:
    Right / Left        step +1 / -1 frame
    Shift+Right / Left  jump +10 / -10
    Up / Down           jump +50 / -50
    Space               play / pause
    Home / End          first / last frame
    q                   quit

Usage:
    python tools/scrub_waypoints.py --prepared training/tcp_lagunaseca_v4
    python tools/scrub_waypoints.py --prepared training/tcp_lagunaseca_v4 --run <dir> --window 70
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ac_dataset.tcp.shadow import RacingLineTarget, discover_racing_line


def ego_to_world(local_xy: np.ndarray, origin: np.ndarray, yaw: float) -> np.ndarray:
    local_xy = np.asarray(local_xy, float).reshape(-1, 2)
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    right = np.array([-np.sin(yaw), np.cos(yaw)])
    return origin + local_xy[:, 0:1] * right + local_xy[:, 1:2] * forward


def track_edges(line_csv: Path):
    """Reconstruct left/right track edges from side_left/side_right distances."""
    df = pd.read_csv(line_csv)
    xy = df[["x", "y"]].to_numpy(float)
    # Unit tangent via central differences (wrap-around closed loop).
    tang = np.gradient(xy, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    normal = np.column_stack((-tang[:, 1], tang[:, 0]))  # +90 deg (left)
    left = xy + normal * df["side_left"].to_numpy(float)[:, None]
    right = xy - normal * df["side_right"].to_numpy(float)[:, None]
    return xy, left, right


def latest_run(output_root: Path) -> Path:
    runs = sorted(p for p in output_root.glob("*/") if (p / "closed_loop.jsonl").exists())
    if not runs:
        raise SystemExit(f"no closed_loop.jsonl under {output_root}")
    return runs[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--window", type=float, default=70.0,
                    help="half-size (m) of the view that follows the car")
    ap.add_argument("--save-frame", type=int, default=None,
                    help="headless: render this frame index to PNG and exit")
    ap.add_argument("--gif", nargs=2, type=int, default=None, metavar=("START", "END"),
                    help="headless: render frames START..END to a GIF and exit")
    ap.add_argument("--gif-step", type=int, default=2,
                    help="use every Nth frame in the GIF range")
    ap.add_argument("--gif-fps", type=int, default=20)
    ap.add_argument("--gif-out", type=Path, default=None)
    args = ap.parse_args()

    run_dir = args.run or latest_run(args.prepared / "closed_loop")
    rows = [json.loads(l) for l in (run_dir / "closed_loop.jsonl").open(encoding="utf-8")]
    line_csv = discover_racing_line(args.prepared)
    xy, left, right = track_edges(line_csv)
    px = np.array([r["position_x"] for r in rows])
    py = np.array([r["position_y"] for r in rows])
    print(f"[scrub] {run_dir}  frames={len(rows)}")

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(left[:, 0], left[:, 1], color="0.55", lw=1.0)
    ax.plot(right[:, 0], right[:, 1], color="0.55", lw=1.0, label="track edge")
    ax.plot(xy[:, 0], xy[:, 1], color="black", lw=1.4, ls="--", label="AI racing line")
    trail, = ax.plot([], [], color="red", lw=2.0, label="driven path")
    wp_line, = ax.plot([], [], color="tab:blue", lw=1.6, marker="o", ms=5,
                       label="model waypoints (0.1-2.0s)")
    car = ax.scatter([], [], color="lime", s=90, zorder=6, ec="k", label="car")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)

    state = {"i": 0, "playing": False}

    def draw():
        i = state["i"]
        r = rows[i]
        origin = np.array([r["position_x"], r["position_y"]])
        yaw = float(r["yaw_rad"])
        wp = ego_to_world(np.array(r["pred_waypoints_m"]), origin, yaw)
        wp = np.vstack([origin, wp])
        wp_line.set_data(wp[:, 0], wp[:, 1])
        lo = max(0, i - 250)
        trail.set_data(px[lo:i + 1], py[lo:i + 1])
        car.set_offsets([origin])
        # true perpendicular distance to the AI line (nearest point), not the
        # logged spline-indexed deviation which is inflated by the AC-spline vs
        # racing-line parameterization offset. On the line this reads ~0.
        dev = float(np.linalg.norm(xy - origin, axis=1).min())
        w = args.window
        ax.set_xlim(origin[0] - w, origin[0] + w)
        ax.set_ylim(origin[1] - w, origin[1] + w)
        ax.set_title(f"frame {i}/{len(rows)-1}   spline {r['spline_position']:.3f}   "
                     f"{r['speed_mps']*3.6:.0f} km/h   dev {dev:.1f} m"
                     f"   [<-/-> step, Shift +-10, Up/Down +-50, Space play]")
        fig.canvas.draw_idle()

    def step(n):
        state["i"] = int(np.clip(state["i"] + n, 0, len(rows) - 1))
        draw()

    def on_key(event):
        k = event.key
        if k == "right": step(1)
        elif k == "left": step(-1)
        elif k == "shift+right": step(10)
        elif k == "shift+left": step(-10)
        elif k == "up": step(50)
        elif k == "down": step(-50)
        elif k == "home": state["i"] = 0; draw()
        elif k == "end": state["i"] = len(rows) - 1; draw()
        elif k == " ": state["playing"] = not state["playing"]
        elif k == "q": plt.close(fig)

    def on_timer():
        if state["playing"]:
            if state["i"] >= len(rows) - 1:
                state["playing"] = False
            else:
                step(2)

    if args.gif is not None:
        from matplotlib.animation import FuncAnimation, PillowWriter
        lo = max(0, int(args.gif[0]))
        hi = min(len(rows) - 1, int(args.gif[1]))
        frames = list(range(lo, hi + 1, max(1, args.gif_step)))
        ax.legend(loc="upper right", fontsize=8)

        def _frame(idx):
            state["i"] = idx
            draw()
            return wp_line, trail, car

        anim = FuncAnimation(fig, _frame, frames=frames, blit=False)
        out = args.gif_out or (run_dir / f"scrub_{lo}_{hi}.gif")
        anim.save(out, writer=PillowWriter(fps=args.gif_fps))
        plt.close(fig)
        print(f"[wrote] {out}  ({len(frames)} frames @ {args.gif_fps}fps)")
        return

    if args.save_frame is not None:
        state["i"] = int(np.clip(args.save_frame, 0, len(rows) - 1))
        draw()
        out = run_dir / f"scrub_preview_{state['i']:05d}.png"
        fig.savefig(out, dpi=120)
        print(f"[wrote] {out}")
        return

    fig.canvas.mpl_connect("key_press_event", on_key)
    timer = fig.canvas.new_timer(interval=40)
    timer.add_callback(on_timer)
    timer.start()
    draw()
    print("[scrub] window open. keys: <-/-> step, Shift +-10, Up/Down +-50, Space play, q quit")
    plt.show()


if __name__ == "__main__":
    main()
