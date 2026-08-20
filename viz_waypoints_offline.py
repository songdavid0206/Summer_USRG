"""Offline (open-loop) waypoint viewer for a trained TCP-v5 model.

No driving needed: it runs the checkpoint over the prepared val/train split and
shows, per sample, the model's predicted 6 waypoints vs the ground-truth
waypoints and the input racing-line reference -- all in the car's ego frame
(origin = car, +y forward, +x right). Use it to see "what trajectory did the
model actually learn" for models like v8 whose closed-loop driving is unstable.

Keys (interactive window):
    Right/Left        +1 / -1 sample
    Shift+Right/Left  +10 / -10
    Up/Down           +100 / -100
    Space             play / pause
    q                 quit

Usage:
    python viz_waypoints_offline.py --prepared training/tcp_lagunaseca_v8
    python viz_waypoints_offline.py --prepared training/tcp_lagunaseca_v8 --split train
    python viz_waypoints_offline.py --prepared training/tcp_lagunaseca_v8 --save-frame 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, default=Path("training/tcp_lagunaseca_v8"))
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="default: <prepared>/checkpoints/best.pt")
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--save-frame", type=int, default=None,
                    help="headless: render this sample index to PNG and exit")
    args = ap.parse_args()

    import torch
    from ac_dataset.tcp.data_v5 import TCPDatasetV5
    from ac_dataset.tcp.model_v5 import build_model_v5

    ckpt_path = args.checkpoint or (args.prepared / "checkpoints" / "best.pt")
    device = ("cuda" if (args.device == "auto" and torch.cuda.is_available())
              else "cpu" if args.device == "auto" else args.device)

    ds = TCPDatasetV5(args.prepared, args.split)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model_v5(False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    norm = json.loads((args.prepared / "normalization.json").read_text(encoding="utf-8"))
    wp_mean = np.asarray(norm["waypoints_local_m"]["mean"])
    wp_std = np.asarray(norm["waypoints_local_m"]["std"])
    ln_mean = np.asarray(norm["racing_line_local_m"]["mean"])
    ln_std = np.asarray(norm["racing_line_local_m"]["std"])
    print(f"[offline] {ckpt_path}  split={args.split}  samples={len(ds)}  device={device}")

    # Track edges: look up side_left/side_right at each racing-line lookahead and
    # offset the ego-frame input racing line perpendicular to its own tangent.
    import pandas as pd
    from ac_dataset.tcp.shadow import discover_racing_line
    from ac_dataset.tcp.config import RACING_LINE_LOOKAHEADS_M
    csv = pd.read_csv(discover_racing_line(args.prepared))
    csv_norm = csv["normalized_position"].to_numpy(float)
    csv_dist = csv["dist"].to_numpy(float)
    csv_sL = csv["side_left"].to_numpy(float)
    csv_sR = csv["side_right"].to_numpy(float)
    csv_xy = csv[["x", "y"]].to_numpy(float)
    track_len = float(csv_dist[-1] + np.linalg.norm(csv_xy[0] - csv_xy[-1]))
    lookaheads = np.asarray(RACING_LINE_LOOKAHEADS_M, float)
    # NOTE: the per-sample spline comes from ds.samples.iloc[i] (the split table the
    # dataset actually indexes). item["source_id"] is not a manifest row index.

    def ego_edges(line_ego, spline):
        t = np.gradient(line_ego, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        n = np.column_stack((-t[:, 1], t[:, 0]))  # left perpendicular in ego frame
        d0 = float(np.interp(spline % 1.0, csv_norm, csv_dist))
        idx = np.searchsorted(csv_dist, (d0 + lookaheads) % track_len).clip(0, len(csv_dist) - 1)
        left = line_ego + n * csv_sL[idx][:, None]
        right = line_ego - n * csv_sR[idx][:, None]
        return left, right

    def infer(i: int):
        s = ds[i]
        with torch.no_grad():
            out = model(s["image"][None].to(device), s["ego"][None].to(device),
                        s["racing_line"][None].to(device), s["command"][None].to(device),
                        s["target_speed"][None].to(device))
        pred = out["waypoints"][0].cpu().numpy() * wp_std + wp_mean
        gt = np.asarray(s["waypoints_m"])
        line = np.asarray(s["racing_line"]) * ln_std + ln_mean
        spd = float(np.asarray(s["speed"]).reshape(-1)[0]) if "speed" in s else float("nan")
        spline = float(ds.samples.iloc[i]["spline_position"]) % 1.0
        left, right = ego_edges(line, spline)
        return pred, gt, line, spd, spline, left, right

    fig, ax = plt.subplots(figsize=(7, 9))
    edgeL_a, = ax.plot([], [], color="0.55", lw=1.4, label="track edge")
    edgeR_a, = ax.plot([], [], color="0.55", lw=1.4)
    line_a, = ax.plot([], [], color="0.6", lw=1.2, marker=".", ms=6, ls="--",
                      label="input racing line (11 pt)")
    gt_a, = ax.plot([], [], color="tab:green", lw=2.0, marker="o", ms=6,
                    label="ground-truth waypoints")
    pred_a, = ax.plot([], [], color="tab:blue", lw=1.8, marker="o", ms=6,
                      label="model predicted waypoints")
    ax.scatter([0], [0], color="lime", s=90, ec="k", zorder=5, label="car")
    ax.legend(loc="upper right", fontsize=8)
    state = {"i": int(np.clip(args.index, 0, len(ds) - 1)), "playing": False}

    def draw():
        i = state["i"]
        pred, gt, line, spd, spline, left, right = infer(i)
        # AC's to_local puts +x on the driver's LEFT (a left turn yaws positive and
        # produces +x waypoints). Negate x so the plot reads like the driver's view:
        # left turns bend left on screen. Cosmetic only; the data is unchanged.
        edgeL_a.set_data(-left[:, 0], left[:, 1])
        edgeR_a.set_data(-right[:, 0], right[:, 1])
        line_a.set_data(-line[:, 0], line[:, 1])
        gt_a.set_data(np.r_[0, -gt[:, 0]], np.r_[0, gt[:, 1]])
        pred_a.set_data(np.r_[0, -pred[:, 0]], np.r_[0, pred[:, 1]])
        err = float(np.linalg.norm(pred - gt, axis=1).mean())
        # Frame the view on the waypoints; edges beyond the window are clipped.
        fwd = max(gt[:, 1].max(), pred[:, 1].max(), 25.0)
        lat = max(np.abs(gt[:, 0]).max(), np.abs(pred[:, 0]).max(), 6.0) + 12.0
        ax.set_xlim(-lat, lat)
        ax.set_ylim(-5, fwd + 10)
        ax.set_aspect("equal")
        ax.set_title(f"sample {i}/{len(ds)-1}  spline {spline*100:.1f}%  {spd*3.6:.0f} km/h  "
                     f"mean wp err {err:.2f} m\n"
                     "[<-/-> step, Shift +-10, Up/Down +-100, Space play]")
        fig.canvas.draw_idle()

    def step(n):
        state["i"] = int(np.clip(state["i"] + n, 0, len(ds) - 1)); draw()

    def on_key(event):
        k = event.key
        if k == "right": step(1)
        elif k == "left": step(-1)
        elif k == "shift+right": step(10)
        elif k == "shift+left": step(-10)
        elif k == "up": step(100)
        elif k == "down": step(-100)
        elif k == " ": state["playing"] = not state["playing"]
        elif k == "q": plt.close(fig)

    if args.save_frame is not None:
        state["i"] = int(np.clip(args.save_frame, 0, len(ds) - 1))
        draw()
        out = args.prepared / f"offline_waypoints_{state['i']:05d}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"[wrote] {out}")
        return

    fig.canvas.mpl_connect("key_press_event", on_key)
    timer = fig.canvas.new_timer(interval=60)
    timer.add_callback(lambda: (state["playing"] and step(2)))
    timer.start()
    draw()
    print("[offline] window open. <-/-> step, Shift +-10, Up/Down +-100, Space play, q quit")
    plt.show()


if __name__ == "__main__":
    main()
