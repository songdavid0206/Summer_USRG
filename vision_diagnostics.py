"""Grad-CAM and occlusion diagnostics for a trained TCP-AC checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ac_dataset.tcp.config import (
    EGO_FEATURE_NAMES, POSITION_SCALE_M, RACING_LINE_LOOKAHEADS_M,
    STANDING_LAP_COMMAND_INDEX, FLYING_LAP_COMMAND_INDEX,
    target_speed_input_width,
)
from ac_dataset.tcp.data import preprocess_rgb
from ac_dataset.tcp.model import build_model


CONTROL_NAMES = ("steer", "throttle", "brake")


def inputs_for(row: dict, image_path: Path, device, target_width: int):
    import torch

    image = Image.open(image_path).convert("RGB")
    tensor = preprocess_rgb(image).unsqueeze(0).to(device)
    ego = torch.tensor(row["ego_normalized"], dtype=torch.float32, device=device).reshape(
        1, len(EGO_FEATURE_NAMES))
    line = torch.tensor(
        row["racing_line_normalized"], dtype=torch.float32, device=device).reshape(
            1, len(RACING_LINE_LOOKAHEADS_M), 2)
    command = torch.zeros((1, 6), dtype=torch.float32, device=device)
    command[0, 3] = 1
    command[0, STANDING_LAP_COMMAND_INDEX if row["lap_phase"] == "standing"
            else FLYING_LAP_COMMAND_INDEX] = 1
    target = None
    if target_width:
        target = torch.tensor(
            row["target_speed_features"], dtype=torch.float32, device=device).reshape(
                1, target_width)
    display = np.asarray(image.resize((512, 256), Image.Resampling.BILINEAR))
    return display, tensor, ego, line, command, target


def overlay(display: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    colored = plt.get_cmap("jet")(np.clip(heatmap, 0, 1))[..., :3]
    return np.clip(display / 255.0 * .55 + colored * .45, 0, 1)


def gradcams(model, image, ego, line, command, target):
    import torch
    import torch.nn.functional as F

    captured = {}

    def hook(_module, _inputs, output):
        captured["feature"] = output
        output.retain_grad()

    handle = model.encoder[-1].register_forward_hook(hook)
    maps = []
    predictions = None
    try:
        for control_index in range(3):
            model.zero_grad(set_to_none=True)
            pred = model(image, ego, line, command, target)
            predictions = pred["controls"][0].detach().cpu().numpy()
            value = pred["controls"][0, control_index]
            if control_index == 0 and float(value.detach()) < 0:
                value = -value
            value.backward()
            feature = captured["feature"]
            weights = feature.grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * feature).sum(dim=1, keepdim=True).relu()
            cam = F.interpolate(cam, size=(256, 512), mode="bilinear", align_corners=False)
            cam = cam[0, 0]
            cam = cam / torch.clamp(cam.max(), min=1e-8)
            maps.append(cam.detach().cpu().numpy())
    finally:
        handle.remove()
    return predictions, maps


def occlusion_map(model, image, ego, line, command, target, rows=6, cols=12):
    import torch
    import torch.nn.functional as F

    with torch.inference_mode():
        baseline = model(image, ego, line, command, target)["controls"]
    variants = []
    for row in range(rows):
        for col in range(cols):
            masked = image.clone()
            y0, y1 = row * 256 // rows, (row + 1) * 256 // rows
            x0, x1 = col * 512 // cols, (col + 1) * 512 // cols
            masked[:, :, y0:y1, x0:x1] = 0.0
            variants.append(masked)
    scores = []
    for start in range(0, len(variants), 12):
        batch = torch.cat(variants[start:start + 12], dim=0)
        count = len(batch)
        with torch.inference_mode():
            controls = model(
                batch, ego.expand(count, -1), line.expand(count, -1, -1),
                command.expand(count, -1),
                target.expand(count, -1) if target is not None else None,
            )["controls"]
        scores.extend((controls - baseline).abs().sum(dim=1).detach().cpu().tolist())
    grid = np.asarray(scores, np.float32).reshape(rows, cols)
    grid /= max(float(grid.max()), 1e-8)
    heatmap = F.interpolate(
        torch.from_numpy(grid)[None, None], size=(256, 512), mode="nearest",
    )[0, 0].numpy()
    return heatmap, grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shadow-log", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 32, 328, 1168])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    width = target_speed_input_width(checkpoint.get("model_schema"))
    model = build_model(False, target_speed_width=width).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    log = {row["index"]: row for row in (
        json.loads(line) for line in args.shadow_log.read_text(encoding="utf-8").splitlines()
        if line.strip())}

    figure, axes = plt.subplots(len(args.indices), 5, figsize=(20, 3.2 * len(args.indices)))
    report = []
    for row_number, index in enumerate(args.indices):
        row = log[index]
        path = Path(row["saved_frame"])
        display, image, ego, line, command, target = inputs_for(
            row, path, device, width)
        prediction, cams = gradcams(model, image, ego, line, command, target)
        occ, grid = occlusion_map(model, image, ego, line, command, target)
        actual = np.asarray([
            row["actual_steer"], row["actual_throttle"], row["actual_brake"]])
        axes[row_number, 0].imshow(display)
        axes[row_number, 0].set_title(
            f"frame {index}, {row['speed_mps']*3.6:.0f} km/h\n"
            f"pred={np.round(prediction,3)} actual={np.round(actual,3)}")
        for column, (name, cam) in enumerate(zip(CONTROL_NAMES, cams), 1):
            axes[row_number, column].imshow(overlay(display, cam))
            axes[row_number, column].set_title(f"Grad-CAM: {name}")
        axes[row_number, 4].imshow(overlay(display, occ))
        peak = np.unravel_index(int(np.argmax(grid)), grid.shape)
        axes[row_number, 4].set_title(
            f"Occlusion sensitivity\npeak grid row={peak[0]} col={peak[1]}")
        for axis in axes[row_number]:
            axis.axis("off")
        report.append({
            "index": index,
            "speed_kmh": row["speed_mps"] * 3.6,
            "prediction": prediction.tolist(),
            "actual": actual.tolist(),
            "occlusion_peak_grid": [int(value) for value in peak],
            "occlusion_grid": grid.tolist(),
        })
    figure.suptitle(
        "TCP-AC image-use diagnostics (red = stronger positive Grad-CAM / larger occlusion effect)")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"[saved] {args.output}")
    print(f"[saved] {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
