"""Save a visual preview of the TCP train-only image augmentation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    table = pd.read_parquet(args.prepared / "manifest.parquet")
    row = table[table["eligible"]].iloc[len(table[table["eligible"]]) // 2]
    source = Image.open(row.image_path).convert("RGB")
    resized = source.resize((512, 256), Image.Resampling.BILINEAR)

    rng = np.random.default_rng(args.seed)
    panels: list[tuple[str, Image.Image]] = [("resized / no augmentation", resized)]
    for number in range(1, 4):
        brightness = float(rng.uniform(.82, 1.18))
        contrast = float(rng.uniform(.86, 1.14))
        color = float(rng.uniform(.88, 1.12))
        image = ImageEnhance.Brightness(resized).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Color(image).enhance(color)
        label = f"aug {number}: B={brightness:.2f} C={contrast:.2f} S={color:.2f}"
        panels.append((label, image))

    canvas = Image.new("RGB", (1024, 576), "#202020")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        x = (index % 2) * 512
        y = (index // 2) * 288
        canvas.paste(image, (x, y + 32))
        draw.text((x + 8, y + 8), label, fill="white")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"[saved] {args.output}")
    print(f"[source] {row.image_path}")


if __name__ == "__main__":
    main()
