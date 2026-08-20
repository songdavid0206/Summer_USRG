"""Convert legacy TCP JSON histories into comparable TensorBoard runs.

The converter never edits checkpoints or source histories.  Each selected
history is written as a separate TensorBoard run with common scalar tag names,
so versions can be enabled and disabled from TensorBoard's run selector.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterator


DEFAULT_RUNS = {
    "v1_initial": "training/tcp_lagunaseca/checkpoints/history.json",
    "v1_phase": "training/tcp_lagunaseca_phase/checkpoints/history.json",
    "v2_smoke_only": "training/tcp_lagunaseca_v2_exact/checkpoints_smoke/history.json",
    "v3_dense": "training/tcp_lagunaseca_v3_dense/checkpoints/history.json",
    "v4": "training/tcp_lagunaseca_v4/checkpoints/history.json",
    "v5": "training/tcp_lagunaseca_v5/checkpoints/history.json",
    "v6": "training/tcp_lagunaseca_v6/checkpoints/history.json",
    "v8": "training/tcp_lagunaseca_v8/checkpoints/history.json",
}


def _numeric_items(prefix: str, value: object) -> Iterator[tuple[str, float]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            yield from _numeric_items(child_prefix, child)
    elif isinstance(value, bool):
        yield prefix, float(value)
    elif isinstance(value, (int, float)):
        yield prefix, float(value)


def _tag(path: str) -> str | None:
    """Map history paths onto stable TensorBoard tag names."""
    if path.startswith("train/"):
        return "loss/train_" + path.removeprefix("train/").replace("/", "_")
    if path.startswith("val/"):
        return "loss/val_" + path.removeprefix("val/").replace("/", "_")
    if path == "lr":
        return "optimization/learning_rate"
    if path == "amp_overflow_steps":
        return "optimization/amp_overflow_steps"
    if path == "best_val":
        return "optimization/best_val_total"
    if path == "is_best":
        return "optimization/is_best"
    if path.startswith("val_by_phase/"):
        return "breakdown/" + path.replace("/", "_")
    if path.startswith("val_by_source/"):
        return "breakdown/" + path.replace("/", "_")
    if path.startswith("causal_probe/"):
        return "diagnostics/" + path.replace("/", "_")
    if path in {"elapsed_s"}:
        return "optimization/" + path
    return None


def _load_history(path: Path) -> list[dict]:
    history = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        raise ValueError(f"history is empty or invalid: {path}")
    return history


def _plot_overview(histories: dict[str, list[dict]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(histories)
    columns = 2
    rows = (len(names) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(14, 3.8 * rows))
    axes = axes.reshape(-1)
    for axis, name in zip(axes, names):
        history = histories[name]
        epochs = [int(row["epoch"]) for row in history]
        train = [float(row["train"]["total"]) for row in history]
        val = [float(row["val"]["total"]) for row in history]
        best_index = min(range(len(val)), key=val.__getitem__)
        axis.plot(epochs, train, label="train_total", linewidth=1.5)
        axis.plot(epochs, val, label="val_total", linewidth=1.5)
        axis.scatter([epochs[best_index]], [val[best_index]], s=35, zorder=3,
                     label=f"best val e{epochs[best_index]}")
        axis.set(title=name, xlabel="epoch", ylabel="version-specific total loss")
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    for axis in axes[len(names):]:
        axis.set_visible(False)
    figure.suptitle(
        "TCP-AC training histories (loss formulas differ across versions)", fontsize=14)
    figure.tight_layout()
    figure.savefig(output / "all_versions_total_loss.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path,
        default=Path("training/tensorboard_history_compare"))
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError("TensorBoard support is required") from error

    root = args.root.resolve()
    output = (root / args.output).resolve()
    if output.exists() and not args.keep_existing:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    histories: dict[str, list[dict]] = {}
    manifest: dict[str, object] = {
        "note": "Total-loss formulas differ across versions; compare trends, not raw magnitudes.",
        "runs": {},
    }
    for run_name, relative in DEFAULT_RUNS.items():
        source = root / relative
        if not source.exists():
            continue
        history = _load_history(source)
        histories[run_name] = history
        writer = SummaryWriter(log_dir=str(output / run_name))
        tags: set[str] = set()
        for row in history:
            step = int(row["epoch"])
            for path, scalar in _numeric_items("", row):
                tag = _tag(path)
                if tag is None:
                    continue
                writer.add_scalar(tag, scalar, step)
                tags.add(tag)
        writer.flush()
        writer.close()
        val = [float(row["val"]["total"]) for row in history]
        best_index = min(range(len(val)), key=val.__getitem__)
        manifest["runs"][run_name] = {
            "source": str(source),
            "epochs": len(history),
            "first_epoch": int(history[0]["epoch"]),
            "last_epoch": int(history[-1]["epoch"]),
            "best_val_epoch": int(history[best_index]["epoch"]),
            "best_val_total": val[best_index],
            "tags": sorted(tags),
            "smoke_only": len(history) == 1,
        }

    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    _plot_overview(histories, output)
    print(json.dumps(manifest, indent=2))
    print(f"[saved] {output / 'all_versions_total_loss.png'}")
    print(f"[tensorboard] tensorboard --logdir {output} --port 6007")


if __name__ == "__main__":
    main()
