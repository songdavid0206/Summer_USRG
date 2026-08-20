"""Offline evaluation and mandatory full causal gate for V6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .profile_v5 import sha256_file
from .train_v5 import _tensor_stats, compute_losses
from .train_v6 import (
    RECIPE_VERSION, _stratified_probe_batch, causal_probe_v6, full_gate_passed,
)


def main(expected_recipe_version: int = RECIPE_VERSION,
         default_prepared: Path = Path("training/tcp_lagunaseca_v5"),
         default_checkpoint: Path = Path("training/tcp_lagunaseca_v6/checkpoints/best.pt"),
         default_output: Path = Path("training/tcp_lagunaseca_v6/evaluation"),
         metrics_filename: str = "metrics_v6.json", dataset_class=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, default=default_prepared)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from .data_v5 import TCPDatasetV5
    from .model_v6 import build_model_v6

    if dataset_class is None:
        dataset_class = TCPDatasetV5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("training_recipe", {}).get("version") != expected_recipe_version:
        raise RuntimeError(
            f"checkpoint is not a v{expected_recipe_version} causal-training checkpoint")
    schema = json.loads((args.prepared / "schema.json").read_text(encoding="utf-8"))
    if checkpoint.get("model_schema") != schema["model_schema"]:
        raise RuntimeError("checkpoint/prepared model-data contract mismatch")
    for name, expected in checkpoint["artifacts"].items():
        if sha256_file(args.prepared / name) != expected:
            raise RuntimeError(f"checkpoint artifact mismatch: {name}")
    normalization = json.loads(
        (args.prepared / "normalization.json").read_text(encoding="utf-8"))
    if checkpoint.get("normalization") != normalization:
        raise RuntimeError("checkpoint/prepared normalization mismatch")

    dataset = dataset_class(args.prepared, "val", augment=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    model = build_model_v6(False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    stats = _tensor_stats(normalization, device)
    sums, batches = {}, 0
    source_names = {0: "normal", 1: "synthetic_recovery", 2: "dagger"}
    source_sums, source_counts = {}, {}
    with torch.no_grad():
        for number, batch in enumerate(loader, 1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            pred = model(batch["image"], batch["ego"], batch["racing_line"],
                         batch["command"], batch["target_speed"])
            _, metrics = compute_losses(pred, batch, stats, planner_only=False)
            for name, value in metrics.items():
                sums[name] = sums.get(name, 0.0) + float(value)
            batches += 1
            for source_id, source_name in source_names.items():
                mask = batch["source_id"] == source_id
                count = int(mask.sum())
                if not count:
                    continue
                sub_pred = {key: value[mask] for key, value in pred.items()}
                sub_batch = {key: value[mask] for key, value in batch.items()}
                _, sub_metrics = compute_losses(sub_pred, sub_batch, stats, planner_only=False)
                source_counts[source_name] = source_counts.get(source_name, 0) + count
                target = source_sums.setdefault(source_name, {})
                for name, value in sub_metrics.items():
                    target[name] = target.get(name, 0.0) + float(value) * count
            if number == 1 or number % 100 == 0:
                print(json.dumps({"batches": number,
                                  "samples": min(number * args.batch_size, len(dataset)),
                                  "total": len(dataset), "device": str(device)}), flush=True)
            if args.max_batches and number >= args.max_batches:
                break

    metrics = {name: value / max(1, batches) for name, value in sums.items()}
    by_source = {
        source: {name: value / source_counts[source] for name, value in values.items()}
        for source, values in source_sums.items()
    }
    probe_batch = _stratified_probe_batch(dataset, device)
    recipe = checkpoint["training_recipe"]
    probe = causal_probe_v6(model, probe_batch, stats, recipe["causal_drop_mps"])
    passed = full_gate_passed(
        probe, recipe["causal_min_speed_drop_mps"],
        recipe["causal_min_longitudinal_delta"],
        recipe["anticipation_min_speed_drop_mps"],
        recipe["anticipation_min_longitudinal_delta"])
    result = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "metrics": metrics, "by_source": by_source,
        "causal_probe": probe, "causal_probe_passed": passed,
        "closed_loop_allowed": passed,
        "note": "closed loop is blocked unless the full planner-to-pedal causal gate passes",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / metrics_filename).write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
