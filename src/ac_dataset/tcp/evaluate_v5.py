"""Offline physical-unit evaluation and causal go/no-go report for v5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .profile_v5 import sha256_file
from .train_v5 import _tensor_stats, causal_probe, compute_losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, default=Path("training/tcp_lagunaseca_v5"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("training/tcp_lagunaseca_v5/checkpoints/best.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("training/tcp_lagunaseca_v5/evaluation"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from .data_v5 import TCPDatasetV5
    from .model_v5 import build_model_v5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    schema = json.loads((args.prepared / "schema.json").read_text(encoding="utf-8"))
    if checkpoint.get("model_schema") != schema["model_schema"]:
        raise RuntimeError("checkpoint/prepared v5 schema mismatch")
    for name, expected in checkpoint["artifacts"].items():
        if sha256_file(args.prepared / name) != expected:
            raise RuntimeError(f"checkpoint artifact mismatch: {name}")
    dataset = TCPDatasetV5(args.prepared, "val", augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0)
    model = build_model_v5(False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    stats = _tensor_stats(checkpoint["normalization"], device)
    sums, batches, source_sums, source_counts, probe_batch = {}, 0, {}, {}, None
    source_names = {0: "normal", 1: "synthetic_recovery", 2: "dagger"}
    with torch.no_grad():
        for number, batch in enumerate(loader, 1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            if probe_batch is None:
                probe_batch = {key: value.detach().clone() for key, value in batch.items()}
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
                print(json.dumps({"batches": number, "samples": min(number * args.batch_size,
                                  len(dataset)), "total": len(dataset), "device": str(device)}),
                      flush=True)
            if args.max_batches and number >= args.max_batches:
                break
    metrics = {name: value / max(1, batches) for name, value in sums.items()}
    by_source = {source: {name: value / source_counts[source]
                         for name, value in values.items()}
                 for source, values in source_sums.items()}
    probe = causal_probe(model, probe_batch, stats)
    causal_pass = (
        probe["future_speed_delta_mps"] <= -.50 and
        probe["future_speed_monotonic_fraction"] >= .75 and
        probe["plan_throttle_delta"] <= 0 and probe["plan_brake_delta"] >= 0 and
        probe["final_throttle_delta"] <= 0 and probe["final_brake_delta"] >= 0 and
        probe["residual_change_abs_max"] <= 1e-6
    )
    result = {
        "checkpoint_epoch": int(checkpoint["epoch"]), "metrics": metrics,
        "by_source": by_source, "causal_probe": probe,
        "causal_probe_passed": causal_pass,
        "closed_loop_allowed": causal_pass,
        "note": "causal probe is a sensitivity/monotonicity gate, not a labeled counterfactual",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics_v5.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
