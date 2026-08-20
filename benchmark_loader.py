"""Short end-to-end CUDA throughput benchmark for TCP DataLoader workers."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import threading
import time


def memory_status() -> dict[str, float]:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return {}
    gib = 1024 ** 3
    return {"total_gib": status.ullTotalPhys / gib,
            "available_gib": status.ullAvailPhys / gib,
            "load_percent": float(status.dwMemoryLoad)}


class GPUMonitor:
    def __init__(self):
        self.values: list[float] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.wait(.5):
            try:
                raw = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits"], text=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self.values.append(float(raw.splitlines()[0].strip()))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass

    def start(self) -> None:
        self.thread.start()

    def finish(self) -> dict[str, float | int]:
        import numpy as np
        self.stop_event.set()
        self.thread.join(timeout=2)
        return ({"samples": len(self.values), "mean": float(np.mean(self.values)),
                 "p95": float(np.percentile(self.values, 95))}
                if self.values else {"samples": 0, "mean": 0.0, "p95": 0.0})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path,
                        default=Path("training/tcp_lagunaseca_v4"))
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--batches", type=int, default=300)
    parser.add_argument("--warmup-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from ac_dataset.tcp.config import target_speed_input_width
    from ac_dataset.tcp.data import TCPDataset
    from ac_dataset.tcp.model import build_model
    from ac_dataset.tcp.train import losses, seed_everything

    seed_everything(42)
    schema = json.loads((args.prepared / "schema.json").read_text(encoding="utf-8"))
    width = target_speed_input_width(schema["model_schema"])
    dataset = TCPDataset(args.prepared, "train", augment=args.augment)
    options = ({"persistent_workers": True, "prefetch_factor": 2}
               if args.workers else {})
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True, **options)
    device = torch.device("cuda")
    model = build_model(False, pred_len=len(dataset.offsets),
                        target_speed_width=width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", init_scale=32768.0)
    iterator = iter(loader)
    monitor = GPUMonitor()
    memory_before = memory_status()
    elapsed = 0.0
    measured = overflow = 0
    monitor.start()
    for number in range(1, args.warmup_batches + args.batches + 1):
        batch = next(iterator)
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        if number == args.warmup_batches + 1:
            torch.cuda.synchronize()
            started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            prediction = model(
                batch["image"], batch["ego"], batch["racing_line"], batch["command"],
                batch["target_speed"] if width else None)
            loss, _ = losses(prediction, batch, speed_weight=0.0 if width else .05)
        before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        overflow += int(scaler.get_scale() < before)
        if number > args.warmup_batches:
            measured += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "workers": args.workers, "augment": args.augment,
        "batches": measured, "batch_size": args.batch_size,
        "elapsed_s": elapsed, "batches_per_s": measured / elapsed,
        "samples_per_s": measured * args.batch_size / elapsed,
        "seconds_per_batch": elapsed / measured, "amp_overflow_steps": overflow,
        "gpu0_utilization": monitor.finish(),
        "memory_before": memory_before, "memory_after": memory_status(),
        "output_shape": list(batch["image"].shape),
        "finite_loss": bool(torch.isfinite(loss).item()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
