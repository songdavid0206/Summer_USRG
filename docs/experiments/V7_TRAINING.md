# TCP-AC V7: 50 ms delay-aligned control supervision

V7 keeps the V5/V6 image, ego, racing-line, target-speed, waypoint, future-speed,
model architecture, sampler, causal loss, and train/validation trajectory split.
Only the expert control labels move from time `t` to `t + 50 ms` (two 40 Hz
frames), matching the measured observation-to-AC-response lag.

## Prepare and audit

```powershell
python scripts/experiments/prepare_v7.py --source-prepared training/tcp_lagunaseca_v5 --output training/tcp_lagunaseca_v7 --delay-frames 2
python scripts/experiments/audit_v7.py --prepared training/tcp_lagunaseca_v7
```

The audit must report `passed: true`, zero train/val overlap, and zero control
alignment error.

## Train

The V5 planner warm-up is reusable because V7 does not change planner targets.
Its control heads were not supervised during warm-up.

```powershell
python scripts/experiments/train_v7.py --prepared training/tcp_lagunaseca_v7 --output training/tcp_lagunaseca_v7/checkpoints --epochs 40 --planner-warmup-epochs 0 --init-planner training/tcp_lagunaseca_v5/checkpoints/planner_warmup.pt --batch-size 32 --workers 16 --lr 1e-4 --weight-decay 1e-5 --imagenet-pretrained --balanced-sampler --amp --early-stopping-patience 8
```

V7 still requires the complete causal gate before `best.pt` is saved. Because
future control is intrinsically harder than same-time control, compare V7
validation metrics to its own epoch history rather than expecting V6's exact
control MAE.

## Scope

This change targets the normal-state phase lag that seeds closed-loop drift. It
does not claim to solve every off-distribution recovery case. No new DAgger data
is required or added by V7.
