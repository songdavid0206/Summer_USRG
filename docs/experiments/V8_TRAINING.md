# V8 single-session convergence experiment

## Frozen cohort

- Session: `20260727_153344_lagunaseca_ferrarisf15t_clear_solo_40hz_stollrevised`
- Train: laps 1--5 (13,918 samples)
- Validation: laps 6--7 (5,503 samples)
- Quality exclusions: three rows (two stale camera-pose metadata rows at lap
  boundaries and one repeated image/state row)
- Standing lap: lap 1, train only
- Recovery/DAgger: not included
- Train/validation trajectory overlap: zero
- One shared model and one joint loss are used for both phases.  The phase bit
  remains an input so the standing launch is not assigned the flying-lap target.

## Model and optimization

- V5 plan-first ResNet34 model: 22,222,412 trainable parameters
- Uniform shuffled train observations; no balanced recovery sampler
- ImageNet initialization, train-only photometric augmentation, AMP
- Joint waypoint, future-speed, current-control, future-control and physical
  consistency losses from epoch 1
- Maximum 1,000 epochs; no early stopping by default
- ReduceLROnPlateau: factor 0.5 after 25 unimproved validation epochs, minimum
  learning rate 1e-6
- `best.pt` always contains the lowest validation loss. `last.pt` is overwritten
  each epoch. Immutable periodic snapshots are disabled by default to preserve
  disk space.

## Commands

```powershell
python scripts/experiments/audit_v8.py --prepared training/tcp_lagunaseca_v8

python scripts/experiments/train_v8.py --prepared training/tcp_lagunaseca_v8 --output training/tcp_lagunaseca_v8/checkpoints --tensorboard-dir training/tcp_lagunaseca_v8/tensorboard --epochs 1000 --batch-size 32 --workers 16 --lr 1e-4 --weight-decay 1e-5 --imagenet-pretrained --amp --early-stopping-patience 0 --checkpoint-every 0
```

In a second PowerShell window:

```powershell
tensorboard --logdir training/tcp_lagunaseca_v8/tensorboard --port 6006
```

Open `http://localhost:6006`.  The important curves are:

- `loss/train_total` versus `loss/val_total`;
- `loss/val_control`, including steer/throttle/brake MAE;
- `loss/val_waypoint_mae_m`;
- `loss/val_future_speed_mae_mps`;
- `optimization/learning_rate`.

Do not select epoch 1000 merely because it is last.  Use `best.pt`, and stop
manually if validation has clearly worsened for a long interval while training
loss continues falling.
