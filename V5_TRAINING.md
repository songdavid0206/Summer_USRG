# TCP-AC v5

V5 is parallel to v4.  It does not load a v4 checkpoint and does not overwrite
`training/tcp_lagunaseca_v4`.

Key invariants:

- expert target speed is built only from actual speed in normal train trajectories;
- validation, recovery and DAgger rows never build that profile;
- `AI_RACING_LINE.csv` speed columns are not used;
- spline position is lookup metadata, never a model input;
- planner predicts six waypoints and six future speeds;
- current control consumes the predicted plan, never ground-truth plan tensors;
- visual correction is physically capped at steer `0.10`, throttle/brake `0.05`;
- planner warm-up must pass the target-speed causal gate before joint training;
- profile and normalization SHA-256 hashes must match at train/eval/live time.

Prepared-data audit (already generated once):

```powershell
python audit_v5.py --prepared training/tcp_lagunaseca_v5
```

Optional 100-batch CUDA smoke test:

```powershell
python train_v5.py --prepared training/tcp_lagunaseca_v5 --output training/tcp_lagunaseca_v5/smoke_checkpoints --epochs 1 --planner-warmup-epochs 1 --batch-size 32 --workers 16 --max-train-batches 100 --no-require-causal-probe --amp
```

Full training from ImageNet initialization (not v4):

```powershell
python train_v5.py --prepared training/tcp_lagunaseca_v5 --output training/tcp_lagunaseca_v5/checkpoints --epochs 50 --planner-warmup-epochs 5 --batch-size 32 --workers 16 --lr 1e-4 --weight-decay 1e-5 --imagenet-pretrained --balanced-sampler --amp --early-stopping-patience 8
```

Resume after an interruption (same output directory):

```powershell
python train_v5.py --prepared training/tcp_lagunaseca_v5 --output training/tcp_lagunaseca_v5/checkpoints --epochs 50 --planner-warmup-epochs 5 --batch-size 32 --workers 16 --lr 1e-4 --weight-decay 1e-5 --balanced-sampler --amp --early-stopping-patience 8 --resume training/tcp_lagunaseca_v5/checkpoints/last.pt
```

Offline evaluation and the full causal go/no-go gate:

```powershell
python evaluate_v5.py --prepared training/tcp_lagunaseca_v5 --checkpoint training/tcp_lagunaseca_v5/checkpoints/best.pt --output training/tcp_lagunaseca_v5/evaluation --workers 16
```

Only when `closed_loop_allowed` is true, use the normal `shadow.py` and
`closed_loop.py` commands with the v5 prepared/checkpoint paths. They detect v5
from the checkpoint schema and load the exact hashed v5 profile and normalization.
