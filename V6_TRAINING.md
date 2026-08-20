# TCP-AC V6

V6 keeps the validated V5 dataset, target-speed profile, normalization, model
parameter layout, predicted-plan control path, and hard-capped visual residual.
It changes the training recipe and its safety gates.

## Why V6 exists

V5 learned an observational shortcut: low target speeds commonly occur near a
corner exit where expert throttle rises. Under a counterfactual 10 m/s target
reduction, its planner predicted a lower future speed but its control head
increased throttle and reduced brake. Ordinary L1 validation loss did not expose
that inverse causal response.

## V6 changes

- Each joint-training batch includes a same-observation/lower-target pair.
- A second pair leaves 0/5/10m targets unchanged and lowers only 20--150m
  targets, directly testing and teaching braking anticipation before the
  current-position target falls.
- The pair loss requires future speed to decrease, forbids throttle/brake from
  moving in the wrong direction, and requires net `throttle - brake` authority
  to fall by at least 0.10 for a 10 m/s target reduction. This is only a
  training/gating metric; throttle and brake remain independent model outputs.
- A fixed 96-sample validation probe contains 32 normal, 32 synthetic recovery,
  and 32 DAgger samples.
- Overall and every source must pass direction, magnitude, and 75% coverage gates.
- `best.pt` is written only for a full-gate-passing epoch.
- A failure by joint epoch 5 stops training instead of wasting 50 epochs.
- V6 checkpoints that did not pass the full gate are blocked from shadow and
  closed-loop loading.

The V5 planner warm-up is reusable because the data/model contract is unchanged.
The optimizer is intentionally restarted for causal joint training.
