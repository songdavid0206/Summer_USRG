# V6 closed-loop root-cause audit

Date: 2026-08-17

## Decision

Do not attribute the V6 closed-loop failure to the two-frame delay alone, and do
not start the full V7 training run until the corrected V6 live-input wiring has
been retested.  The two-frame alignment is a plausible phase compensation at
40 Hz, but it cannot by itself repair off-policy state drift or conflicting
expert policies.

## Confirmed correct

- Checkpoint, normalization, speed-profile and prepared-artifact hashes match.
- Offline and live ego axes, units, steering sign and vJoy scale match.
- Training and live image preprocessing differ numerically only negligibly at
  the model output (well below 0.002 control units in the sampled comparison).
- The two `AI_RACING_LINE.csv` byte hashes contain identical numeric geometry
  and columns.  Same-spline geometry error is exactly 0 m.
- Train/validation splitting is by whole trajectory/session with zero overlap.
- Future labels do not cross a lap or recovery-episode boundary.
- V7's two-frame control-label view changes only the intended control labels;
  its audit reports zero alignment error and zero split overlap.

## Confirmed live-input bug (fixed)

Training always supplied a six-wide command with both the fixed route bit and a
standing/flying lap-phase bit.  V5/V6 checkpoints omitted descriptive
`lap_phase` metadata, so live inference supplied only the route bit and created
an input combination never observed during training.

`ShadowPredictor` now enables phase bits for every schema-v5 checkpoint, and
future V5/V6/V7 checkpoints save the metadata explicitly.  Existing V6 weights
do not require retraining for this fix.  Sixteen focused tests and all 27 TCP
tests pass.  On saved failed-run observations, restoring the phase bit changes
steering by mean 0.037 and p95 0.109: material, but not sufficient to explain
all divergence by itself.

## External recovery/DAgger audit

Prepared V5 contains:

- normal: 155,541 eligible frames, 56 trajectories, 8 sessions;
- synthetic recovery: 21,092 eligible of 30,320 frames, 236 trajectories,
  5 sessions (9,228 forced-perturbation frames are masked);
- DAgger: 23,237 eligible frames, 148 trajectories, 3 sessions.

The data is not generally corrupt.  Held-out DAgger episodes usually finish
inside 1 m of the line, and session/episode boundaries and synchronization
flags are valid.  One known CSP-sidebar episode and one missing episode are
excluded.

However, it is not one homogeneous expert:

- normal: AC AI 98, DXCAM, approximately 38--40 image Hz and 100 telemetry Hz;
- synthetic recovery: AC AI 90, WGC, approximately 11--14 image Hz and
  27--36 telemetry Hz;
- DAgger: AC AI 85 or 90, WGC, approximately 21--40 image Hz and
  26--50 telemetry Hz.

Near-line matched-state comparisons still show larger policy differences for
DAgger (approximately 0.103 throttle and 0.041 brake absolute difference from
normal).  The balanced sampler forces synthetic recovery and DAgger to 15%
each per epoch, and repeats some rare buckets three or more times per epoch.
This is useful diversity for planning/recovery, but it gives one direct-control
head conflicting expert targets without an expert-policy identifier.

Held-out V6 error confirms the domain gap: DAgger total loss is about 0.783,
versus normal 0.250 and synthetic recovery 0.199; DAgger control loss is about
0.061 versus normal 0.014.

## Model-structure audit

V6 is no longer a minimal TCP baseline.  It is a single-frame ResNet/state
planner, a learned plan-to-control inverse model, and a capped image residual.
The GRU unrolls over prediction horizons within one frame; its hidden state is
reset on every live frame.  It therefore has no cross-frame memory, previous
action, or actuator state.  At racing speed this is much less tolerant of
latency and state drift than a low-speed urban baseline.

The visual residual and plan-control components are not independently
supervised; only their sum receives the current-control loss.  They can learn
cancellation.  This cancellation is useful on the AI distribution, but can
break when the model visits its own states.

The planner is strong on the AI shadow distribution, not proven perfect in
closed loop.  Shadow lateral waypoint errors are roughly 0.03--0.52 m across
0.1--2.0 s horizons, while forward position has a persistent roughly 0.6--0.9 m
bias.  Once the car leaves that distribution, neither low validation loss nor
high shadow correlation guarantees stable feedback.

## Delay conclusion

At 40 Hz, observed command-to-telemetry steering correlation peaks around two
frames, so a +50 ms direct-control target is reasonable.  Timing stages overlap;
adding capture latency and the measured two-frame telemetry lag as independent
serial delays double-counts part of the chain.  The two-frame shift can reduce
the initial phase error, but cannot explain or fix the radically different
behavior after the policy has entered unseen states.

## Required next sequence

1. Retest the existing V6 checkpoint at 40 Hz after the phase-input fix.  No
   retraining is needed.
2. Compare a clean AI handoff run with the prior clean handoff, especially the
   first corner's deviation, steering, throttle and brake.
3. Only if the corrected V6 still shows a repeatable phase-lag signature should
   V7's two-frame aligned-control training be considered.
4. If it still diverges materially, prefer a learned planner plus a
   speed-scheduled geometric steering controller and longitudinal PID.  Train
   the planner from all valid trajectories, but do not force heterogeneous
   AI85/90 recovery pedals into the same direct-control policy as normal AI98.
5. If pure direct-control IL remains mandatory, first make source policy and
   capture timing homogeneous (or explicitly condition the policy), then
   collect recovery from the actual deployed controller distribution.

## Corrected clean-handoff result

Run `20260817_221801` used the corrected phase input and handed control over
from AC AI at 106 km/h.  It disproves phase metadata and starting-grid offset
as the sole causes:

- live phase input was enabled;
- initial racing-line deviation was 1.46 m;
- from roughly 180 to 270 km/h, predicted steer/throttle and track-position
  speed closely followed the normal AI98 expert;
- failure began specifically at the braking transition.

At normalized spline position 0.1042, the live car was already at 276.8 km/h
while the normal expert median was 256.0 km/h.  The stored target profile was
256.9 km/h, so the profile itself was correct.  Nevertheless, the model emitted
throttle 0.913 and brake 0.205, while the expert emitted throttle 0.469 and
brake 0.646.  At spline 0.1153 the live car was 238.7 km/h versus the expert's
183.0 km/h, after which lateral deviation rapidly exceeded 10 m.

This establishes a late longitudinal transition as the initiating failure.
The model is accurate before the transition and produces strong braking only
after it is already overspeed.  V6's causal gate verifies relative response to
an artificial target change; it does not guarantee the correct absolute pedal
at the real braking boundary.

The expert sequence at spline 0.1042 changes substantially over the next two
40-Hz frames (median throttle 0.469 -> 0.332, brake 0.646 -> 0.748), so a
future-aligned target is directionally justified.  However, the onset trace and
the approximately 25 ms image-to-command pipeline suggest that a fixed
two-frame shift may under-compensate the full closed-loop transition by about
one frame.  It remains a hypothesis rather than a guaranteed complete fix.
