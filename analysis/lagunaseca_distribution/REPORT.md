# Laguna Seca dataset distribution audit

## Scope

- Solo: 11 selected sessions, 70 complete laps, 195,147 frames
- Multi: 12 selected complete sessions, 84 complete laps, 245,197 frames
- Total: 440,344 frames at 40 Hz, about 183.5 minutes
- Excluded by selection: ego-retired runs and the earlier multi trial run
- Unit of redundancy check: per-run speed profile over 100 normalized track bins

The audit reads the structured session index and synchronized ego labels. For
sessions with the raw parquet still present, it uses the corresponding exported
`source_index`; early solo exports without raw parquet are read from `lap_*/EGO_INFO`.
Schema-v2 root-index rows outside retained complete laps are not included.

## Main findings

### 1. Normal racing coverage is already sufficient

- Every 1% track-progress bin has at least 1,249 solo frames and 1,569 multi frames.
- Solo and multi cover the full racing line and all braking zones.
- Median between-run speed-profile correlation is 0.991 for solo and 0.987 for multi.

Additional ordinary AI laps on the same track/settings would mostly add redundant
samples. Do not collect more bulk laps before training the first baseline.

### 2. Solo appearance coverage is weather-imbalanced

| Mode | Clear | Scattered clouds | Mist |
|---|---:|---:|---:|
| Solo | 65.2 min / 9 runs | 8.1 min / 1 run | 8.1 min / 1 run |
| Multi | 42.9 min / 5 runs | 25.8 min / 3 runs | 33.5 min / 4 runs |

Multi is reasonably diverse, but solo is dominated by clear weather. This does
not block the first baseline. Use run/weather-balanced sampling or augmentation
before collecting more ordinary solo laps.

### 3. Control coverage is broad, but starts are rare

Solo and multi control distributions are very similar. Approximately half the
frames have `|steer| < 0.05`, about 10.5-10.9% have `|steer| >= 0.5`, and braking
is active in about 20.5% of solo and 21.8% of multi frames.

Only about 0.7-0.9% of frames are below 60 km/h and about 0.1-0.2% are below
20 km/h. The collector intentionally retains very little stationary pre-motion
data, so standing-start behavior is a targeted weakness for a pure learned
controller.

### 4. Multi adds useful lateral and interaction variation

- `|offset from AI line| > 2 m`: solo 6.0%, multi 15.5%
- `|offset from AI line| > 4 m`: solo 1.36%, multi 3.41%
- Valid opponent rows within 20 m: 78,916 / 1,128,355 (about 7.0%)
- Unique interaction-union frames: 63,228 / 245,197 multi frames (25.8%)

The multi set is not merely a visual duplicate of solo. It adds meaningful line
variation and close-agent exposure.

### 5. Interaction classes are imbalanced

| Interaction | Instances |
|---|---:|
| close_follow | 178 |
| side_by_side | 66 |
| overtake | 42 |
| car_contact | 38 |
| overtaken_by | 6 |
| overtake_attempt | 3 |
| overtaken_by_attempt | 1 |

Successful overtaking has enough examples for an initial weighted-sampling
experiment. Being overtaken and failed-attempt behavior are too sparse for a
standalone class or strong conclusion. Contact clips must not be treated as
desirable imitation targets; they are useful as risk/hard-negative labels.

`barrier_contact` clip counts are not uniform across the historical export
schemas, so they are not used here to judge barrier-collision coverage.

### 6. Capture gaps are not a dataset-level blocker

There are 424 reported missing 40 Hz grid slots across the selected sessions
(about 0.10% relative to retained frames). Most are in older solo exports; the
largest selected runs have 107 solo gaps and 49 multi gaps. A temporal loader
must break sequences across non-25 ms time gaps rather than padding through
them, but individual frames remain usable.

## Collection decision

Freeze the current dataset and train the first single-frame baseline now.
More ordinary Laguna Seca laps are not justified by the current distributions.

Collect additional data only after the first closed-loop failure analysis, and
make it targeted:

1. standing-start/low-speed sequences if the learned car cannot launch reliably;
2. recovery demonstrations if it drifts away from the racing line and cannot
   return;
3. `overtaken_by` and failed-overtake scenarios if those behaviors are part of
   the interaction-aware experiment;
4. one or two extra non-clear solo runs only if weather-specific validation
   exposes a visual generalization gap.

Do not delete redundant runs yet. First split by run, train the baseline, and
use track-bin/run-balanced sampling. Physical deletion can be considered only
after confirming that downsampling does not hurt closed-loop completion.

## Plot guide

The plots in this directory analyze synchronized numeric labels attached to the
camera frames. They do not yet analyze RGB pixel content or image embeddings.

- `01_control_distributions.png`: overall speed/steer/throttle/brake histograms.
- `02_track_profiles.png`: solo and multi average speed/brake versus track progress.
- `03_spatial_coverage.png`: all-frame spatial density and overall lateral-offset
  histogram. This is an aggregate view, not a run comparison.
- `04_balance_and_interactions.png`: weather duration and interaction-instance counts.
- `05_opponent_relative_positions.png`: all valid opponent positions in the ego
  coordinate frame. It includes valid non-visible opponents and is not an image
  visibility heatmap.
- `06_run_redundancy.png`: correlation between run speed profiles; yellow means
  two runs have very similar speed patterns.
- `07_run_speed_heatmap.png`: requested run-by-run speed heatmap. Every row is one
  run and every column is 1% of track progress.
- `08_run_lateral_offset_heatmap.png`: requested run-by-run signed offset from the
  AI racing line. White is near the line; red and blue are opposite sides.
- `09_run_trajectory_overlays.png`: black dashed AI racing line with every
  run's actual ego trajectory colored by absolute lateral error.
- `10_run_lateral_offset_summary.png`: mean, 95th-percentile, and 99th-percentile
  absolute AI-line deviation for each run.

## Run-by-run viewer

Open `run_viewer.html` to inspect one run at a time. It contains the data
directly, so no web server or package installation is required.

- Use the dropdown or the left/right arrow keys to change runs.
- Filter to solo or multi with the mode selector.
- Keep `전체 run 요약` for the 100-bin aggregate, or select a lap to inspect
  real frame samples from that lap at 0.5% track-progress intervals.
- The left track map colors the actual ego trajectory by mean speed.
- The right track map colors it by absolute AI-line error. It can switch between
  mean and 95th-percentile error in the run aggregate. A selected lap instead
  shows the sampled frames' actual speed and actual absolute error.
- The two lower charts show the same values over normalized track progress.

To rebuild both the analysis and viewer from the project directory:

```powershell
python tools\analyze_lagunaseca_distribution.py
python tools\build_lagunaseca_viewer.py
start analysis\lagunaseca_distribution\run_viewer.html
```

If the analysis is already current, only the last command is needed to open it.
