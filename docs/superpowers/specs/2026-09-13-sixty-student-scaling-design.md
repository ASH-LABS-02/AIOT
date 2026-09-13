# ClassSense AI — scaling to 60 students, and making the rest of the repo honest

Date: 2026-09-13
Status: approved, in implementation

## Problem

Three separate problems, discovered by reading the repo and measuring the machine.

**1. The live detector cannot handle a full classroom.** `scripts/live_detect.py` runs one
shared MediaPipe detector sequentially over every tracked person, inside the capture loop.
Display and analysis are the same thread, so the window freezes for the duration of
inference. The constants that gesture at a solution (`MP_POOL_SIZE`, `BATCH_SIZE`,
`MediaPipePool`, `ThreadPoolExecutor`) are declared but never wired into `main()`.

**2. The classifier is not connected to reality.** It was trained on raw normalised head-pose
values (`scaler.mean_` ≈ 0.011 / 0.518, `scale_` ≈ 0.95 / 1.21) but is fed calibrated
degrees at inference — values scaled by 70 and 60. Yaw and pitch carry 47% of feature
importance, so the model is effectively a constant predictor in live use.

**3. The training run reports numbers that are not true.** `train_model.py:189` writes
`(y_probs >= best_thresh).astype(int)` where `y_probs` is P(distracted); exceeding the
threshold means distracted but the expression emits `1`, which means *attentive*. Every
"tuned threshold" metric from that run is label-inverted — the two saved confusion matrices
are an exact transpose of each other, which is the signature. Separately the 60/20/20 split
is random over clips drawn from only 13 subjects, so the same person appears in train and
test.

## Measured constraints

Hardware: 20 CPU cores, **no GPU** (`torch 2.12.1+cpu`). Input: 1080p webcam.

From `scripts/benchmark.py` on this machine:

| Measurement | Result |
|---|---|
| MediaPipe `detect()` @ 96 / 128 / 192px crop | 7.13 / 7.21 / 6.71 ms — **flat** |
| MediaPipe `detect()` @ 256 / 320px crop | 14.67 / 14.46 ms — **doubles** |
| Parallel scaling, 8 threads | **3.21×**, 2.09 ms/face effective |
| Parallel scaling, 12 threads | 3.08×, 26% efficiency — past the knee |
| YOLOv8n @ 640 / 960 / 1280px | 36.1 / 64.1 / 107.6 ms |

Two findings drive the design:

- `detect()` releases the GIL, so a thread pool is genuinely parallel. Eight is the knee.
- Crop resolution above 192px costs double and buys nothing. The current code feeds raw
  crops, so a front-row student costs 2× what they should.

**Budget for 60 students:** 60 × 2.09 ms = 125 ms of landmark work, plus 108 ms for YOLO at
1280px, gives a ~233 ms cycle — every student re-examined ~4.3 times per second. The full
cohort fits in every cycle; round-robin batching is not required and is retained only as a
degradation path for slower hardware.

## Data reality

Only 13 subjects and 821 clips of DAiSEE were ever downloaded (741 `.avi` + 80 `.mp4`),
against 5,358 rows in `TrainLabels.csv`. `DataSet/` contains only `Train/` — there are no
Validation or Test videos, which is why `data/validation_features.csv` is empty.

Engagement is distributed `0:1, 1:31, 2:419, 3:370`. The current binary cut at `>= 2`
produces a 789/32 split — 3.9% minority, from 8 subjects. No resampling technique recovers a
usable classifier from that.

Cutting at `>= 3` instead produces **451/370**, which is nearly balanced and learnable. The
target changes meaning from "engaged vs disengaged" to **"fully engaged vs drifting"**, which
is the more useful signal for a classroom monitor anyway — the point is to surface students
losing focus, not to catch the rare student who has checked out entirely.

## Design

### Resolution tiers

1080p cannot give 60 faces equal quality. Rather than emit uniform-looking labels of
non-uniform trustworthiness, each student is assigned a tier from measured face width, and
the tier bounds which states they can be assigned.

| Tier | Face width | Analysis | States reachable |
|---|---|---|---|
| `FULL` | ≥ 64px | EAR, MAR, yaw/pitch/roll | all, including Sleepy |
| `COARSE` | 40–64px | pose only; EAR/MAR suppressed | Attentive, Distracted |
| `PRESENCE` | < 40px | YOLO box geometry only | Unknown |

The rationale for suppressing EAR below 64px: eye aspect ratio is computed from landmarks
~3px apart at that size, so a one-pixel jitter is a ~10% swing. Head pose uses landmarks
spanning the whole face and degrades gracefully. Reporting "Unknown" in gray is the honest
output for the back row, and the existing code already reserves that colour.

### Threading

Three stages, decoupled so the window never blocks:

- **Capture thread** — owns `VideoCapture`, keeps only the newest frame (drops stale ones so
  latency cannot accumulate).
- **Analysis thread** — YOLO → tracker association → parallel landmark extraction over the
  `MediaPipePool` (8 detectors) → state update. Runs at its own rate.
- **Main/render thread** — draws the newest frame with the most recent known states at full
  camera framerate.

Crops are resized to a 192px long side with aspect ratio preserved before entering
MediaPipe. Aspect preservation matters: EAR and MAR are ratios, scale-invariant only if both
axes scale together.

### Adaptive scheduling

When the student count exceeds what the cycle budget allows, students are prioritised rather
than cycled blindly. Priority rises with state uncertainty (mid-transition, eyes closing) and
falls with stability (solidly Attentive for 30s+). On this hardware at 60 students the budget
is not exceeded and every student is processed each cycle; the scheduler is the safety valve
for weaker machines or larger rooms.

### Module split

`live_detect.py` is 662 lines mixing capture, tracking, geometry, state logic and rendering.
Split into a `classsense/` package so each piece is testable alone:

| Module | Responsibility |
|---|---|
| `config.py` | All tunable constants. Single source of truth, shared with training. |
| `geometry.py` | EAR / MAR / head pose / IoU. Shared by extraction and live, which is what keeps train and serve in the same units. |
| `mp_pool.py` | Thread-safe MediaPipe detector pool. |
| `tiers.py` | Face-width → tier classification. |
| `tracker.py` | Identity association across frames. |
| `states.py` | Temporal state machine. |
| `render.py` | Overlay and dashboard. |
| `pipeline.py` | Thread orchestration. |

`geometry.py` being shared by both `extract_features.py` and the live path is the structural
fix for problem 2 — the skew existed because the two paths each had their own copy of
`head_pose_angles` and they drifted apart.

## Correctness fixes

- Feature skew: training and serving both use `geometry.py`, in the same units.
- Inverted labels: `train_model.py` threshold application corrected.
- Subject leakage: `GroupKFold` grouped by person ID, so no subject spans train and test.
- Label boundary moved to `>= 3`, documented as "fully engaged vs drifting".
- DAiSEE path mismatch between `check_paths.py` and `extract_features.py` resolved.
- Working directory: every script resolves paths from the repo root, not the caller's cwd.

## Verification

- Synthetic 60-cell tiled feed (one webcam frame replicated into a grid) to prove the
  throughput claim under real load rather than asserting it from the benchmark.
- Live small-group testing for state correctness.
- Unit tests for geometry (scale invariance), tier assignment, and tracker association.
- Retraining reports GroupKFold metrics, whatever they turn out to be.

## Explicitly out of scope

- GPU support. `torch` is a CPU build; installing CUDA is the user's call.
- Acquiring more DAiSEE data. The model is bounded by 13 subjects and this design does not
  pretend otherwise.
- Multi-camera input.
