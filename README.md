# ClassSense AI

Real-time classroom engagement monitoring. Detects people with YOLOv8, reads facial
landmarks with MediaPipe, and classifies each student as **Attentive**, **Sleepy**,
**Distracted**, or **Unknown** using temporal rules over eye, mouth and head-pose signals.

Measured at **60 students in a single 1080p frame at 6.0 analysis cycles per second on
CPU** — every student re-examined roughly six times a second, no GPU.

Deploying to a Raspberry Pi? **Read [deploy/README-raspberry-pi.md](deploy/README-raspberry-pi.md)
first, and run `scripts/calibrate.py` before anything else.** Every figure below was
measured on a 20-core desktop; inheriting those constants on four slower cores does not
produce a slow system, it produces a confidently wrong one.

---

## Capacity — how many students this machine may watch

The student cap is a measured consequence, not a setting. It follows from one
requirement: **every temporal gate must be sampled several times inside its own window.**
The shortest gate is `DISTRACTED_DURATION` at 0.8s and the default asks for 3 samples, so
the refresh must stay under 0.27s.

This matters because the failure is silent. A student examined every two seconds can sleep
through a lesson reading "Attentive" — the pipeline still runs, still draws boxes, still
prints an engagement percentage. Nothing looks broken.

Two ceilings apply, and capacity is the lower of them:

- **Compute** — how many faces fit inside the refresh budget. Binds on a Pi.
- **Resolution** — how many faces the camera can resolve above the tier floor. Binds on a
  wide 1080p shot; the model independently puts this near 62, which is where the measured
  60-student design point sits.

```bash
python scripts/calibrate.py
```

Measures this machine and writes `classsense/tuning.json` — gitignored, because it
describes one machine. Without it the pipeline falls back to an estimate from the core
count and says so on every start.

Students beyond capacity are tracked and counted but reported **`Unmonitored`** rather than
given a state the sampling rate cannot support. `--allow-over-capacity` rotates everyone
through at reduced fidelity instead.

---

## Quick start

```bash
python -m venv classsense-env
classsense-env\Scripts\activate
pip install -r requirements.txt
```

```bash
python scripts/live_detect.py
```

Keys: `Q` quit · `S` snapshot · `D` telemetry HUD.

Headless, with a web view and JSON status — for a Pi, or any machine with no display:

```bash
python scripts/live_detect.py --headless --serve 8080
```

`/` live view and tallies · `/stream` MJPEG · `/status` JSON · `/snapshot` one JPEG.
There is no authentication and it streams a live camera feed of a room, so keep it on a
trusted network.

Every script runs from the repo root and resolves its own paths, so none of them care
which directory you launch from.

---

## What the states mean

| State | Colour | Trigger |
|---|---|---|
| **Attentive** | green | eyes open, posture forward. Blinks do not disturb it |
| **Sleepy** | orange | eyes closed ≥ 1.2s, yawn ≥ 1.5s, or head nodding with eyes closing |
| **Distracted** | red | head turned/reclined/tilted ≥ 0.8s, or face lost ≥ 0.5s |
| **Unknown** | gray | face too small to read reliably — see tiers below |
| **Unmonitored** | gray | present, but beyond this machine's measured capacity |

Every trigger is **wall-clock gated**, not frame-counted. A blink and a microsleep look
identical in one frame and differ only in duration, so the system waits before committing.
This is why the labels do not strobe.

---

## Resolution tiers — why some students read "Unknown"

A 1080p frame cannot give 60 faces equal quality. The front row may land 120px of face
width and the back row 40px. Rather than print identical-looking labels of wildly
different trustworthiness, each student is tiered by **measured face width** and the tier
bounds what may be concluded:

| Tier | Face size | Analysis | States reachable |
|---|---|---|---|
| `FULL` | ≥ 64px | eyes, mouth, head pose | all, including Sleepy |
| `COARSE` | 40–64px | head pose only | Attentive, Distracted |
| `PRESENCE` | < 40px | presence only | Unknown |

Eye aspect ratio is suppressed below `FULL` because at a 50px face the eye landmarks sit
about 3px apart — one pixel of jitter moves EAR by ~10%, more than the gap between an open
and a closed eye. Head pose spans the whole face and degrades gracefully, so it survives
one tier lower.

"Face size" is deliberately not cheek-to-cheek width. Width projects as cos(yaw), so a
student turning 40° loses about a quarter of their apparent width and drops a tier — losing
Sleepy detection at the moment they are most worth watching. Face *height* does not
foreshorten with yaw, so `face_size_px` takes the larger of the width and the height scaled
by the measured 0.887 width-to-height ratio. A frontal face uses whichever is cleaner; a
turned face keeps its tier.

The dashboard computes class engagement over **readable students only**. Counting the
unreadable back row as disengaged would make the percentage a measure of camera placement
rather than of the class.

**If too many students read Unknown, move the camera closer or raise its resolution.**
That is a physical problem and no amount of tuning fixes it.

---

## Measured performance

`scripts/benchmark.py` profiles the machine; `scripts/stress_test.py` runs the real
pipeline against a synthetic tiled classroom.

On 20 CPU cores, no GPU, `torch 2.12.1+cpu`:

| Students | Mean cycle | Median | p95 | Cycles/s | Refresh |
|---|---|---|---|---|---|
| 10 | 59 ms | 32 ms | 131 ms | 16.9 | 0.06 s |
| 20 | 82 ms | 49 ms | 157 ms | 12.3 | 0.08 s |
| 30 | 101 ms | 70 ms | 172 ms | 9.9 | 0.10 s |
| 45 | 127 ms | 102 ms | 213 ms | 7.9 | 0.13 s |
| **60** | **168 ms** | **143 ms** | **284 ms** | **6.0** | **0.17 s** |

"Refresh" is the gap between successive looks at any one student. At 0.17s a 1.2s eye
closure gets roughly seven samples, so Sleepy detection is comfortable at full capacity.

Mean and median differ because person detection runs only every third cycle — cheap
landmark-only cycles sit between costly detect ones. The **mean** is the honest figure for
sustained throughput; a median would land on a cheap cycle and flatter the result.

Four findings shaped the design:

- **MediaPipe `detect()` releases the GIL** — 3.21× across 8 threads. A thread pool is
  genuinely parallel, which is what makes 60 students affordable without a GPU.
- **Crop resolution above 192px costs double and buys nothing** — 7.13 / 7.21 / 6.71 ms at
  96 / 128 / 192px, then 14.67 ms at 256px. Crops are capped at 192px, aspect preserved.
- **Batching the classifier matters enormously** — 60 separate `predict_proba` calls on a
  300-tree forest measure 3.7s; the same 60 rows batched measure 56ms.
- **YOLO was 47% of the cycle** re-finding people who had not moved (124ms of 263ms).
  Person detection now runs every `DETECT_EVERY` cycles while landmark analysis runs
  continuously, since eyes, mouth and head angle are what actually change. This alone took
  60 students from 269ms to 168ms. The cost is admission latency: a student entering the
  room is picked up within about half a second.

Run them yourself:

```bash
python scripts/benchmark.py
```

```bash
python scripts/stress_test.py --cells 60 --cycles 20
```

---

## The classifier: what it is actually worth

**It is off by default, on purpose.**

Evaluated with `GroupKFold` over subject id — no person appears in both train and test:

| | |
|---|---|
| Out-of-fold accuracy | **0.559** |
| Majority-class baseline | 0.549 |
| ROC AUC | **0.591** |
| Per-fold accuracy | 0.473 – 0.611 |
| Subjects | 13 |

A **1-point gain over always guessing**, and on some held-out people it is worse than a
coin. `train_model.py` marks such a model `advisory_only` in `models/model_meta.json`, and
`live_detect.py` reads that flag, prints the score, and refuses to let it change any
student's state. Pass `--use-weak-model` to override.

This is not caution for its own sake. With the model enabled on a test frame the
heuristics read as calm, it labels 57 of 60 students "Distracted". A monitor that cries
wolf costs the operator their trust in the colours the heuristics get right.

**The limit is the data, not the algorithm.** Only 13 of DAiSEE's ~70 subjects were
downloaded (821 of 5,358 clips), and no Validation or Test videos at all. Thirteen subjects
cannot support a model that generalises to new faces. Run `python scripts/check_paths.py`
to see exactly what is on disk. Downloading more of DAiSEE is the only thing that moves
these numbers.

The label boundary is `engagement >= 3`, giving a balanced 451/370 split. The obvious
boundary, `>= 2`, yields 789/32 — 3.9% minority from 8 subjects, which is not learnable.
The target therefore means **"fully engaged" vs "drifting"**, not "engaged" vs "disengaged".

---

## Layout

```
classsense/            the package — importable, testable, no entry points
  config.py            every tunable value; shared by training and inference
  geometry.py          EAR / MAR / head pose / IoU
  mp_pool.py           thread-safe MediaPipe pool + crop preparation
  tiers.py             face width -> tier -> permitted states
  tracker.py           identity association across frames
  states.py            the temporal state machine
  render.py            overlay and dashboard
  pipeline.py          capture / analysis / render threading
  capacity.py          what this machine may honestly watch
  detector.py          backend selection + NCNN width guard
  server.py            MJPEG + JSON output, standard library only

scripts/
  live_detect.py       live detection (entry point)
  benchmark.py         profile this machine
  stress_test.py       prove throughput at N students
  extract_features.py  DAiSEE clips -> features
  train_model.py       train + subject-independent evaluation
  check_paths.py       what dataset is actually on disk
  calibrate.py         measure THIS machine, write its tuning
  export_ncnn.py       export YOLO to NCNN for ARM
  diagnose_duplicates.py  why one person is being counted as more than one

tests/                 134 tests
```

`geometry.py` is imported by **both** the training extractor and the live pipeline. That
sharing is structural, not stylistic: the two paths previously carried separate copies of
the head-pose maths, drifted into different units, and the model ended up being fed values
~19 standard deviations outside its training distribution — a constant predictor that
looked like it was working. One implementation cannot drift, and
`test_live_feature_values_fall_inside_training_range` guards it.

---

## Retraining

```bash
python scripts/check_paths.py
```

```bash
python scripts/extract_features.py
```

```bash
python scripts/train_model.py
```

Point `DAISEE_ROOT` at the dataset if it is not at the default path:

```bash
set DAISEE_ROOT=D:\datasets\DAiSEE\DAiSEE
```

Training writes the model, scaler, threshold, two out-of-fold confusion matrices, and
`model_meta.json` — which carries the honest metrics and the `advisory_only` verdict that
the live pipeline acts on.

---

## Tests

```bash
python -m pytest tests/ -v
```

Covers geometry scale-invariance, yaw-robust face sizing, tier boundaries, tracker
association (including that adjacent students do not swap identity), the temporal gates,
and the heuristic/model handoff. Several are regression guards for specific bugs: the
feature-unit skew, and a tracking deadlock where wall-clock retirement ran faster than
confirmation and the room read as permanently empty.

---

## Known limits

- **CPU only.** `torch` is a CPU build; all figures above are CPU figures. A CUDA wheel
  would help YOLO, not MediaPipe.
- **Boxes lag the image** by up to one analysis cycle (~170ms at 60 students). Invisible
  for seated students; not suitable for fast motion.
- **A student entering the room takes up to ~0.5s to be picked up**, because person
  detection runs every third cycle. Lower `DETECT_EVERY` to 1 if admission latency matters
  more than throughput.
- **The stress test proves throughput, not accuracy.** It tiles one face 60 times — same
  angle, same lighting. Real classrooms vary in both.
- **Tracking is IoU-based**, which suits seated students. People who cross paths may swap
  identities and with them their temporal history.
- **A student who moves far enough to break box association can read as two for up to
  0.6s** before the abandoned track leaves the presence window. The duplicate cannot be
  removed outright — for a moment an abandoned track and a briefly-occluded one are
  genuinely indistinguishable — but it is a flicker rather than a standing miscount.
- **13 subjects** bounds everything the classifier can claim.
