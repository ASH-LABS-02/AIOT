# ClassSense AI

Real-time classroom engagement monitoring. Detects people with YOLOv8, reads facial
landmarks with MediaPipe, and classifies each student as **Attentive**, **Sleepy**,
**Distracted**, or **Unknown** using temporal rules over eye, mouth and head-pose signals.

Measured at **60 students in a single 1080p frame at 3.7 analysis cycles per second on
CPU** — every student re-examined roughly four times a second, no GPU.

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

Every trigger is **wall-clock gated**, not frame-counted. A blink and a microsleep look
identical in one frame and differ only in duration, so the system waits before committing.
This is why the labels do not strobe.

---

## Resolution tiers — why some students read "Unknown"

A 1080p frame cannot give 60 faces equal quality. The front row may land 120px of face
width and the back row 40px. Rather than print identical-looking labels of wildly
different trustworthiness, each student is tiered by **measured face width** and the tier
bounds what may be concluded:

| Tier | Face width | Analysis | States reachable |
|---|---|---|---|
| `FULL` | ≥ 64px | eyes, mouth, head pose | all, including Sleepy |
| `COARSE` | 40–64px | head pose only | Attentive, Distracted |
| `PRESENCE` | < 40px | presence only | Unknown |

Eye aspect ratio is suppressed below `FULL` because at a 50px face the eye landmarks sit
about 3px apart — one pixel of jitter moves EAR by ~10%, more than the gap between an open
and a closed eye. Head pose spans the whole face and degrades gracefully, so it survives
one tier lower.

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

| Students | Cycle | Cycles/s | Refresh |
|---|---|---|---|
| 10 | 135 ms | 7.4 | 0.14 s |
| 20 | 164 ms | 6.1 | 0.16 s |
| 30 | 196 ms | 5.1 | 0.20 s |
| 45 | 229 ms | 4.4 | 0.23 s |
| **60** | **269 ms** | **3.7** | **0.27 s** |

"Refresh" is the gap between successive looks at any one student. At 0.27s a 1.2s eye
closure gets roughly four samples, so Sleepy detection remains viable at full capacity.

Three findings shaped the design:

- **MediaPipe `detect()` releases the GIL** — 3.21× across 8 threads. A thread pool is
  genuinely parallel, which is what makes 60 students affordable without a GPU.
- **Crop resolution above 192px costs double and buys nothing** — 7.13 / 7.21 / 6.71 ms at
  96 / 128 / 192px, then 14.67 ms at 256px. Crops are capped at 192px, aspect preserved.
- **Batching the classifier matters enormously** — 60 separate `predict_proba` calls on a
  300-tree forest measure 3.7s; the same 60 rows batched measure 56ms.

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

scripts/
  live_detect.py       live detection (entry point)
  benchmark.py         profile this machine
  stress_test.py       prove throughput at N students
  extract_features.py  DAiSEE clips -> features
  train_model.py       train + subject-independent evaluation
  check_paths.py       what dataset is actually on disk

tests/                 66 tests
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

Covers geometry scale-invariance, tier boundaries, tracker association (including that
adjacent students do not swap identity), the temporal gates, and the heuristic/model
handoff. Several are regression guards for specific bugs: the feature-unit skew, and a
tracking deadlock where wall-clock retirement ran faster than confirmation and the room
read as permanently empty.

---

## Known limits

- **CPU only.** `torch` is a CPU build; all figures above are CPU figures. A CUDA wheel
  would help YOLO, not MediaPipe.
- **Boxes lag the image** by up to one analysis cycle (~270ms at 60 students). Invisible
  for seated students; not suitable for fast motion.
- **The stress test proves throughput, not accuracy.** It tiles one face 60 times — same
  angle, same lighting. Real classrooms vary in both.
- **Tracking is IoU-based**, which suits seated students. People who cross paths may swap
  identities and with them their temporal history.
- **13 subjects** bounds everything the classifier can claim.
