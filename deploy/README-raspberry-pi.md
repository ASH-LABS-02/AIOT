# Running ClassSense on a Raspberry Pi 5

Target: Pi 5, 8GB, 64-bit Raspberry Pi OS (Bookworm), no accelerator.

**Read this first.** Every performance figure in the main README was measured on
a 20-core x86 desktop. A Pi 5 has four slower cores. Copying those constants
across does not produce a slow system — it produces a *confidently wrong* one,
because the failure is silent: the pipeline still runs, still draws boxes, still
prints an engagement percentage, and simply stops sampling often enough for any
of it to mean anything. A student examined every two seconds can sleep through a
lesson reading "Attentive".

That is why step 4 is not optional.

---

## 1. System packages

```bash
sudo apt update && sudo apt install -y python3-venv python3-dev libgl1 libglib2.0-0
```

`libgl1` and `libglib2.0-0` are what OpenCV links against. Without them the
import fails with a message about `libGL.so.1` that looks unrelated.

## 2. Virtual environment

```bash
cd ~/machine_learning
python3 -m venv classsense-env
source classsense-env/bin/activate
pip install --upgrade pip
```

## 3. Dependencies

Install torch first, from the CPU index. Pulling it in as an ultralytics
dependency can drag in a build far larger than needed:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -r requirements-pi.txt
```

Expect 10–20 minutes and about 2GB of disk. Memory is not the constraint — 8GB
is ample; the pipeline peaks near 1GB with a pool of 2.

Check it:

```bash
python scripts/test_setup.py
```

## 4. Calibrate — do not skip

```bash
python scripts/calibrate.py
```

This measures *this* machine: landmark cost per face, detection cost at each
input width, and whether each width actually finds people reliably. It then
computes how many students can be watched while still sampling every temporal
gate often enough, and writes `classsense/tuning.json`.

It is gitignored on purpose. It describes one machine.

Point it at a frame that contains people, or the detection-quality half of the
measurement means nothing:

```bash
python scripts/calibrate.py --source /path/to/a/classroom/clip.mp4
```

Expect a Pi 5 to land somewhere around **10–20 students**, limited by compute.
If that is below what you need, the options in order of effect are: a wider
camera at higher resolution won't help (compute is the limit, not pixels);
an AI HAT with a Hailo-8L moves detection off the CPU entirely; or accept
fewer students.

To see the trade-offs without committing:

```bash
python scripts/calibrate.py --dry-run
```

## 5. Run

```bash
python scripts/live_detect.py --headless --serve 8080
```

Then from any phone or laptop on the same network:

| URL | What it is |
|---|---|
| `http://<pi-ip>:8080/` | live view plus tallies |
| `http://<pi-ip>:8080/stream` | raw MJPEG, works in an `<img>` tag or VLC |
| `http://<pi-ip>:8080/status` | JSON, for a dashboard or a polling script |
| `http://<pi-ip>:8080/snapshot` | single JPEG |

**There is no authentication, and the stream is a live camera feed of a room.**
Keep it on a trusted network. Do not port-forward it.

## 6. Run on boot

```bash
sudo cp deploy/classsense.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now classsense
journalctl -u classsense -f
```

The unit refuses to start if `tuning.json` is missing, for the reason in the
opening paragraph.

---

## Reading the status endpoint

```json
{
  "students": 12,
  "capacity": 14,
  "capacity_compute": 14,
  "capacity_resolution": 62,
  "limited_by": "compute",
  "over_capacity": 0,
  "duplicate_merge_rate": 0.02,
  "analysis_per_sec": 4.1,
  "tuning_source": "measured"
}
```

Four fields are worth watching:

- **`over_capacity`** above 0 means students are being reported `Unmonitored`.
  They are counted as present but deliberately not judged, because the sampling
  rate would not support a judgement. Reduce the cohort, narrow the camera, or
  add hardware.
- **`duplicate_merge_rate`** above ~0.25 means person detection is unstable and
  one person is repeatedly spawning two tracks. Deduplication is holding the
  headcount together, but it is wasting a lot of work. Usually the wrong
  `--yolo-width` for the source resolution — re-run calibration against a real
  frame. On a 640×480 clip run at 1280px this hit 47%; at 640px it was zero.
- **`limited_by`** tells you which wall you are against. `compute` means faster
  hardware or fewer students; `resolution` means a closer or higher-resolution
  camera.
- **`tuning_source`** must read `measured`. Anything else means step 4 was
  skipped and the numbers are a guess.

---

## Tuning the trade-off

The student cap comes from one requirement: every temporal gate must be sampled
`FIDELITY_SAMPLES_PER_GATE` times inside its own window. Default 3, against the
shortest gate (`DISTRACTED_DURATION`, 0.8s), which means a refresh under 0.27s.

Relaxing it buys students and spends confidence:

```bash
python scripts/calibrate.py --samples-per-gate 2
```

Below 2 the gates stop being meaningful — a 0.8s event sampled once is a
coincidence, not a detection. If you go there, say so in whatever the output
feeds, because the states will keep looking exactly as authoritative.

To ask for a specific cohort and let the tool pick the best configuration that
carries it:

```bash
python scripts/calibrate.py --students 20
```

If it cannot reach 20 it says so rather than pretending.

---

## Known Pi-specific limitations

- **Detection width must suit the source.** Running YOLO far above the camera's
  native resolution makes its boxes unstable and spawns duplicate tracks.
  Calibration now measures this, but only against the frame you give it.
- **NCNN export is not wired up.** Ultralytics can export YOLOv8n to NCNN, which
  is typically 2–3× faster than PyTorch on ARM. That is the single largest
  remaining speedup available here and would directly raise the student cap.
- **The Pi has not been tested by the author.** Every Pi figure in this file is
  extrapolated from x86 measurements. Calibration exists precisely so the
  machine tells you the truth rather than inheriting my guesses — trust its
  output over this document.
- **Thermals.** A Pi 5 under sustained all-core load will throttle without a
  heatsink or fan. Watch `vcgencmd measure_temp`; throttling shows up as the
  analysis rate quietly dropping, and capacity was computed at full speed.
