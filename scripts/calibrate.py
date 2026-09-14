# scripts/calibrate.py
# Measure this machine and write the tuning it should actually run with.
#
#   python scripts/calibrate.py                 # measure and save
#   python scripts/calibrate.py --dry-run       # measure, print, save nothing
#   python scripts/calibrate.py --samples-per-gate 4    # stricter fidelity
#
# RUN THIS ON THE MACHINE THAT WILL DO THE WATCHING. Every performance constant
# in this project was measured on a 20-core desktop; a Raspberry Pi 5 has four
# slower cores. Copying those constants across does not produce a slow system,
# it produces a confidently wrong one - students sampled too rarely for a 1.2s
# eye closure to be caught, while the dashboard still prints a percentage.
#
# The output is classsense/tuning.json, which is gitignored because it
# describes one machine.

import argparse
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classsense import capacity as cap_mod                      # noqa: E402
from classsense import config                                   # noqa: E402
from classsense.capacity import Capacity, default_pool_size     # noqa: E402
from classsense.detector import (                                # noqa: E402
    ncnn_available, ncnn_export_width, load_detector,
)
from classsense.mp_pool import MediaPipePool, ensure_model      # noqa: E402
import mediapipe as mp                                          # noqa: E402

RULE = "=" * 68


def sample_frame(source=None):
    """A real frame to measure on: a given file, DAiSEE, or the camera."""
    if source:
        cap = cv2.VideoCapture(source)
        ok, frame = cap.read()
        cap.release()
        if ok:
            return frame, str(source)

    train = os.path.join(config.DAISEE_ROOT, "DataSet", "Train")
    if os.path.isdir(train):
        for person in sorted(os.scandir(train), key=lambda e: e.name):
            if not person.is_dir():
                continue
            for clip in sorted(os.scandir(person.path), key=lambda e: e.name):
                if not clip.is_dir():
                    continue
                for f in os.scandir(clip.path):
                    if f.name.lower().endswith((".avi", ".mp4")):
                        c = cv2.VideoCapture(f.path)
                        ok, frame = c.read()
                        c.release()
                        if ok:
                            return frame, f"DAiSEE {f.name}"

    c = cv2.VideoCapture(config.CAPTURE_INDEX)
    ok, frame = c.read()
    c.release()
    if not ok:
        raise RuntimeError(
            "No frame source. Pass --source <video file>, attach a camera, "
            "or set DAISEE_ROOT."
        )
    return frame, "camera"


def measure_per_face(frame, pool_size, trials=25):
    """
    Effective cost of one face, with the pool running flat out.

    Measured as total wall time for a batch divided by the batch size, not as
    a single call, because that is how the pipeline actually uses it - what
    matters is throughput under contention, not latency in isolation.
    """
    ensure_model()
    pool = MediaPipePool(pool_size, face_conf=config.MP_FACE_CONF_THRESH)
    executor = ThreadPoolExecutor(max_workers=pool_size)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    side = config.MP_CROP_MAX_SIDE
    h, w = rgb.shape[:2]
    scale = side / float(max(h, w))
    small = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=np.ascontiguousarray(small))

    for _ in range(pool_size * 2):          # warm every detector
        pool.detect(image)

    batch = max(pool_size * 3, 12)
    per_batch = []
    for _ in range(max(3, trials // 4)):
        t0 = time.perf_counter()
        list(executor.map(lambda _: pool.detect(image), range(batch)))
        per_batch.append((time.perf_counter() - t0) * 1000.0 / batch)

    executor.shutdown(wait=True)
    pool.close()
    return statistics.median(per_batch)


def measure_detect(frame, width, capture_size, trials=6, backend="pytorch"):
    """
    Cost AND reliability of one YOLO pass at this input width.

    Reliability matters as much as speed and is easy to miss. Running the
    detector at a width far from the source resolution makes its boxes
    unstable - they jitter, associate badly, and spawn duplicate tracks on one
    person. Measured here: a 640x480 clip detected at 1280px produced a
    duplicate merge on 47% of cycles and half the throughput; at 640px, none.
    A width that is fast but finds nobody is worse than a slower one.

    The frame is scaled to the capture size the pipeline will actually use, so
    the measurement reflects deployment rather than whatever clip was handy.
    """
    yolo, actual = load_detector(backend=backend, yolo_width=width, strict=False)
    if actual != backend:
        return None            # asked for NCNN, could not have it at this width
    full = cv2.resize(frame, capture_size, interpolation=cv2.INTER_LINEAR)

    for _ in range(2):
        yolo(full, classes=[0], imgsz=width, verbose=False)

    samples = []
    found = []
    confs = []
    for _ in range(trials):
        t0 = time.perf_counter()
        out = yolo(full, classes=[0], imgsz=width, verbose=False)[0]
        samples.append((time.perf_counter() - t0) * 1000.0)
        kept = [float(b.conf[0]) for b in out.boxes
                if float(b.conf[0]) >= config.YOLO_CONF_THRESH]
        found.append(len(kept))
        confs.extend(kept)

    return {
        "ms": statistics.median(samples),
        "people": statistics.median(found),
        "conf": statistics.median(confs) if confs else 0.0,
        # The same frame every time, so any variation in the count is the
        # detector being unstable rather than the scene changing.
        "stable": len(set(found)) == 1,
        "backend": backend,
    }


def main():
    p = argparse.ArgumentParser(description="Calibrate ClassSense to this machine")
    p.add_argument("--source", default=None, help="video file to sample a frame from")
    p.add_argument("--pool", type=int, default=None,
                   help="detector threads (default: derived from core count)")
    p.add_argument("--samples-per-gate", type=float,
                   default=config.FIDELITY_SAMPLES_PER_GATE,
                   help="samples required inside the shortest temporal gate")
    p.add_argument("--detect-every", type=int, default=None,
                   help="cycles between person-detection passes")
    p.add_argument("--widths", type=int, nargs="*", default=[480, 640, 960, 1280],
                   help="YOLO input widths to try")
    p.add_argument("--students", type=int, default=None,
                   help="cohort you need; picks the best-quality config that carries it")
    p.add_argument("--max-admission", type=float, default=1.0,
                   help="seconds a new student may go unnoticed (default 1.0)")
    p.add_argument("--dry-run", action="store_true", help="measure but do not save")
    args = p.parse_args()

    cores = os.cpu_count() or 4
    pool_size = args.pool or default_pool_size(cores)

    print(RULE)
    print("ClassSense calibration")
    print(RULE)
    print(f"host      : {sys.platform}, {cores} cores")
    try:
        import platform
        print(f"machine   : {platform.machine()}  {platform.processor() or ''}".rstrip())
    except Exception:
        pass
    print(f"pool size : {pool_size}")

    frame, origin = sample_frame(args.source)
    print(f"frame from: {origin} ({frame.shape[1]}x{frame.shape[0]})")

    # Resolution ceiling is judged against the capture size the pipeline will
    # actually run at, not the size of whatever frame was handy for timing.
    frame_w, frame_h = config.CAPTURE_WIDTH, config.CAPTURE_HEIGHT
    print(f"capture   : {frame_w}x{frame_h} (config.CAPTURE_WIDTH/HEIGHT)")

    # ── 1. per-face cost ───────────────────────
    print("\n" + RULE)
    print("1. Landmark cost per face (pool running flat out)")
    print(RULE)
    per_face = measure_per_face(frame, pool_size)
    print(f"{per_face:.2f} ms per face at {config.MP_CROP_MAX_SIDE}px crops")

    # ── 2. detection cost by width ─────────────
    print("\n" + RULE)
    print("2. Person detection cost by input width")
    print(RULE)
    # Both backends are measured rather than assumed. NCNN is built for ARM
    # NEON and is the point of exporting at all on a Pi, but on x86 it measured
    # SLOWER than PyTorch (53ms against 29ms at 640px). Neither is universally
    # right, so the machine decides.
    backends = ["pytorch"]
    if ncnn_available():
        exported = ncnn_export_width(config.YOLO_NCNN_DIR)
        backends.append("ncnn")
        print(f"NCNN export found (width {exported}); measuring both backends.")
        if exported is not None and exported not in args.widths:
            print(f"  Only {exported}px can use NCNN - an export has a fixed")
            print("  input shape and returns nothing at any other width.")
    else:
        print("No NCNN export. For ARM, see scripts/export_ncnn.py.")
    print()

    print(f"{'backend':<9} {'width':>6} {'ms':>8} {'people':>8} {'conf':>7}  detection")
    print("-" * 60)
    detect_costs = {}
    detect_info = {}
    for width in args.widths:
        for backend in backends:
            info = measure_detect(frame, width, (frame_w, frame_h),
                                  backend=backend)
            if info is None:
                print(f"{backend:<9} {width:>6} {'-':>8} "
                      f"{'-':>8} {'-':>7}  unavailable at this width")
                continue
            if info["people"] == 0:
                verdict = "FINDS NOBODY"
            elif not info["stable"]:
                verdict = "unstable count"
            else:
                verdict = "ok"
            print(f"{backend:<9} {width:>6} {info['ms']:>8.0f} "
                  f"{info['people']:>8.0f} {info['conf']:>7.2f}  {verdict}")

            # Keep whichever backend is both usable and faster at this width.
            best_here = detect_info.get(width)
            usable = info["people"] > 0
            if best_here is None or (usable and info["ms"] < best_here["ms"]):
                if usable or best_here is None:
                    detect_info[width] = info
                    detect_costs[width] = info["ms"]
    print()
    chosen_backends = {w: detect_info[w]["backend"] for w in detect_info}
    if len(set(chosen_backends.values())) > 1 or "ncnn" in chosen_backends.values():
        print("Fastest backend per width: "
              + ", ".join(f"{w}px={b}" for w, b in sorted(chosen_backends.items())))

    # A width finding more people than another, at lower confidence, on the
    # same frame has two readings and this tool cannot tell them apart:
    # either it is genuinely resolving faces the narrower one missed, or it is
    # splitting one person into several. Both happen. Duplicate tracks get
    # merged downstream by face position, so the headcount survives either way,
    # but the splitting case wastes a great deal of work - a 640x480 clip run
    # at 1280px merged a duplicate on 47% of cycles and ran at half speed.
    counts_seen = {w: detect_info[w]["people"] for w in args.widths}
    if len(set(counts_seen.values())) > 1:
        fewest = min(counts_seen.values())
        print()
        print("NOTE: the widths disagree on how many people are in the frame:")
        for w in args.widths:
            i = detect_info[w]
            print(f"  {w:>4}px -> {i['people']:.0f} at confidence {i['conf']:.2f}")
        print(f"If the frame really holds {fewest:.0f}, the higher counts are")
        print("one person being split, and the narrower width is the right")
        print("choice. If it holds more, the wider width is finding the rest.")
        print("Confirm with: python scripts/diagnose_duplicates.py")

    usable = [w for w in args.widths if detect_info[w]["people"] > 0]
    if not usable:
        print("\nNo width detected anyone in the sample frame. If the frame")
        print("has no person in it that is expected - pass --source with one")
        print("that does, or these numbers say nothing about detection quality.")
    elif len(usable) < len(args.widths):
        skipped = [w for w in args.widths if w not in usable]
        print(f"\nExcluding {skipped}: found nobody in the sample frame.")
        args.widths = usable

    # ── 3. choose a configuration ──────────────
    print("\n" + RULE)
    print("3. Configurations that meet the fidelity budget")
    print(RULE)

    shortest = min(config.SLEEPY_EYES_DURATION, config.DISTRACTED_DURATION)
    budget = shortest / args.samples_per_gate
    print(f"shortest temporal gate : {shortest}s "
          f"({'SLEEPY_EYES' if shortest == config.SLEEPY_EYES_DURATION else 'DISTRACTED'}_DURATION)")
    print(f"samples required       : {args.samples_per_gate:g}")
    print(f"-> refresh must stay under {budget:.2f}s\n")

    print(f"{'width':>7} {'every':>7} {'detect ms':>11} {'students':>10} "
          f"{'refresh':>9} {'admission':>11}  limit")
    print("-" * 70)

    options = []
    every_choices = [args.detect_every] if args.detect_every else [1, 2, 3, 4, 6]
    for width in args.widths:
        for every in every_choices:
            candidate = Capacity(
                per_face_ms=per_face,
                detect_ms=detect_costs[width],
                cores=cores,
                pool_size=pool_size,
                yolo_width=width,
                detect_every=every,
                samples_per_gate=args.samples_per_gate,
                backend=detect_info[width]["backend"],
            )
            n = candidate.max_students(frame_w, frame_h)
            if n <= 0:
                continue
            admission = candidate.admission_seconds
            over = admission > args.max_admission
            options.append((candidate, n, admission, over))
            flag = "  <-- slow to notice arrivals" if over else ""
            print(f"{width:>7} {every:>7} {detect_costs[width]:>10.0f} "
                  f"{n:>10} {candidate.refresh_at(n):>8.2f}s "
                  f"{admission:>10.1f}s  "
                  f"{candidate.limiting_factor(frame_w, frame_h)}{flag}")

    if not options:
        print("\nNo configuration meets the budget on this machine.")
        print("Options: lower --samples-per-gate, raise --detect-every, or")
        print("shorten the gates in config.py - but each of those trades away")
        print("confidence in the states, so decide it deliberately.")
        sys.exit(1)

    # Selection, in order of what actually matters.
    #
    # Maximising the student count alone is a trap: the narrowest detector and
    # the longest detection interval always "win" it, and both are paid for in
    # things the number does not show - a 480px detector misses small back-row
    # faces entirely, and a long interval leaves an arriving student unnoticed
    # for seconds. So admission latency is a hard constraint, and among the
    # configurations that clear it, the WIDEST detector that still carries the
    # required cohort wins. Detection quality is kept unless it has to be spent.
    viable = [o for o in options if not o[3]] or options
    if all(o[3] for o in options):
        print(f"\nNOTE: nothing met the {args.max_admission:.0f}s admission "
              f"limit; choosing the least-bad.")

    if args.students:
        enough = [o for o in viable if o[1] >= args.students]
        if enough:
            best, n_best, _, _ = max(
                enough,
                key=lambda o: (round(detect_info[o[0].yolo_width]['conf'], 1),
                               o[0].yolo_width, -o[0].detect_every))
            print(f"\nNeed {args.students} students: choosing the widest "
                  f"detector that carries them.")
        else:
            best, n_best, _, _ = max(viable, key=lambda o: o[1])
            print(f"\nCannot reach {args.students} students on this machine. "
                  f"Best is {n_best}.")
    else:
        # No target given: take the most students, then prefer the widest
        # detector and shortest interval among ties.
        peak = max(o[1] for o in viable)
        # Anything within 15% of peak is equivalent in practice; buy detection
        # quality with the difference rather than chasing the last few seats.
        near = [o for o in viable if o[1] >= peak * 0.85]
        best, n_best, _, _ = max(
            near,
            key=lambda o: (round(detect_info[o[0].yolo_width]['conf'], 1),
                           o[0].yolo_width, -o[0].detect_every))

    print("\n" + RULE)
    print("Chosen")
    print(RULE)
    print(best.summary())

    # A wider detector finds small back-row faces that a narrow one misses, so
    # note when the throughput-optimal choice is also the narrowest tried.
    if best.yolo_width == min(args.widths):
        print()
        print(f"NOTE: {best.yolo_width}px was the narrowest width tried. It is")
        print("fastest, but small or distant faces may go undetected entirely.")
        print("Check with scripts/diagnose_duplicates.py on a real view before")
        print("trusting the headcount.")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    path = cap_mod.save(best)
    print(f"\nwritten: {path}")
    print("live_detect.py will now use these values. Re-run this after any")
    print("hardware change, or after moving the code to another machine.")


if __name__ == "__main__":
    main()
