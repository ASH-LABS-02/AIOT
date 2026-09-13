# scripts/stress_test.py
# Proves the throughput claim under real load instead of extrapolating it.
#
#   python scripts/stress_test.py                 # 10,20,30,45,60 students
#   python scripts/stress_test.py --cells 60 --cycles 30
#
# Builds a synthetic 1080p frame with N faces tiled into it and runs the real
# AnalysisWorker over it. What this does and does not prove:
#
#   does     - the actual cost of N YOLO detections plus N landmark passes plus
#              N state updates through the real pipeline and its locking
#   does not - anything about accuracy. Every face is the same face at the same
#              angle under the same light. This is a throughput test.

import argparse
import os
import statistics
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classsense import config                                   # noqa: E402
from classsense.pipeline import AnalysisWorker, make_tiled_source  # noqa: E402
from classsense.tiers import Tier, TIER_LABELS                  # noqa: E402

DAISEE_TRAIN = os.path.join(config.DAISEE_ROOT, "DataSet", "Train")


def sample_face_frame():
    """A real face to tile: DAiSEE if present, else the webcam."""
    if os.path.isdir(DAISEE_TRAIN):
        for person in sorted(os.scandir(DAISEE_TRAIN), key=lambda e: e.name):
            if not person.is_dir():
                continue
            for clip in sorted(os.scandir(person.path), key=lambda e: e.name):
                if not clip.is_dir():
                    continue
                for f in os.scandir(clip.path):
                    if f.name.lower().endswith((".avi", ".mp4")):
                        cap = cv2.VideoCapture(f.path)
                        ok, frame = cap.read()
                        cap.release()
                        if ok:
                            return frame, f.name
    cap = cv2.VideoCapture(config.CAPTURE_INDEX)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("No face source: DAiSEE missing and no webcam.")
    return frame, "webcam"


def run_case(worker, frame, cycles):
    """Time `cycles` analysis passes over one frame."""
    worker.analyse(frame)              # warm: first pass pays model warmup
    worker.tracker.students.clear()

    timings = []
    for _ in range(cycles):
        t0 = time.perf_counter()
        worker.analyse(frame)
        timings.append((time.perf_counter() - t0) * 1000.0)

    confirmed = worker.tracker.confirmed()
    tiers = {t: 0 for t in Tier}
    for student in confirmed.values():
        tiers[student.tier] += 1

    return {
        "median_ms": statistics.median(timings),
        "p95_ms": sorted(timings)[int(len(timings) * 0.95) - 1],
        "tracked": len(worker.tracker.students),
        "confirmed": len(confirmed),
        "tiers": tiers,
    }


def main():
    p = argparse.ArgumentParser(description="ClassSense throughput stress test")
    p.add_argument("--cells", type=int, nargs="*",
                   default=[10, 20, 30, 45, 60],
                   help="student counts to test")
    p.add_argument("--cycles", type=int, default=12,
                   help="timed analysis cycles per case")
    p.add_argument("--pool", type=int, default=config.MP_POOL_SIZE)
    p.add_argument("--yolo-width", type=int, default=config.YOLO_INPUT_WIDTH)
    p.add_argument("--save-frames", action="store_true",
                   help="write each synthetic frame to data/snapshots")
    args = p.parse_args()

    print("=" * 74)
    print("ClassSense stress test - synthetic tiled classroom @ 1080p")
    print("=" * 74)
    print(f"cores {os.cpu_count()}   pool {args.pool}   "
          f"yolo {args.yolo_width}px   cycles/case {args.cycles}")

    frame, origin = sample_face_frame()
    print(f"face source: {origin}  ({frame.shape[1]}x{frame.shape[0]})\n")

    worker = AnalysisWorker(
        classifier=None, scaler=None,
        pool_size=args.pool, yolo_width=args.yolo_width,
        max_per_cycle=max(args.cells) if args.cells else 60,
    )

    print(f"{'cells':>6} {'median':>9} {'p95':>9} {'cycles/s':>9} "
          f"{'refresh':>9} {'found':>7} {'full':>6} {'coarse':>7} {'far':>5}")
    print("-" * 74)

    rows = []
    try:
        for cells in args.cells:
            tiled = make_tiled_source(frame, cells)
            if args.save_frames:
                os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
                cv2.imwrite(
                    os.path.join(config.SNAPSHOT_DIR, f"stress_{cells}.jpg"),
                    tiled,
                )

            worker.tracker.students.clear()
            r = run_case(worker, tiled, args.cycles)

            cps = 1000.0 / r["median_ms"] if r["median_ms"] else 0.0
            # Every scheduled student is analysed each cycle here, so refresh
            # equals cycle time. It would exceed it only once the cohort
            # outgrows max_per_cycle and scheduling starts rotating.
            refresh = r["median_ms"] / 1000.0
            flag = "" if refresh <= 1.0 else " SLOW"

            print(f"{cells:>6} {r['median_ms']:>8.1f}ms {r['p95_ms']:>8.1f}ms "
                  f"{cps:>9.2f} {refresh:>8.2f}s {r['confirmed']:>7} "
                  f"{r['tiers'][Tier.FULL]:>6} {r['tiers'][Tier.COARSE]:>7} "
                  f"{r['tiers'][Tier.PRESENCE]:>5}{flag}")
            rows.append((cells, r, refresh))
    finally:
        worker.stop()

    print("\n" + "=" * 74)
    print("Reading this")
    print("=" * 74)
    print("refresh = seconds between successive looks at any one student.")
    print("Eye-closure detection needs refresh well under SLEEPY_EYES_DURATION")
    print(f"({config.SLEEPY_EYES_DURATION}s) to get several samples inside a closure.")
    print()
    print("full/coarse/far are resolution tiers:")
    for tier in (Tier.FULL, Tier.COARSE, Tier.PRESENCE):
        print(f"  {tier.name:<9} {TIER_LABELS[tier]}")
    print()

    worst = [(c, rf) for c, _, rf in rows if rf > 1.0]
    if worst:
        print(f"Over budget at: {', '.join(str(c) for c, _ in worst)} students.")
        print("Lower --yolo-width, or MAX_STUDENTS_PER_CYCLE to enable rotation.")
    else:
        biggest = rows[-1] if rows else None
        if biggest:
            cells, r, refresh = biggest
            print(f"Within budget through {cells} students "
                  f"({r['median_ms']:.0f}ms/cycle, {refresh:.2f}s refresh).")


if __name__ == "__main__":
    main()
