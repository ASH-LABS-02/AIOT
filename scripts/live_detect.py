# scripts/live_detect.py
# Stage 5 - live classroom engagement detection.
#
#   python scripts/live_detect.py                 # webcam
#   python scripts/live_detect.py --source clip.mp4 --loop
#   python scripts/live_detect.py --no-model      # heuristics only
#
# Keys:  Q quit   S snapshot   D debug HUD
#
# The work lives in the classsense package; this file is argument parsing, the
# render loop, and keyboard handling.

import argparse
import os
import sys
import time

import cv2

# Run from anywhere: put the repo root on the path before importing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classsense import config                                    # noqa: E402
from classsense.pipeline import AnalysisWorker, FrameSource      # noqa: E402
from classsense.render import (                                  # noqa: E402
    draw_student, draw_dashboard, draw_banner, CROWDED_THRESHOLD,
)


def load_model(allow_weak=False):
    """
    Load the classifier, or return None and carry on without it.

    Deliberately non-fatal, and deliberately sceptical. The heuristic layer is
    self-sufficient; a missing, stale, or near-chance model should quietly
    degrade the system rather than stop it or quietly corrupt it.
    """
    import json
    import joblib

    paths = (config.MODEL_PATH, config.SCALER_PATH, config.THRESH_PATH)
    if not all(os.path.exists(p) for p in paths):
        print("No trained model found - running on heuristics only.", flush=True)
        return None, None, 0.5

    # Refuse a model that measured close to chance. A classifier at AUC ~0.59
    # paints red boxes on attentive students often enough to cost the operator
    # their trust in the colours the heuristics get right, and a monitor nobody
    # believes is worse than one that says less.
    meta = {}
    if os.path.exists(config.META_PATH):
        try:
            with open(config.META_PATH, encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            meta = {}

    if meta:
        print(
            f"Model card: {meta.get('oof_accuracy', '?')} accuracy vs "
            f"{meta.get('majority_baseline', '?')} baseline, "
            f"AUC {meta.get('oof_roc_auc', '?')}, "
            f"{meta.get('n_subjects', '?')} subjects.",
            flush=True,
        )

    if meta.get("advisory_only") and not allow_weak:
        print(f"  -> {meta.get('advisory_reason', 'model is near chance')}.",
              flush=True)
        print("  -> Not used for decisions. Heuristics only. "
              "Pass --use-weak-model to override.", flush=True)
        return None, None, 0.5

    try:
        classifier = joblib.load(config.MODEL_PATH)
        scaler = joblib.load(config.SCALER_PATH)
        threshold = float(joblib.load(config.THRESH_PATH))
    except Exception as exc:
        print(f"Could not load model ({exc}) - heuristics only.", flush=True)
        return None, None, 0.5

    expected = len(config.FEATURE_COLS)
    actual = getattr(scaler, "n_features_in_", expected)
    if actual != expected:
        # This is exactly the failure that made the previous model a constant
        # predictor: a scaler fitted on a different feature set than the one
        # being handed to it. Refuse it rather than feed it nonsense.
        print(
            f"Model expects {actual} features, pipeline produces {expected}. "
            f"Refusing to use a mismatched model - retrain with "
            f"scripts/train_model.py. Running on heuristics only.",
            flush=True,
        )
        return None, None, 0.5

    print(f"Model loaded (threshold {threshold:.2f}).", flush=True)
    return classifier, scaler, threshold


def parse_args():
    p = argparse.ArgumentParser(description="ClassSense AI live detection")
    p.add_argument("--source", default=None,
                   help="video file path; omit for the webcam")
    p.add_argument("--camera", type=int, default=config.CAPTURE_INDEX,
                   help="camera index (default 0)")
    p.add_argument("--loop", action="store_true",
                   help="loop a video file source")
    p.add_argument("--no-model", action="store_true",
                   help="skip the classifier, heuristics only")
    p.add_argument("--use-weak-model", action="store_true",
                   help="use the classifier even if it measured near chance")
    p.add_argument("--pool", type=int, default=config.MP_POOL_SIZE,
                   help=f"MediaPipe detectors (default {config.MP_POOL_SIZE})")
    p.add_argument("--yolo-width", type=int, default=config.YOLO_INPUT_WIDTH,
                   help=f"YOLO input width (default {config.YOLO_INPUT_WIDTH})")
    p.add_argument("--max-per-cycle", type=int,
                   default=config.MAX_STUDENTS_PER_CYCLE,
                   help="students analysed per cycle before scheduling kicks in")
    p.add_argument("--debug", action="store_true", help="start with the HUD on")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60, flush=True)
    print("ClassSense AI - live detection", flush=True)
    print("Attentive (green) | Sleepy (orange) | Distracted (red) | "
          "Unknown (gray)", flush=True)
    print("=" * 60, flush=True)

    os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)

    classifier, scaler, threshold = (None, None, 0.5)
    if not args.no_model:
        classifier, scaler, threshold = load_model(
            allow_weak=args.use_weak_model
        )

    source_arg = args.source if args.source else args.camera
    print(f"Opening source: {source_arg!r}", flush=True)
    source = FrameSource(source_arg, loop=args.loop).start()
    print(f"Source ready at {source.actual_size[0]}x{source.actual_size[1]}.",
          flush=True)

    worker = AnalysisWorker(
        classifier=classifier,
        scaler=scaler,
        threshold=threshold,
        pool_size=args.pool,
        yolo_width=args.yolo_width,
        max_per_cycle=args.max_per_cycle,
    ).start(source)

    print("\nQ quit   S snapshot   D debug HUD\n", flush=True)

    show_debug = args.debug
    window = "ClassSense AI"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)

    fps_ema = None
    last_t = time.time()

    try:
        while source.running:
            frame = source.read()
            if frame is None:
                break

            now = time.time()
            dt = now - last_t
            last_t = now
            inst = 1.0 / dt if dt > 0 else 0.0
            fps_ema = inst if fps_ema is None else fps_ema * 0.9 + inst * 0.1

            students, counts, tracked, readable = worker.snapshot()
            crowded = len(students) > CROWDED_THRESHOLD

            for student in students:
                draw_student(frame, student, crowded, show_debug)

            draw_dashboard(
                frame, counts, fps_ema, worker.analysis_fps,
                tracked, readable, show_debug, worker.cycle_ms,
                worker.merged_total,
            )

            if worker.cycles == 0:
                draw_banner(frame, "Warming up - first analysis cycle running")
            elif crowded:
                draw_banner(
                    frame,
                    "Crowded view: only students needing attention are labelled",
                )

            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                path = os.path.join(
                    config.SNAPSHOT_DIR, f"snap_{int(time.time())}.jpg"
                )
                cv2.imwrite(path, frame)
                print(f"Snapshot saved: {path}", flush=True)
            if key in (ord("d"), ord("D")):
                show_debug = not show_debug
                print(f"HUD {'on' if show_debug else 'off'}", flush=True)

    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
    finally:
        worker.stop()
        source.stop()
        cv2.destroyAllWindows()

    if worker.cycles:
        print(f"\n{worker.cycles} analysis cycles, "
              f"{worker.cycle_ms:.0f}ms each "
              f"({worker.analysis_fps:.1f}/s).", flush=True)
    print("Stopped.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
