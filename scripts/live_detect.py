# scripts/live_detect.py
# Stage 5 - live classroom engagement detection.
#
#   python scripts/live_detect.py                      # window, webcam
#   python scripts/live_detect.py --serve 8080         # window + web stream
#   python scripts/live_detect.py --headless --serve 8080    # Raspberry Pi
#   python scripts/live_detect.py --source clip.mp4 --loop
#
# Keys (windowed only):  Q quit   S snapshot   D debug HUD
#
# Tuning comes from classsense/tuning.json when present, written by
# scripts/calibrate.py on the machine that will run this. Without it the
# pipeline falls back to an estimate from the core count and says so - the
# constants in config.py were measured on a 20-core desktop and are wrong
# anywhere else, quietly.

import argparse
import os
import signal
import sys
import time

import cv2

# Run from anywhere: put the repo root on the path before importing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classsense import capacity as cap_mod                      # noqa: E402
from classsense import config                                   # noqa: E402
from classsense.pipeline import AnalysisWorker, FrameSource     # noqa: E402
from classsense.render import (                                 # noqa: E402
    draw_student, draw_dashboard, draw_banner, CROWDED_THRESHOLD,
)
from classsense.server import StreamProvider, serve             # noqa: E402
from classsense.session import SessionRecorder                  # noqa: E402


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

    p.add_argument("--headless", action="store_true",
                   help="no window; for a Pi over SSH or as a service")
    p.add_argument("--serve", type=int, metavar="PORT", default=None,
                   help="serve the view and status over HTTP on this port")
    p.add_argument("--stream-fps", type=int, default=8,
                   help="cap the MJPEG rate (default 8), to leave CPU for analysis")

    p.add_argument("--no-model", action="store_true",
                   help="skip the classifier, heuristics only")
    p.add_argument("--use-weak-model", action="store_true",
                   help="use the classifier even if it measured near chance")

    p.add_argument("--students", type=int, default=None,
                   help="override the measured student cap (not advised)")
    p.add_argument("--pool", type=int, default=None,
                   help="MediaPipe detectors (default: from calibration)")
    p.add_argument("--yolo-width", type=int, default=None,
                   help="YOLO input width (default: from calibration)")
    p.add_argument("--backend", choices=("pytorch", "ncnn"), default=None,
                   help="detector backend (default: from calibration)")
    p.add_argument("--detect-every", type=int, default=None,
                   help="cycles between detection passes (default: from calibration)")
    p.add_argument("--allow-over-capacity", action="store_true",
                   help="analyse everyone at reduced fidelity instead of "
                        "reporting the overflow as Unmonitored")
    p.add_argument("--duration", type=float, default=None, metavar="SECONDS",
                   help="stop cleanly after this long and write the report; "
                        "a lesson has a length, and a hard kill loses the session")
    p.add_argument("--no-record", action="store_true",
                   help="skip session recording and the written report")
    p.add_argument("--report-dir", default=None,
                   help="where to write the session report (default data/reports)")
    p.add_argument("--debug", action="store_true", help="start with the HUD on")
    return p.parse_args()


def build_status(counts, students, readable, worker, cap, over, fps,
                 effective_cap, merge_rate):
    """
    The JSON a dashboard or a polling script reads.

    `effective_cap` is passed in rather than re-derived from `cap`, because the
    capacity that matters is the one after the camera's real resolution has
    been taken into account - a 640x480 source cannot carry the cohort a 1080p
    ceiling promises, and reporting the nominal figure would overstate it.
    """
    attentive = counts.get("Attentive", 0)
    note = ""
    if over:
        note = (f"{over} student(s) beyond the capacity of {effective_cap} are "
                f"reported Unmonitored. Run scripts/calibrate.py, narrow the "
                f"camera, or use faster hardware.")
    elif merge_rate > 0.25:
        # Every merge is one person who briefly held two tracks. A steady high
        # rate means person detection is unstable on this input - often the
        # wrong yolo_width for the frame size - and the headcount is only
        # correct because deduplication keeps catching it.
        note = (f"Detection is unstable: {merge_rate:.0%} of cycles merged a "
                f"duplicate. Deduplication is holding the count together, but "
                f"try a different --yolo-width for this source.")
    elif cap.source != "measured":
        note = ("Running on an estimate, not a measurement. Run "
                "scripts/calibrate.py on this machine.")
    return {
        "students": len(students),
        "readable": readable,
        "counts": counts,
        "engagement_pct": round((attentive / readable * 100) if readable else 0.0),
        "analysis_per_sec": round(worker.analysis_fps, 1),
        "cycle_ms": round(worker.cycle_ms),
        "display_fps": round(fps or 0.0),
        "capacity": effective_cap,
        "capacity_compute": cap.compute_ceiling(),
        "capacity_resolution": cap.resolution_ceiling(),
        "limited_by": cap.limiting_factor(),
        "over_capacity": over,
        "duplicates_merged": worker.merged_total,
        "duplicate_merge_rate": round(merge_rate, 3),
        "tuning_source": cap.source,
        "note": note,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def write_report(recorder, directory=None):
    """Close the session and write both the readable and machine copies."""
    from classsense import report_html

    recorder.finalise()
    report = recorder.report()
    head = report["headline"]

    directory = directory or os.path.join(config.DATA_DIR, "reports")
    os.makedirs(directory, exist_ok=True)
    stem = time.strftime("session_%Y%m%d_%H%M%S",
                         time.localtime(recorder.started))

    json_path = recorder.save(directory, stem)
    html_path = os.path.join(directory, f"{stem}.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(report_html.render(report))

    print("\n" + "=" * 60, flush=True)
    print("Session report", flush=True)
    print("=" * 60, flush=True)
    print(f"  duration        : {report['session']['duration_s']:.0f}s", flush=True)
    print(f"  students seen   : {head['students_seen']} "
          f"(peak {head['peak_present']} at once)", flush=True)
    if head["mean_attentiveness"] is not None:
        print(f"  attentiveness   : {head['mean_attentiveness'] * 100:.0f}% mean"
              f"   {head['weighted_attentiveness'] * 100:.0f}% time-weighted",
              flush=True)
    if head["pct_never_slept"] is not None:
        print(f"  never slept     : {head['pct_never_slept']:.0f}% of students "
              f"({head['students_slept']} did)", flush=True)
    if head["pct_time_not_sleeping"] is not None:
        print(f"  awake           : {head['pct_time_not_sleeping']:.1f}% "
              f"of monitored time", flush=True)
    print(f"  sleep episodes  : {head['sleep_episodes']}, longest "
          f"{head['longest_sleep_s']:.0f}s", flush=True)
    mean_cov = report["coverage"]["mean_coverage"]
    if mean_cov is not None:
        print(f"  mean coverage   : {mean_cov * 100:.0f}% of time readable",
              flush=True)
    print(f"\n  report : {html_path}", flush=True)
    print(f"  data   : {json_path}", flush=True)
    return html_path


def main():
    args = parse_args()

    print("=" * 66, flush=True)
    print("ClassSense AI - live detection", flush=True)
    print("Attentive (green) | Sleepy (orange) | Distracted (red) | "
          "Unknown / Unmonitored (gray)", flush=True)
    print("=" * 66, flush=True)

    os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)

    # ── what this machine can actually do ──────
    cap = cap_mod.load()
    pool_size = args.pool or cap.pool_size
    yolo_width = args.yolo_width or cap.yolo_width
    detect_every = args.detect_every or cap.detect_every
    backend = args.backend or cap.backend
    max_students = args.students or cap.max_students()

    print(cap.summary(), flush=True)
    if max_students <= 0:
        print("\nThis machine cannot meet the fidelity budget. Refusing to "
              "start rather than report states nothing supports.", flush=True)
        print("Run: python scripts/calibrate.py   to see the options.",
              flush=True)
        return 1
    print(flush=True)

    classifier, scaler, threshold = (None, None, 0.5)
    if not args.no_model:
        classifier, scaler, threshold = load_model(
            allow_weak=args.use_weak_model
        )

    source_arg = args.source if args.source else args.camera
    source = FrameSource(source_arg, loop=args.loop).start()
    actual_w, actual_h = source.actual_size
    print(f"Source ready at {actual_w}x{actual_h}.", flush=True)

    # A camera that gave less than asked changes the resolution ceiling, and
    # therefore the number of students that can be read at all.
    if (actual_w, actual_h) != (config.CAPTURE_WIDTH, config.CAPTURE_HEIGHT):
        by_pixels = cap.resolution_ceiling(actual_w, actual_h)
        print(f"  Note: requested {config.CAPTURE_WIDTH}x{config.CAPTURE_HEIGHT}. "
              f"At {actual_w}x{actual_h} the camera can resolve about "
              f"{by_pixels} students.", flush=True)
        max_students = min(max_students, by_pixels)

    worker = AnalysisWorker(
        classifier=classifier, scaler=scaler, threshold=threshold,
        pool_size=pool_size, yolo_width=yolo_width,
        max_per_cycle=max_students, detect_every=detect_every,
        refuse_beyond_capacity=not args.allow_over_capacity,
        backend=backend,
    ).start(source)

    recorder = None if args.no_record else SessionRecorder()
    if recorder is not None:
        print("Recording session. Report written on exit; live at /report "
              "when serving.", flush=True)

    provider, httpd = None, None
    if args.serve:
        provider = StreamProvider(stream_fps=args.stream_fps,
                                  recorder=recorder)
        httpd = serve(provider, port=args.serve)
        print(f"\nServing on http://0.0.0.0:{args.serve}  "
              f"(/, /stream, /status, /snapshot, /report)", flush=True)
        print("  No authentication - keep this on a trusted network.",
              flush=True)

    show_debug = args.debug
    window = "ClassSense AI"
    if not args.headless:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1280, 720)
        print("\nQ quit   S snapshot   D debug HUD\n", flush=True)
    else:
        print("\nHeadless. Ctrl-C to stop.\n", flush=True)

    stopping = {"now": False}

    def _stop(_sig, _frm):
        stopping["now"] = True

    signal.signal(signal.SIGINT, _stop)
    try:
        signal.signal(signal.SIGTERM, _stop)     # systemd stop
    except (AttributeError, ValueError):
        pass

    fps_ema = None
    last_t = time.time()
    deadline = (time.time() + args.duration) if args.duration else None

    try:
        while source.running and not stopping["now"]:
            if deadline is not None and time.time() >= deadline:
                print(f"\nReached --duration {args.duration:.0f}s.", flush=True)
                break
            frame = source.read()
            if frame is None:
                break

            now = time.time()
            dt = now - last_t
            last_t = now
            inst = 1.0 / dt if dt > 0 else 0.0
            fps_ema = inst if fps_ema is None else fps_ema * 0.9 + inst * 0.1

            students, counts, tracked, readable = worker.snapshot()
            over = worker.over_capacity

            # Fed from the same snapshot the renderer uses, so recording can
            # never perturb the analysis thread or contend for its lock.
            if recorder is not None:
                recorder.observe(students, now)

            # Drawing is not free on a Pi. Skip it entirely when nobody can
            # see the result, so those cycles go to analysis instead.
            drawing = (not args.headless) or provider is not None
            if drawing:
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
                elif over:
                    draw_banner(
                        frame,
                        f"{over} student(s) beyond capacity - reported Unmonitored",
                    )
                elif crowded:
                    draw_banner(
                        frame,
                        "Crowded view: only students needing attention are labelled",
                    )

            if provider is not None:
                merge_rate = (worker.merged_total / worker.cycles
                              if worker.cycles else 0.0)
                provider.publish(
                    frame,
                    build_status(counts, students, readable, worker, cap,
                                 over, fps_ema, max_students, merge_rate),
                )

            if args.headless:
                # Nothing to pump, and no waitKey to pace the loop. Sleep so
                # the render thread does not spin a core for no reason.
                time.sleep(max(0.0, (1.0 / max(1, args.stream_fps)) / 2))
                continue

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
        pass
    finally:
        worker.stop()
        source.stop()
        if httpd is not None:
            httpd.shutdown()
        if not args.headless:
            cv2.destroyAllWindows()

    if worker.cycles:
        print(f"\n{worker.cycles} analysis cycles, "
              f"{worker.cycle_ms:.0f}ms each "
              f"({worker.analysis_fps:.1f}/s).", flush=True)

    if recorder is not None:
        write_report(recorder, args.report_dir)

    print("Stopped.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
