# scripts/benchmark.py
# Measures the real throughput ceiling of this machine before we tune the
# live pipeline. Run from the repo root:  python scripts/benchmark.py
#
# Answers three questions that decide the live-detection architecture:
#   1. How long does one MediaPipe FaceLandmarker.detect() take on a face crop?
#   2. Does detect() release the GIL, i.e. do parallel detectors actually help?
#   3. What does YOLOv8n cost per frame at each candidate input width?

import os
import sys
import time
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MP_MODEL = os.path.join("scripts", "face_landmarker.task")
DAISEE_TRAIN = r"C:\Users\AR\Downloads\DAiSEE\DAiSEE\DataSet\Train"

WARMUP = 3
TRIALS = 30


def find_sample_face_frame():
    """Pull a real face frame from DAiSEE, else fall back to the webcam."""
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
                            print(f"Sample face frame from: {f.name}")
                            return frame
    print("DAiSEE not found, trying webcam...")
    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("No face source available (no DAiSEE, no webcam).")
    return frame


def time_calls(fn, trials=TRIALS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples), min(samples), max(samples)


def bench_mediapipe(face_bgr):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    def make_detector():
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MP_MODEL),
            num_faces=1,
            min_face_detection_confidence=0.45,
            min_face_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        return mp_vision.FaceLandmarker.create_from_options(opts)

    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    print("\n" + "=" * 64)
    print("1. MediaPipe detect() latency by crop size (single thread)")
    print("=" * 64)
    print(f"{'crop':>12} {'median ms':>11} {'min':>8} {'max':>8} {'faces':>7}")
    print("-" * 64)

    det = make_detector()
    per_size = {}
    for size in (96, 128, 192, 256, 320):
        resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(resized))
        found = len(det.detect(img).face_landmarks)
        med, lo, hi = time_calls(lambda: det.detect(img))
        per_size[size] = med
        print(f"{size:>9}px {med:>11.2f} {lo:>8.2f} {hi:>8.2f} {found:>7}")

    # Parallel scaling: the question that decides whether a pool is worth building.
    print("\n" + "=" * 64)
    print("2. Parallel scaling — does detect() release the GIL?")
    print("=" * 64)

    size = 192
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(resized))
    serial_ms = per_size[size]

    print(f"Baseline (1 thread, {size}px crop): {serial_ms:.2f} ms/detect")
    print(f"{'threads':>8} {'total ms':>10} {'ms/detect':>11} {'speedup':>9} {'efficiency':>11}")
    print("-" * 64)

    results = {}
    for n_threads in (1, 2, 4, 6, 8, 12):
        detectors = [make_detector() for _ in range(n_threads)]
        work_per_thread = 12

        def run_one(d):
            for _ in range(work_per_thread):
                d.detect(img)

        for d in detectors:          # warm each detector
            d.detect(img)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            list(pool.map(run_one, detectors))
        total_ms = (time.perf_counter() - t0) * 1000.0

        n_calls = n_threads * work_per_thread
        ms_each = total_ms / n_calls
        speedup = serial_ms / ms_each
        eff = speedup / n_threads * 100
        results[n_threads] = (ms_each, speedup)
        print(f"{n_threads:>8} {total_ms:>10.1f} {ms_each:>11.2f} {speedup:>8.2f}x {eff:>10.0f}%")

        for d in detectors:
            d.close()

    best_n = max(results, key=lambda k: results[k][1])
    print(f"\nBest speedup at {best_n} threads ({results[best_n][1]:.2f}x).")
    if results[best_n][1] < 1.4:
        print("VERDICT: detect() appears GIL-bound. A thread pool will NOT help;")
        print("         use process-based parallelism or reduce students per cycle.")
    else:
        print(f"VERDICT: detect() releases the GIL. Thread pool is worth it.")
        print(f"         Recommended MP_POOL_SIZE = {best_n}")

    det.close()
    return results, per_size


def bench_yolo(frame):
    from ultralytics import YOLO

    print("\n" + "=" * 64)
    print("3. YOLOv8n person detection by input width (CPU)")
    print("=" * 64)

    yolo = YOLO("yolov8n.pt")
    full = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)

    print(f"{'imgsz':>8} {'median ms':>11} {'min':>8} {'max':>8} {'persons':>9}")
    print("-" * 64)
    out = {}
    for imgsz in (480, 640, 960, 1280):
        n = len(yolo(full, classes=[0], imgsz=imgsz, verbose=False)[0].boxes)
        med, lo, hi = time_calls(
            lambda: yolo(full, classes=[0], imgsz=imgsz, verbose=False),
            trials=10, warmup=2,
        )
        out[imgsz] = med
        print(f"{imgsz:>8} {med:>11.2f} {lo:>8.2f} {hi:>8.2f} {n:>9}")
    return out


def project_budget(mp_results, mp_per_size, yolo_ms):
    """Translate the measurements into a students-per-second budget."""
    print("\n" + "=" * 64)
    print("4. Projected budget for 60 students @ 1080p")
    print("=" * 64)

    best_n = max(mp_results, key=lambda k: mp_results[k][1])
    ms_per_face = mp_results[best_n][0]
    yolo_960 = yolo_ms.get(960, 0.0)

    print(f"MediaPipe per face (pooled, {best_n} threads): {ms_per_face:.2f} ms")
    print(f"YOLO @ 960px                                : {yolo_960:.2f} ms")
    print()
    print(f"{'batch':>7} {'mp ms':>9} {'cycle ms':>10} {'cycles/s':>10} {'refresh 60':>12}")
    print("-" * 64)
    for batch in (8, 12, 16, 20, 30, 60):
        mp_ms = ms_per_face * batch
        cycle = yolo_960 + mp_ms
        cps = 1000.0 / cycle if cycle else 0
        refresh = (60 / batch) / cps if cps else float("inf")
        flag = "" if refresh <= 1.0 else "  <-- too slow"
        print(f"{batch:>7} {mp_ms:>9.1f} {cycle:>10.1f} {cps:>10.2f} {refresh:>11.2f}s{flag}")

    print("\n'refresh 60' = seconds for every one of 60 students to be re-examined.")
    print("Target <= 1.0s so a 1.2s eye-closure window still gets >= 2 samples.")


def main():
    if not os.path.exists(MP_MODEL):
        print(f"ERROR: {MP_MODEL} missing. Run from the repo root.")
        sys.exit(1)

    print("=" * 64)
    print("ClassSense throughput benchmark")
    print("=" * 64)
    print(f"CPU cores: {os.cpu_count()}")
    try:
        import torch
        print(f"torch    : {torch.__version__}  CUDA={torch.cuda.is_available()}")
    except ImportError:
        pass

    frame = find_sample_face_frame()
    print(f"Frame    : {frame.shape[1]}x{frame.shape[0]}")

    mp_results, mp_per_size = bench_mediapipe(frame)
    yolo_ms = bench_yolo(frame)
    project_budget(mp_results, mp_per_size, yolo_ms)

    print("\nDone.")


if __name__ == "__main__":
    main()
