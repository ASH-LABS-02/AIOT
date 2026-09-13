# classsense/pipeline.py
# Thread orchestration.
#
# The old design ran capture, inference and drawing in one loop, so the window
# froze for the whole of every inference pass and the visible framerate was the
# inference rate. Here the three run independently:
#
#   capture   - owns the camera, always holds the newest frame and discards the
#               rest, so latency cannot accumulate behind a slow analysis pass
#   analysis  - YOLO, then landmark extraction fanned across the detector pool,
#               then state updates; runs as fast as it can and no faster
#   render    - the caller's thread, drawing the newest frame against the most
#               recent known states at full camera framerate
#
# Consequence worth knowing: boxes lag the image by up to one analysis cycle
# (~230ms at 60 students). For seated students that is invisible. It would not
# be acceptable for fast motion, which this is not built for.

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from classsense.config import (
    YOLO_CONF_THRESH, YOLO_INPUT_WIDTH, MP_POOL_SIZE, MP_FACE_CONF_THRESH,
    MAX_STUDENTS_PER_CYCLE, CROP_PADDING, CAPTURE_WIDTH, CAPTURE_HEIGHT,
    CAPTURE_INDEX, YOLO_WEIGHTS,
)
from classsense.geometry import extract_feature_row, face_width_px
from classsense.mp_pool import MediaPipePool, prepare_crop
from classsense.tiers import tier_for_face_width, Tier
from classsense.tracker import StudentTracker


class FrameSource:
    """
    Camera or video file on its own thread, holding only the newest frame.

    Dropping frames is the point. A queue would let the analysis thread fall
    progressively further behind real time; keeping exactly one slot means a
    slow pass costs freshness, never latency.
    """

    def __init__(self, source=CAPTURE_INDEX, width=CAPTURE_WIDTH,
                 height=CAPTURE_HEIGHT, loop=False):
        self.source = source
        self.loop = loop
        self._cap = cv2.VideoCapture(source)
        if isinstance(source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source!r}")

        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        self.frames_read = 0
        self.actual_size = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def start(self):
        self._thread.start()
        # Block until the first frame lands, so callers never see None.
        deadline = time.time() + 5.0
        while self.read() is None and time.time() < deadline:
            time.sleep(0.01)
        if self.read() is None:
            raise RuntimeError("Video source opened but produced no frames.")
        return self

    def _run(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                if self.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self._stop.set()
                break
            with self._lock:
                self._frame = frame
                self.frames_read += 1

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    @property
    def running(self):
        return not self._stop.is_set()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cap.release()


class AnalysisWorker:
    """
    YOLO detection, parallel landmark extraction, and state updates.

    One tracker lives here and is read by the render thread, so every mutation
    is under `lock`. The expensive work - YOLO and the landmark fan-out -
    happens outside that lock; only the short state-update pass holds it.
    """

    def __init__(self, classifier=None, scaler=None, threshold=0.5,
                 pool_size=MP_POOL_SIZE, yolo_width=YOLO_INPUT_WIDTH,
                 max_per_cycle=MAX_STUDENTS_PER_CYCLE):
        from ultralytics import YOLO

        self.yolo = YOLO(YOLO_WEIGHTS)
        self.pool = MediaPipePool(pool_size, face_conf=MP_FACE_CONF_THRESH)
        self.executor = ThreadPoolExecutor(max_workers=pool_size)

        self.classifier = classifier
        self.scaler = scaler
        self.threshold = threshold
        self.yolo_width = yolo_width
        self.max_per_cycle = max_per_cycle

        self.tracker = StudentTracker()
        self.lock = threading.Lock()

        self.cycle_ms = 0.0
        self.analysis_fps = 0.0
        self.cycles = 0
        self._stop = threading.Event()
        self._thread = None

    def start(self, source):
        self._thread = threading.Thread(
            target=self._run, args=(source,), daemon=True
        )
        self._thread.start()
        return self

    def _run(self, source):
        ema = None
        while not self._stop.is_set() and source.running:
            frame = source.read()
            if frame is None:
                time.sleep(0.005)
                continue

            t0 = time.perf_counter()
            self.analyse(frame)
            elapsed = (time.perf_counter() - t0) * 1000.0

            # Smoothed so the HUD reports a rate rather than flickering.
            ema = elapsed if ema is None else ema * 0.8 + elapsed * 0.2
            self.cycle_ms = ema
            self.analysis_fps = 1000.0 / ema if ema > 0 else 0.0
            self.cycles += 1

    def analyse(self, frame):
        """One full analysis cycle over the given frame."""
        now = time.time()
        boxes = self.detect_people(frame)

        with self.lock:
            self.tracker.update(boxes, now)
            batch = self.tracker.schedule(self.max_per_cycle, now)

        # Landmark extraction, fanned across the pool. This is the parallel
        # section and the reason 60 students fit in a cycle: detect() releases
        # the GIL, measured at 3.21x over 8 threads.
        jobs = []
        for student in batch:
            prepared = prepare_crop(frame, student.box, CROP_PADDING)
            if prepared is None:
                jobs.append((student, None))
                continue
            jobs.append((student, prepared))

        futures = {}
        for student, prepared in jobs:
            if prepared is None:
                continue
            image, cw, ch, scale = prepared
            futures[self.executor.submit(self.pool.detect, image)] = (
                student, cw, ch, scale
            )

        results = []
        for future, (student, cw, ch, scale) in futures.items():
            try:
                landmarks = future.result()
            except Exception:
                landmarks = None
            results.append((student, landmarks, cw, ch, scale))

        # Short critical section: only the state mutation is serialised.
        with self.lock:
            for student, landmarks, cw, ch, scale in results:
                student.last_analysed = now
                if landmarks is None:
                    student.engagement.mark_face_lost(student.face_confirmed, now)
                    continue

                student.note_face_hit()
                # Undo the crop downscale before tiering. The tier has to
                # describe how many pixels the camera actually put on this
                # face, not how many survived prepare_crop's 192px cap.
                width = face_width_px(landmarks, cw) / max(scale, 1e-6)
                tier = tier_for_face_width(width)
                features = extract_feature_row(landmarks, cw, ch)
                student.engagement.update(features, tier, now=now)

            # Students whose crop was degenerate never reached the pool. Mark
            # them analysed anyway, or their priority stays pinned high and
            # they crowd everyone else out of the schedule.
            for student, prepared in jobs:
                if prepared is None:
                    student.last_analysed = now
                    student.engagement.mark_face_lost(student.face_confirmed, now)

            for student in batch:
                student.analysis_count += 1

            self._apply_model(batch)

    def _apply_model(self, batch):
        """
        Score every inconclusive student in one call.

        The forest is 300 trees, and calling it per student costs about 62ms
        each in Python overhead - 3.7s across 60 students, against 56ms for
        the same 60 rows batched. Per-student calls pushed the cycle past the
        track-retirement window and stopped anything from ever confirming, so
        batching here is load-bearing, not a micro-optimisation.

        Caller already holds the lock.
        """
        if self.classifier is None or self.scaler is None:
            return

        pending = [s for s in batch if s.engagement.pending_model]
        if not pending:
            return

        rows = np.array([s.engagement.model_vector() for s in pending])
        try:
            proba = self.classifier.predict_proba(self.scaler.transform(rows))
        except Exception:
            # A broken or mismatched model must not take the pipeline with it;
            # the heuristics have already set a usable state for everyone here.
            for student in pending:
                student.engagement.pending_model = False
            return

        for student, row in zip(pending, proba):
            student.engagement.apply_model(float(row[0]), self.threshold)

    def detect_people(self, frame):
        """
        YOLO person boxes above the confidence gate.

        Runs at YOLO_INPUT_WIDTH rather than native resolution. 1280 over 960
        costs ~43ms more per cycle but recovers back-row students, who shrink
        below reliable detection size once a 1080p frame is scaled to 960.
        """
        out = self.yolo(
            frame, classes=[0], imgsz=self.yolo_width, verbose=False
        )[0]

        boxes = []
        for box in out.boxes:
            if float(box.conf[0]) < YOLO_CONF_THRESH:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
        return boxes

    def snapshot(self):
        """Copy of what the render thread needs, taken under the lock."""
        with self.lock:
            confirmed = list(self.tracker.confirmed().values())
            counts = self.tracker.counts()
            tracked = len(self.tracker.students)
        readable = sum(1 for s in confirmed if s.tier >= Tier.COARSE)
        return confirmed, counts, tracked, readable

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self.executor.shutdown(wait=False)
        self.pool.close()


def make_tiled_source(frame, cells, out_size=(1920, 1080)):
    """
    Replicate one frame into a grid, to load-test with N faces.

    Not a substitute for a real classroom - every face is identical and equally
    lit - but it does exercise the genuine cost, which is N landmark passes and
    a YOLO frame with N people in it. Used by scripts/stress_test.py.
    """
    cols = int(np.ceil(np.sqrt(cells)))
    rows = int(np.ceil(cells / cols))
    cell_w = out_size[0] // cols
    cell_h = out_size[1] // rows

    tile = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_size[1], out_size[0], 3), dtype=np.uint8)

    placed = 0
    for r in range(rows):
        for c in range(cols):
            if placed >= cells:
                break
            y, x = r * cell_h, c * cell_w
            canvas[y:y + cell_h, x:x + cell_w] = tile
            placed += 1
    return canvas
