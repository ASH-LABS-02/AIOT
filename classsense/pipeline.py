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
    CAPTURE_INDEX, DETECT_EVERY, REFUSE_BEYOND_CAPACITY,
)
from classsense.geometry import (
    extract_feature_row, face_size_px, face_centre_in_frame,
)
from classsense.detector import load_detector
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
                 max_per_cycle=MAX_STUDENTS_PER_CYCLE,
                 detect_every=DETECT_EVERY,
                 refuse_beyond_capacity=REFUSE_BEYOND_CAPACITY,
                 backend="pytorch"):
        # load_detector refuses an NCNN model whose export width does not
        # match, because that combination returns zero detections silently.
        self.yolo, self.backend = load_detector(
            backend=backend, yolo_width=yolo_width, strict=False
        )
        self.pool = MediaPipePool(pool_size, face_conf=MP_FACE_CONF_THRESH)
        self.executor = ThreadPoolExecutor(max_workers=pool_size)

        self.classifier = classifier
        self.scaler = scaler
        self.threshold = threshold
        self.yolo_width = yolo_width
        self.max_per_cycle = max_per_cycle
        self.detect_every = max(1, detect_every)
        self.refuse_beyond_capacity = refuse_beyond_capacity
        self.over_capacity = 0
        self._since_detect = 0          # 0 means "detect on the next cycle"

        self.tracker = StudentTracker()
        self.lock = threading.Lock()

        self.cycle_ms = 0.0
        self.analysis_fps = 0.0
        self.cycles = 0
        self.merged_last_cycle = 0
        self.merged_total = 0
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
        """
        One full analysis cycle over the given frame.

        Person detection runs every DETECT_EVERY cycles; landmark analysis runs
        every cycle. Seated students do not move much between passes, and YOLO
        measured 47% of a 60-student cycle spent re-finding them, while the
        signals that actually change are all read from landmarks.
        """
        now = time.time()

        run_detect = (self._since_detect <= 0)
        if run_detect:
            boxes = self.detect_people(frame)
            self._since_detect = self.detect_every - 1
        else:
            self._since_detect -= 1

        with self.lock:
            if run_detect:
                self.tracker.update(boxes, now)
            batch = self.tracker.schedule(self.max_per_cycle, now)

            # Anyone past capacity is still tracked and still counted, but is
            # reported Unmonitored rather than given a state derived from a
            # sampling rate too slow to support it.
            if self.refuse_beyond_capacity:
                scheduled = {s.track_id for s in batch}
                overflow = [
                    s for s in self.tracker.students.values()
                    if s.track_id not in scheduled and s.face_confirmed
                ]
                for student in overflow:
                    student.engagement.mark_unmonitored()
                self.over_capacity = len(overflow)

        # Landmark extraction, fanned across the pool. This is the parallel
        # section and the reason 60 students fit in a cycle: detect() releases
        # the GIL, measured at 3.21x over 8 threads.
        #
        # Cropping rides along inside the workers rather than running serially
        # first. cv2's resize and colour conversion also drop the GIL, so the
        # ~22ms of crop preparation parallelises for free - and one task per
        # student is simpler than staging the work in two passes.
        futures = {
            self.executor.submit(self._crop_and_detect, frame, student.box):
                student
            for student in batch
        }

        results = []
        for future, student in futures.items():
            try:
                outcome = future.result()
            except Exception:
                outcome = None
            results.append((student, outcome))

        # Short critical section: only the state mutation is serialised.
        with self.lock:
            for student, outcome in results:
                student.last_analysed = now
                student.analysis_count += 1

                # outcome is None when the crop was degenerate; its landmarks
                # are None when the crop was fine but held no detectable face.
                # Both mean "no reading this cycle", and both must still count
                # as an attempt or the student's priority stays pinned high and
                # crowds everyone else out of the schedule.
                if outcome is None or outcome[0] is None:
                    student.engagement.mark_face_lost(student.face_confirmed, now)
                    continue

                landmarks, cw, ch, scale = outcome
                # Undo the crop downscale before tiering. The tier has to
                # describe how many pixels the camera actually put on this
                # face, not how many survived prepare_crop's 192px cap.
                size = face_size_px(landmarks, cw, ch) / max(scale, 1e-6)
                centre = face_centre_in_frame(
                    landmarks, student.box, cw, ch, scale, CROP_PADDING
                )
                student.note_face_hit(face_xy=centre, face_size=size, now=now)
                # A student with a visible face is present, whether or not YOLO
                # ran this cycle. Without this, tracks would age toward
                # retirement through every skipped detection pass.
                student.last_seen = now

                tier = tier_for_face_width(size)
                features = extract_feature_row(landmarks, cw, ch)
                student.engagement.update(features, tier, now=now)

            # Now that every face position for this cycle is known, collapse
            # any tracks that turned out to be the same person. Must run here,
            # after the landmark pass, because it is the face positions - not
            # the boxes - that reveal a duplicate.
            self.merged_last_cycle = self.tracker.dedupe_by_face(now)
            self.merged_total += self.merged_last_cycle

        # Outside the lock on purpose. A batched predict over 60 students costs
        # ~56ms, and the render thread calls snapshot() every frame (~33ms at
        # 30fps) - holding the lock across the forest would stall drawing for
        # nearly two frames every cycle. apply_model only touches per-student
        # state the render thread reads atomically, so the brief inconsistency
        # is a student still showing last cycle's label, which is exactly what
        # every student between cycles is already showing.
        self._apply_model(batch)

    def _crop_and_detect(self, frame, box):
        """
        One student's whole landmark pass, to be run on a worker thread.

        Returns (landmarks, crop_w, crop_h, scale), with landmarks None when no
        face was found, or None outright when the box yielded no usable crop.
        """
        prepared = prepare_crop(frame, box, CROP_PADDING)
        if prepared is None:
            return None
        image, cw, ch, scale = prepared
        return self.pool.detect(image), cw, ch, scale

    def _apply_model(self, batch):
        """
        Score every inconclusive student in one call.

        The forest is 300 trees, and calling it per student costs about 62ms
        each in Python overhead - 3.7s across 60 students, against 56ms for
        the same 60 rows batched. Per-student calls pushed the cycle past the
        track-retirement window and stopped anything from ever confirming, so
        batching here is load-bearing, not a micro-optimisation.

        Called without the lock - see the note at the call site.
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
        now = time.time()
        with self.lock:
            confirmed = list(self.tracker.confirmed(now).values())
            counts = self.tracker.counts(now)
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
