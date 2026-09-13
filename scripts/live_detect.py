# scripts/live_detect.py
# Stage 5 - Live detection: YOLO + MediaPipe + Temporal Tracker (Sleepy, Attentive, Distracted)
# Scales to 60+ students via parallel MediaPipe thread pool and round-robin batching
# Press Q to quit, S to save snapshot, D to toggle debug HUD

import cv2
import numpy as np
import joblib
import os
import time
import threading
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Paths ──────────────────────────────────────
MODEL_PATH   = os.path.join("models", "engagement_classifier.pkl")
SCALER_PATH  = os.path.join("models", "scaler.pkl")
THRESH_PATH  = os.path.join("models", "threshold.pkl")
MP_MODEL     = os.path.join("scripts", "face_landmarker.task")
SNAPSHOT_DIR = os.path.join("data", "snapshots")
# ───────────────────────────────────────────────

# ── Landmark indices ───────────────────────────
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]
MOUTH      = [61, 291, 13, 14]
NOSE_TIP   = 4
FOREHEAD   = 10
CHIN       = 152
LEFT_FACE  = 234
RIGHT_FACE = 454

# ── Colours (BGR) ──────────────────────────────
GREEN  = (0, 220, 0)      # Attentive
ORANGE = (0, 140, 255)    # Sleepy / Drowsy
RED    = (0, 0, 220)      # Distracted
YELLOW = (0, 220, 220)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
BLUE   = (220, 100, 0)
GRAY   = (50, 50, 50)
DARK_GRAY = (25, 25, 25)

# ── Physiological & Temporal Thresholds ────────
EAR_CLOSED_THRESH      = 0.20   # Eye aspect ratio below this is considered closed
MAR_YAWN_THRESH        = 0.52   # Mouth aspect ratio above this indicates yawning

# Head Pose & Posture Thresholds (in calibrated degrees)
YAW_DISTRACTED_THRESH  = 18.0   # |deg|: Turned head away from screen
ROLL_RECLINED_THRESH   = 20.0   # |deg|: Head tilted sideways / lying down on shoulder or bed
PITCH_RECLINED_THRESH  = -15.0  # deg: Chin up / head tilted back / reclining in chair
PITCH_NOD_THRESH       = 18.0   # deg: Chin down / head nodding off / looking at lap

# Object & Face Confidence Gates (Prevents furniture/wood from being detected)
YOLO_CONF_THRESH       = 0.35   # Minimum YOLO person confidence (lowered for distant students)
MP_FACE_CONF_THRESH    = 0.45   # Minimum MediaPipe face confidence

# Durations in seconds required to confirm a state
SLEEPY_EYES_DURATION   = 1.2    # Sustained closed eyes (blinks < 0.4s are ignored)
YAWN_DURATION          = 1.5    # Sustained open mouth
DISTRACTED_DURATION    = 0.8    # Sustained turned head or lying down
FACE_LOST_DURATION     = 0.5    # Confirmed person turning face completely away
SMOOTHING_WINDOW       = 5      # Rolling buffer for moving average

# ── Scaling Parameters (for 60+ students) ──────
MP_POOL_SIZE           = 6      # Number of parallel MediaPipe detector instances
MIN_FACE_WIDTH         = 45    # Minimum crop width (px) to attempt landmark detection
MIN_FACE_HEIGHT        = 55    # Minimum crop height (px) to attempt landmark detection
BATCH_SIZE             = 12    # Students processed per inference cycle (round-robin)
INFER_EVERY            = 4     # Frames between inference cycles
YOLO_INPUT_WIDTH       = 640   # Downscale frame for YOLO (faster, still detects people)
GRAY   = (140, 140, 140)       # Unknown / Too far state color (overrides earlier GRAY def)

# ── Helper functions ───────────────────────────
def download_mp_model():
    if not os.path.exists(MP_MODEL):
        print("Downloading MediaPipe model...", flush=True)
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        urllib.request.urlretrieve(url, MP_MODEL)
        print("Done.", flush=True)
    else:
        print("MediaPipe model found.", flush=True)

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    v1  = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2  = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    h1  = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return round((v1 + v2) / (2.0 * h1 + 1e-6), 4)

def mouth_aspect_ratio(landmarks, w, h):
    top    = np.array([landmarks[MOUTH[2]].x * w, landmarks[MOUTH[2]].y * h])
    bottom = np.array([landmarks[MOUTH[3]].x * w, landmarks[MOUTH[3]].y * h])
    left   = np.array([landmarks[MOUTH[0]].x * w, landmarks[MOUTH[0]].y * h])
    right  = np.array([landmarks[MOUTH[1]].x * w, landmarks[MOUTH[1]].y * h])
    return round(
        np.linalg.norm(top - bottom) / (np.linalg.norm(left - right) + 1e-6), 4
    )

def head_pose_angles(landmarks, w, h):
    nose          = landmarks[NOSE_TIP]
    face_centre_x = (landmarks[LEFT_FACE].x + landmarks[RIGHT_FACE].x) / 2
    face_centre_y = (landmarks[FOREHEAD].y  + landmarks[CHIN].y)        / 2
    raw_yaw   = round((nose.x - face_centre_x) * 100, 4)
    raw_pitch = round((nose.y - face_centre_y) * 100, 4)

    # Calibrated scale-invariant angles (in degrees):
    # 1. Yaw: Ratio of distance from nose to cheeks (left vs right)
    d_left  = abs(nose.x - landmarks[LEFT_FACE].x)
    d_right = abs(nose.x - landmarks[RIGHT_FACE].x)
    yaw_ratio = (d_left - d_right) / (d_left + d_right + 1e-6)
    yaw_deg   = round(yaw_ratio * 70.0, 2)

    # 2. Pitch: Ratio of distance from nose to forehead vs chin (up vs down)
    d_forehead = abs(nose.y - landmarks[FOREHEAD].y)
    d_chin     = abs(nose.y - landmarks[CHIN].y)
    pitch_ratio = (d_forehead - d_chin) / (d_forehead + d_chin + 1e-6)
    pitch_deg   = round(pitch_ratio * 60.0, 2)

    # 3. Roll: Angle of eye-line from horizontal (tilt sideways / lying down)
    dx = (landmarks[263].x - landmarks[33].x) * w
    dy = (landmarks[263].y - landmarks[33].y) * h
    roll_deg = round(float(np.degrees(np.arctan2(dy, dx))), 2)

    return raw_yaw, raw_pitch, yaw_deg, pitch_deg, roll_deg

def compute_box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# ── Thread-safe MediaPipe Detector Pool ────────
class MediaPipePool:
    """
    Manages a fixed pool of MediaPipe FaceLandmarker instances so that multiple
    threads can run face detection concurrently without sharing state.
    Each thread checks out a detector, uses it, then returns it to the pool.
    """
    def __init__(self, pool_size: int, mp_model_path: str, face_conf: float):
        self._semaphore = threading.Semaphore(pool_size)
        self._lock      = threading.Lock()
        self._pool      = []

        base_opts = mp_python.BaseOptions(model_asset_path=mp_model_path)
        options   = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            num_faces=1,
            min_face_detection_confidence=face_conf,
            min_face_presence_confidence=face_conf,
            min_tracking_confidence=face_conf,
        )
        for _ in range(pool_size):
            self._pool.append(mp_vision.FaceLandmarker.create_from_options(options))
        print(f"MediaPipe pool: {pool_size} detectors ready.", flush=True)

    def detect(self, mp_image):
        """Acquire a free detector, run detection, release it back to the pool."""
        self._semaphore.acquire()
        with self._lock:
            detector = self._pool.pop()
        try:
            result = detector.detect(mp_image)
        finally:
            with self._lock:
                self._pool.append(detector)
            self._semaphore.release()
        return result

# ── Student Tracking & Temporal State ──────────
class TrackedStudent:
    def __init__(self, track_id, box):
        self.track_id = track_id
        self.box = box
        self.created_at = time.time()
        self.last_seen = self.created_at
        
        # Face verification gates (prevents furniture/wood from being detected as a student)
        self.face_confirmed = False
        self.face_hit_count = 0

        # Temporal event start timestamps
        self.eye_closed_start  = None
        self.yawn_start        = None
        self.distracted_start  = None
        self.face_lost_start   = None

        # Rolling buffers for smooth metric tracking
        self.ear_hist   = deque(maxlen=SMOOTHING_WINDOW)
        self.mar_hist   = deque(maxlen=SMOOTHING_WINDOW)
        self.yaw_hist   = deque(maxlen=SMOOTHING_WINDOW)
        self.pitch_hist = deque(maxlen=SMOOTHING_WINDOW)
        self.roll_hist  = deque(maxlen=SMOOTHING_WINDOW)

        # Classification results
        self.state        = "Attentive"
        self.state_reason = "Normal"
        self.confidence   = 0.90
        self.color        = GREEN
        self.dist_reason  = ""     # Stores sub-reason for Distracted label

        # Live telemetry (calibrated degrees)
        self.curr_ear   = 0.28
        self.curr_mar   = 0.02
        self.curr_yaw   = 0.0    # raw normalised
        self.curr_pitch = 0.0    # raw normalised
        self.curr_yaw_deg   = 0.0
        self.curr_pitch_deg = 0.0
        self.curr_roll_deg  = 0.0
        self.closed_sec = 0.0
        self.yawn_sec   = 0.0
        self.dist_sec   = 0.0

    def update_with_landmarks(self, landmarks, crop_w, crop_h, classifier, scaler, threshold):
        now = time.time()
        self.last_seen = now
        self.face_lost_start = None
        self.face_hit_count += 1
        if self.face_hit_count >= 2:
            self.face_confirmed = True

        # Compute instant features
        left_ear   = eye_aspect_ratio(landmarks, LEFT_EYE,  crop_w, crop_h)
        right_ear  = eye_aspect_ratio(landmarks, RIGHT_EYE, crop_w, crop_h)
        instant_ear = (left_ear + right_ear) / 2.0
        instant_mar = mouth_aspect_ratio(landmarks, crop_w, crop_h)
        raw_yaw, raw_pitch, yaw_deg, pitch_deg, roll_deg = head_pose_angles(landmarks, crop_w, crop_h)

        # Update smoothing buffers
        self.ear_hist.append(instant_ear)
        self.mar_hist.append(instant_mar)
        self.yaw_hist.append(yaw_deg)
        self.pitch_hist.append(pitch_deg)
        self.roll_hist.append(roll_deg)

        self.curr_ear       = float(np.mean(self.ear_hist))
        self.curr_mar       = float(np.mean(self.mar_hist))
        self.curr_yaw       = raw_yaw
        self.curr_pitch     = raw_pitch
        self.curr_yaw_deg   = float(np.mean(self.yaw_hist))
        self.curr_pitch_deg = float(np.mean(self.pitch_hist))
        self.curr_roll_deg  = float(np.mean(self.roll_hist))

        # 1. Check Eye Closure Duration (Sleepy / Microsleep)
        if instant_ear < EAR_CLOSED_THRESH:
            if self.eye_closed_start is None:
                self.eye_closed_start = now
            self.closed_sec = now - self.eye_closed_start
        else:
            self.eye_closed_start = None
            self.closed_sec = 0.0

        # 2. Check Yawning Duration (Sleepy)
        if instant_mar > MAR_YAWN_THRESH:
            if self.yawn_start is None:
                self.yawn_start = now
            self.yawn_sec = now - self.yawn_start
        else:
            self.yawn_start = None
            self.yawn_sec = 0.0

        # 3. Check Distraction: head turned (yaw), reclined/lying down (pitch up), or tilted sideways (roll)
        is_turned     = abs(self.curr_yaw_deg) > YAW_DISTRACTED_THRESH
        is_reclined   = self.curr_pitch_deg < PITCH_RECLINED_THRESH        # chin up / lying back
        is_tilted     = abs(self.curr_roll_deg) > ROLL_RECLINED_THRESH     # head sideways/lying
        is_nodding_down = self.curr_pitch_deg > PITCH_NOD_THRESH and self.closed_sec < 0.3  # chin to chest

        is_distracted = is_turned or is_reclined or is_tilted or is_nodding_down
        if is_distracted:
            # Build sub-reason string
            reasons = []
            if is_turned:        reasons.append(f"Turned {self.curr_yaw_deg:+.0f}\u00b0")
            if is_reclined:      reasons.append(f"Reclined ({self.curr_pitch_deg:+.0f}\u00b0)")
            if is_tilted:        reasons.append(f"Tilted ({self.curr_roll_deg:+.0f}\u00b0)")
            if is_nodding_down:  reasons.append("Head Down")
            self.dist_reason = ", ".join(reasons)

            if self.distracted_start is None:
                self.distracted_start = now
            self.dist_sec = now - self.distracted_start
        else:
            self.distracted_start = None
            self.dist_sec = 0.0
            self.dist_reason = ""

        # ── State Arbitration ───────────────────────
        # Priority 1: Sleepy (Eyes closed >= 1.2s or Yawning >= 1.5s or head drooping with eyes closing)
        if self.closed_sec >= SLEEPY_EYES_DURATION:
            self.state = "Sleepy"
            self.state_reason = f"Eyes Closed ({self.closed_sec:.1f}s)"
            self.confidence = min(0.99, 0.75 + (self.closed_sec * 0.1))
            self.color = ORANGE

        elif self.yawn_sec >= YAWN_DURATION:
            self.state = "Sleepy"
            self.state_reason = f"Yawning ({self.yawn_sec:.1f}s)"
            self.confidence = 0.90
            self.color = ORANGE

        elif self.closed_sec >= 0.8 and self.curr_pitch_deg > PITCH_NOD_THRESH:
            self.state = "Sleepy"
            self.state_reason = "Head Nodding Off"
            self.confidence = 0.88
            self.color = ORANGE

        # Priority 2: Distracted (any postural signal sustained >= DISTRACTED_DURATION)
        elif self.dist_sec >= DISTRACTED_DURATION:
            self.state = "Distracted"
            self.state_reason = self.dist_reason
            self.confidence = min(0.98, 0.70 + (self.dist_sec * 0.1))
            self.color = RED

        # Priority 3: Momentary Eye Blink (< 1.2s) -> do NOT flicker state
        elif self.closed_sec > 0.0:
            if self.state != "Sleepy":
                self.state = "Attentive"
                self.state_reason = "Blinking"
                self.color = GREEN
                self.confidence = 0.85

        # Priority 4: Eyes open & posture OK -> ML classifier for subtle engagement
        else:
            feat = np.array([[left_ear, right_ear, self.curr_ear, self.curr_mar,
                              self.curr_yaw_deg, self.curr_pitch_deg]])
            feat_scaled = scaler.transform(feat)
            proba = classifier.predict_proba(feat_scaled)[0]
            dist_prob = proba[0]
            att_prob  = proba[1]

            effective_thresh = max(0.60, threshold)
            if dist_prob >= effective_thresh:
                self.state = "Distracted"
                self.state_reason = "Inattentive"
                self.confidence = dist_prob
                self.color = RED
            else:
                self.state = "Attentive"
                self.state_reason = "Engaged"
                self.confidence = att_prob
                self.color = GREEN

    def update_without_landmarks(self):
        now = time.time()
        self.last_seen = now
        if not self.face_confirmed:
            return  # Inanimate object with no confirmed human face: do nothing

        if self.face_lost_start is None:
            self.face_lost_start = now
        lost_dur = now - self.face_lost_start

        if lost_dur >= FACE_LOST_DURATION:
            self.state = "Distracted"
            self.state_reason = "Looking Away"
            self.confidence = 0.85
            self.color = RED

class StudentTracker:
    def __init__(self):
        self.students = {}
        self.next_id = 1

    def update(self, detected_boxes):
        now = time.time()
        matched_ids = set()

        # Match detected boxes to existing students via IoU or distance
        for box in detected_boxes:
            best_iou = 0.0
            best_id  = None

            for sid, student in self.students.items():
                if sid in matched_ids:
                    continue
                iou = compute_box_iou(box, student.box)
                if iou > best_iou:
                    best_iou = iou
                    best_id = sid

            if best_id is not None and best_iou >= 0.20:
                self.students[best_id].box = box
                matched_ids.add(best_id)
            else:
                # New student detected
                new_student = TrackedStudent(self.next_id, box)
                self.students[self.next_id] = new_student
                matched_ids.add(self.next_id)
                self.next_id += 1

        # Purge stale tracks (not seen for > 2.0s, or never had a confirmed face for > 1.0s)
        stale_ids = [
            sid for sid, s in self.students.items()
            if (now - s.last_seen > 2.0) or (not s.face_confirmed and (now - s.created_at > 1.0))
        ]
        for sid in stale_ids:
            del self.students[sid]

        return self.students

# ── Visual Rendering ───────────────────────────
def draw_label_box(frame, student, show_debug=False):
    x1, y1, x2, y2 = student.box
    color = student.color
    label = student.state
    conf  = student.confidence
    reason = student.state_reason

    # Bounding Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Top Tag: State + Confidence
    top_text = f"{label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(top_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    tag_y1 = max(0, y1 - th - 10)
    cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, top_text, (x1 + 5, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2)

    # Bottom Sub-Tag: State Reason
    (rw, rh), _ = cv2.getTextSize(reason, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(frame, (x1, y2), (x1 + rw + 8, y2 + rh + 8), color, -1)
    cv2.putText(frame, reason, (x1 + 4, y2 + rh + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1)

    # Telemetry Debug Overlay (Key 'D')
    if show_debug:
        yaw_flag   = "TURNED" if abs(student.curr_yaw_deg) > YAW_DISTRACTED_THRESH else "OK"
        pitch_flag = "RECLINED" if student.curr_pitch_deg < PITCH_RECLINED_THRESH else ("NOD" if student.curr_pitch_deg > PITCH_NOD_THRESH else "OK")
        roll_flag  = "TILTED" if abs(student.curr_roll_deg) > ROLL_RECLINED_THRESH else "OK"
        dbg_lines = [
            f"EAR: {student.curr_ear:.2f} ({'CLOSED' if student.curr_ear < EAR_CLOSED_THRESH else 'OPEN'})",
            f"MAR: {student.curr_mar:.2f} ({'YAWN' if student.curr_mar > MAR_YAWN_THRESH else 'OK'})",
            f"Yaw:   {student.curr_yaw_deg:+5.1f}deg  [{yaw_flag}]",
            f"Pitch: {student.curr_pitch_deg:+5.1f}deg  [{pitch_flag}]",
            f"Roll:  {student.curr_roll_deg:+5.1f}deg  [{roll_flag}]",
            f"EyeClose:{student.closed_sec:.1f}s  Dist:{student.dist_sec:.1f}s",
        ]
        dy = y1 + 18
        for line in dbg_lines:
            cv2.putText(frame, line, (x1 + 6, dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, BLACK, 3)
            cv2.putText(frame, line, (x1 + 6, dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, YELLOW, 1)
            dy += 15

def draw_dashboard(frame, students, fps, show_debug):
    h, w = frame.shape[:2]
    total      = len(students)
    attentive  = sum(1 for s in students.values() if s.state == "Attentive")
    sleepy     = sum(1 for s in students.values() if s.state == "Sleepy")
    distracted = sum(1 for s in students.values() if s.state == "Distracted")
    engagement = (attentive / total * 100) if total > 0 else 0

    # Semi-transparent background panel
    panel_w, panel_h = 280, 205
    px, py = w - panel_w - 12, 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), BLACK, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Title
    cv2.putText(frame, "ClassSense AI",
                (px + 10, py + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, BLUE, 2)

    # Stats Rows
    cv2.putText(frame, f"Students   : {total}",
                (px + 10, py + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1)
    cv2.putText(frame, f"Attentive  : {attentive}",
                (px + 10, py + 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, GREEN, 1)
    cv2.putText(frame, f"Sleepy     : {sleepy}",
                (px + 10, py + 96),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, ORANGE, 1)
    cv2.putText(frame, f"Distracted : {distracted}",
                (px + 10, py + 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, RED, 1)

    # Engagement Bar
    bx, by = px + 10, py + 134
    bw = panel_w - 20
    cv2.rectangle(frame, (bx, by), (bx + bw, by + 14), (50, 50, 50), -1)
    fill = int(bw * engagement / 100)
    bar_color = GREEN if engagement >= 70 else YELLOW if engagement >= 40 else RED
    if fill > 0:
        cv2.rectangle(frame, (bx, by), (bx + fill, by + 14), bar_color, -1)
    cv2.putText(frame, f"{engagement:.0f}% class engagement",
                (bx, by + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1)

    # Key Help & HUD status
    hud_txt = "[D] HUD: ON" if show_debug else "[D] HUD: OFF"
    cv2.putText(frame, f"[Q] Quit   [S] Snap   {hud_txt}",
                (px + 10, py + 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, YELLOW if show_debug else WHITE, 1)

    # FPS Counter
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

# ── Main ───────────────────────────────────────
def main():
    print("=" * 50, flush=True)
    print("Starting ClassSense AI Live Detection...", flush=True)
    print("Supports: Attentive (Green) | Sleepy (Orange) | Distracted (Red)", flush=True)
    print("=" * 50, flush=True)

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    download_mp_model()

    # Load trained model & scaler
    print("Loading trained model...", flush=True)
    classifier = joblib.load(MODEL_PATH)
    scaler     = joblib.load(SCALER_PATH)
    threshold  = joblib.load(THRESH_PATH)
    print(f"Model loaded. Decision threshold: {threshold:.2f}", flush=True)

    # Load YOLO
    print("Loading YOLOv8...", flush=True)
    yolo = YOLO("yolov8n.pt")
    print("YOLO ready.", flush=True)

    # Load MediaPipe Face Landmarker
    print("Loading MediaPipe Face Landmarker...", flush=True)
    base_options  = mp_python.BaseOptions(model_asset_path=MP_MODEL)
    options       = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=10,
        min_face_detection_confidence=MP_FACE_CONF_THRESH,
        min_face_presence_confidence=MP_FACE_CONF_THRESH,
        min_tracking_confidence=MP_FACE_CONF_THRESH,
    )
    face_detector = mp_vision.FaceLandmarker.create_from_options(options)
    print("MediaPipe ready.", flush=True)

    # Open webcam
    print("Opening camera feed...", flush=True)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("ERROR: Could not open camera.", flush=True)
        return

    print("\nControls:", flush=True)
    print("  Press 'Q' to quit", flush=True)
    print("  Press 'S' to save snapshot", flush=True)
    print("  Press 'D' to toggle debug telemetry HUD\n", flush=True)

    tracker      = StudentTracker()
    INFER_EVERY  = 3  # Infer every 3 frames for responsive temporal tracking
    frame_count  = 0
    fps_timer    = time.time()
    fps          = 0.0
    show_debug   = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.", flush=True)
            break

        frame_count += 1
        now       = time.time()
        fps       = 1.0 / (now - fps_timer + 1e-6)
        fps_timer = now

        h, w = frame.shape[:2]

        if frame_count % INFER_EVERY == 0:
            # Step 1: Detect persons with YOLO
            yolo_out = yolo(frame, classes=[0], verbose=False)[0]
            detected_boxes = []

            for box in yolo_out.boxes:
                conf = float(box.conf[0])
                if conf < YOLO_CONF_THRESH:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_boxes.append((x1, y1, x2, y2))

            # Step 2: Associate with tracker
            active_students = tracker.update(detected_boxes)

            # Step 3: Extract MediaPipe landmarks for each student
            for student in active_students.values():
                x1, y1, x2, y2 = student.box
                pad = 20
                x1p = max(0, x1 - pad)
                y1p = max(0, y1 - pad)
                x2p = min(w, x2 + pad)
                y2p = min(h, y2 + pad)
                crop = frame[y1p:y2p, x1p:x2p]

                if crop.size == 0:
                    student.update_without_landmarks()
                    continue

                crop_h, crop_w = crop.shape[:2]
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                mp_res = face_detector.detect(mp_image)

                if not mp_res.face_landmarks:
                    student.update_without_landmarks()
                else:
                    landmarks = mp_res.face_landmarks[0]
                    student.update_with_landmarks(
                        landmarks, crop_w, crop_h,
                        classifier, scaler, threshold
                    )

        # Draw only confirmed human students (filters out furniture/wood/inanimate objects)
        confirmed_students = {sid: s for sid, s in tracker.students.items() if s.face_confirmed}
        for student in confirmed_students.values():
            draw_label_box(frame, student, show_debug=show_debug)

        # Draw dashboard overlay
        draw_dashboard(frame, confirmed_students, fps, show_debug=show_debug)

        cv2.imshow("ClassSense AI - Live Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            print("Quitting...", flush=True)
            break
        elif key == ord("s") or key == ord("S"):
            snap = os.path.join(SNAPSHOT_DIR, f"snap_{int(time.time())}.jpg")
            cv2.imwrite(snap, frame)
            print(f"Snapshot saved: {snap}", flush=True)
        elif key == ord("d") or key == ord("D"):
            show_debug = not show_debug
            print(f"Debug telemetry HUD: {'ON' if show_debug else 'OFF'}", flush=True)

    cap.release()
    cv2.destroyAllWindows()
    print("Live detection ended cleanly.", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("CRASH:", e, flush=True)
        traceback.print_exc()
        input("Press Enter to close...")