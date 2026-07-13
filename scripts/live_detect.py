# scripts/live_detect.py
# Stage 5 - Live detection: YOLO + MediaPipe + trained classifier
# Press Q to quit, S to save a snapshot

import cv2
import numpy as np
import joblib
import os
import time
import urllib.request
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

# ── Colours ────────────────────────────────────
GREEN  = (0, 220, 0)
RED    = (0, 0, 220)
YELLOW = (0, 220, 220)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
BLUE   = (220, 100, 0)

# ── Helper functions ───────────────────────────
def download_mp_model():
    if not os.path.exists(MP_MODEL):
        print("Downloading MediaPipe model...")
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        urllib.request.urlretrieve(url, MP_MODEL)
        print("Done.")
    else:
        print("MediaPipe model found.")

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
    yaw   = round((nose.x - face_centre_x) * 100, 4)
    pitch = round((nose.y - face_centre_y) * 100, 4)
    return yaw, pitch

def extract_features(landmarks, w, h):
    left_ear   = eye_aspect_ratio(landmarks, LEFT_EYE,  w, h)
    right_ear  = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
    avg_ear    = round((left_ear + right_ear) / 2, 4)
    mar        = mouth_aspect_ratio(landmarks, w, h)
    yaw, pitch = head_pose_angles(landmarks, w, h)
    return np.array([[left_ear, right_ear, avg_ear, mar, yaw, pitch]])

def draw_label_box(frame, x1, y1, x2, y2, label, prob, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text        = f"{label} {prob:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2)

def draw_dashboard(frame, results, fps):
    h, w        = frame.shape[:2]
    total       = len(results)
    attentive   = sum(1 for r in results if r[0] == "Attentive")
    distracted  = total - attentive
    engagement  = (attentive / total * 100) if total > 0 else 0

    # Semi-transparent background panel
    panel_w, panel_h = 260, 160
    px, py = w - panel_w - 10, 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), BLACK, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Title
    cv2.putText(frame, "ClassSense AI",
                (px + 8, py + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, BLUE, 2)

    # Stats
    cv2.putText(frame, f"Students   : {total}",
                (px + 8, py + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    cv2.putText(frame, f"Attentive  : {attentive}",
                (px + 8, py + 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)
    cv2.putText(frame, f"Distracted : {distracted}",
                (px + 8, py + 96),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)

    # Engagement bar
    bx, by = px + 8, py + 115
    bw     = panel_w - 16
    cv2.rectangle(frame, (bx, by), (bx + bw, by + 14), (50, 50, 50), -1)
    fill      = int(bw * engagement / 100)
    bar_color = GREEN if engagement >= 70 else YELLOW if engagement >= 40 else RED
    if fill > 0:
        cv2.rectangle(frame, (bx, by), (bx + fill, by + 14), bar_color, -1)
    cv2.putText(frame, f"{engagement:.0f}% class engagement",
                (bx, by + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1)

    # FPS counter
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

# ── Main ───────────────────────────────────────
def main():
    print("Starting ClassSense AI...", flush=True)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    download_mp_model()

    # Load trained model
    print("Loading trained model...", flush=True)
    classifier = joblib.load(MODEL_PATH)
    scaler     = joblib.load(SCALER_PATH)
    threshold  = joblib.load(THRESH_PATH)
    print(f"Model loaded. Threshold: {threshold:.2f}", flush=True)

    # Load YOLO
    print("Loading YOLO...", flush=True)
    yolo = YOLO("yolov8n.pt")
    print("YOLO ready.", flush=True)

    # Load MediaPipe
    print("Loading MediaPipe...", flush=True)
    base_options  = mp_python.BaseOptions(model_asset_path=MP_MODEL)
    options       = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=10,
        min_face_detection_confidence=0.2,
        min_face_presence_confidence=0.2,
        min_tracking_confidence=0.2,
    )
    face_detector = mp_vision.FaceLandmarker.create_from_options(options)
    print("MediaPipe ready.", flush=True)

    # Open webcam
    print("Opening camera...", flush=True)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("ERROR: Could not open camera.", flush=True)
        return

    print("Running. Press Q to quit, S to snapshot.\n", flush=True)

    INFER_EVERY  = 5
    frame_count  = 0
    last_results = []
    fps_timer    = time.time()
    fps          = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        frame_count += 1
        now       = time.time()
        fps       = 1.0 / (now - fps_timer + 1e-6)
        fps_timer = now

        if frame_count % INFER_EVERY == 0:
            h, w    = frame.shape[:2]
            results = []

            # Step 1 — YOLO: detect all persons
            yolo_out = yolo(frame, classes=[0], verbose=False)[0]

            for box in yolo_out.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf            = float(box.conf[0])
                if conf < 0.25:
                    continue

                # Pad crop slightly for better face detection
                pad  = 20
                x1p  = max(0, x1 - pad)
                y1p  = max(0, y1 - pad)
                x2p  = min(w,  x2 + pad)
                y2p  = min(h,  y2 + pad)
                crop = frame[y1p:y2p, x1p:x2p]
                if crop.size == 0:
                    continue

                # Step 2 — MediaPipe: landmarks on crop
                crop_h, crop_w = crop.shape[:2]
                rgb      = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                mp_res   = face_detector.detect(mp_image)

                if not mp_res.face_landmarks:
                    # Face not visible = looking away = distracted
                    results.append(("Distracted", 0.85))
                    draw_label_box(frame, x1, y1, x2, y2, "Distracted", 0.85, RED)
                    continue

                # Step 3 — Extract features
                landmarks       = mp_res.face_landmarks[0]
                features        = extract_features(landmarks, crop_w, crop_h)

                # Step 4 — Scale
                features_scaled = scaler.transform(features)

                # Step 5 — Classify
                proba      = classifier.predict_proba(features_scaled)[0]
                dist_prob  = proba[0]
                att_prob   = proba[1]
                label      = "Distracted" if dist_prob >= threshold else "Attentive"
                color      = RED if label == "Distracted" else GREEN
                prob       = dist_prob if label == "Distracted" else att_prob

                results.append((label, prob))
                draw_label_box(frame, x1, y1, x2, y2, label, prob, color)

            last_results = results

        # Dashboard drawn every frame
        draw_dashboard(frame, last_results, fps)
        cv2.imshow("ClassSense AI - Live Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quitting...")
            break
        elif key == ord("s"):
            snap = os.path.join(SNAPSHOT_DIR, f"snap_{int(time.time())}.jpg")
            cv2.imwrite(snap, frame)
            print(f"Snapshot saved: {snap}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("CRASH:", e, flush=True)
        traceback.print_exc()
        input("Press Enter to close...")