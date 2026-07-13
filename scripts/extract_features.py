# scripts/extract_features.py
# Stage 3 - Feature extraction using NEW MediaPipe Tasks API
# Compatible with mediapipe 0.10.30+

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import pandas as pd
import numpy as np
import os
import urllib.request
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURE THESE PATHS TO MATCH YOUR SYSTEM
# ─────────────────────────────────────────────
DAISEE_ROOT  = r"C:\Users\AR\Downloads\DAiSEE\DAiSEE"
SPLIT        = "Validation"
LABELS_CSV   = os.path.join(DAISEE_ROOT, "Labels", f"{SPLIT}Labels.csv")
DATASET_DIR  = os.path.join(DAISEE_ROOT, "DataSet", SPLIT)
OUTPUT_CSV   = os.path.join("..", "data", f"{SPLIT.lower()}_features.csv")
MODEL_PATH   = "face_landmarker.task"
SAMPLE_EVERY = 15
# ─────────────────────────────────────────────

# ── Download model if not already present ─────
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe face landmarker model (~30MB)...")
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Model downloaded.")
    else:
        print("Model already present, skipping download.")

# ── Landmark indices ───────────────────────────
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]
MOUTH      = [61, 291, 13, 14]
NOSE_TIP   = 4
FOREHEAD   = 10
CHIN       = 152
LEFT_FACE  = 234
RIGHT_FACE = 454

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
    nose         = landmarks[NOSE_TIP]
    face_centre_x = (landmarks[LEFT_FACE].x + landmarks[RIGHT_FACE].x) / 2
    face_centre_y = (landmarks[FOREHEAD].y  + landmarks[CHIN].y)        / 2
    yaw   = round((nose.x - face_centre_x) * 100, 4)
    pitch = round((nose.y - face_centre_y) * 100, 4)
    return yaw, pitch

def label_to_binary(engagement_score):
    return 1 if engagement_score >= 2 else 0

def extract_features_from_video(video_path, detector):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    features_list = []
    frame_idx     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SAMPLE_EVERY == 0:
            h, w = frame.shape[:2]
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = detector.detect(mp_image)

            if result.face_landmarks:
                lms       = result.face_landmarks[0]
                left_ear  = eye_aspect_ratio(lms, LEFT_EYE,  w, h)
                right_ear = eye_aspect_ratio(lms, RIGHT_EYE, w, h)
                avg_ear   = round((left_ear + right_ear) / 2, 4)
                mar       = mouth_aspect_ratio(lms, w, h)
                yaw, pitch = head_pose_angles(lms, w, h)

                features_list.append({
                    "left_ear"  : left_ear,
                    "right_ear" : right_ear,
                    "avg_ear"   : avg_ear,
                    "mar"       : mar,
                    "yaw"       : yaw,
                    "pitch"     : pitch,
                })

        frame_idx += 1

    cap.release()
    return features_list

# ── Main ───────────────────────────────────────
def main():
    os.makedirs(os.path.join("..", "data"), exist_ok=True)
    download_model()

    # Set up detector once, reuse across all videos (much faster)
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options      = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = mp_vision.FaceLandmarker.create_from_options(options)

    print(f"Reading labels from: {LABELS_CSV}")
    labels_df = pd.read_csv(LABELS_CSV)
    print(f"Total clips: {len(labels_df)}")

    all_rows  = []
    processed = 0
    skipped   = 0

    for _, row in labels_df.iterrows():
        clip_id    = row["ClipID"]
        engagement = row["Engagement"]
        label      = label_to_binary(engagement)

        clip_name  = Path(clip_id).stem
        person_id  = clip_name[:6]
        video_path = os.path.join(DATASET_DIR, person_id, clip_name, clip_id)

        if not os.path.exists(video_path):
            skipped += 1
            continue

        frame_features = extract_features_from_video(video_path, detector)

        if not frame_features:
            skipped += 1
            continue

        avg_features = {
            k: round(np.mean([f[k] for f in frame_features]), 4)
            for k in frame_features[0].keys()
        }
        avg_features["label"]      = label
        avg_features["clip_id"]    = clip_id
        avg_features["engagement"] = engagement
        all_rows.append(avg_features)

        processed += 1
        if processed % 50 == 0:
            print(f"  Processed {processed} clips...")

    output_df = pd.DataFrame(all_rows)
    output_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDone.")
    print(f"  Processed : {processed} clips")
    print(f"  Skipped   : {skipped} clips")
    print(f"  Saved to  : {OUTPUT_CSV}")
    print(f"\nLabel distribution:")
    print(output_df["label"].value_counts())
    print(f"\nSample output:")
    print(output_df.head())

if __name__ == "__main__":
    main()