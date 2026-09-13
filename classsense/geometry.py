# classsense/geometry.py
# Facial geometry, computed one way for the whole project.
#
# Both scripts/extract_features.py and the live pipeline import from here. That
# sharing is deliberate and structural: the train/serve skew this replaces
# existed because each path carried its own copy of head_pose_angles, and the
# two copies drifted into different units. A single implementation cannot drift.
#
# Every measure here is scale-invariant - a ratio of distances, or an angle -
# so a student at the back of the room and one at the front produce comparable
# numbers, and a resized crop produces the same numbers as the original.

import numpy as np

from classsense.config import (
    LEFT_EYE, RIGHT_EYE, MOUTH, NOSE_TIP, FOREHEAD, CHIN,
    LEFT_FACE, RIGHT_FACE, LEFT_EYE_OUTER, RIGHT_EYE_OUTER,
    FACE_WIDTH_TO_HEIGHT,
)

EPS = 1e-6


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    """
    Vertical eye opening over horizontal eye width.

    Roughly 0.28 for an open eye, under 0.20 closed. A ratio, so it does not
    change when the crop is resized - provided the resize preserved aspect
    ratio, which crop_to_max_side does.
    """
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    h1 = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return float(round((v1 + v2) / (2.0 * h1 + EPS), 4))


def mouth_aspect_ratio(landmarks, w, h):
    """Vertical mouth opening over mouth width. Above ~0.52 reads as a yawn."""
    top    = np.array([landmarks[MOUTH[2]].x * w, landmarks[MOUTH[2]].y * h])
    bottom = np.array([landmarks[MOUTH[3]].x * w, landmarks[MOUTH[3]].y * h])
    left   = np.array([landmarks[MOUTH[0]].x * w, landmarks[MOUTH[0]].y * h])
    right  = np.array([landmarks[MOUTH[1]].x * w, landmarks[MOUTH[1]].y * h])
    return float(round(
        np.linalg.norm(top - bottom) / (np.linalg.norm(left - right) + EPS), 4
    ))


def head_pose_angles(landmarks, w, h):
    """
    Head orientation in degrees: (yaw, pitch, roll).

    Yaw and pitch come from distance *ratios* rather than raw pixel offsets, so
    they hold up as a student's apparent size changes across the room. The
    earlier implementation used `(nose.x - face_centre_x) * 100`, which scales
    with how large the face happens to be in the crop and so meant different
    things for the front and back rows.

      yaw   negative = turned toward their own left, positive = their right
      pitch negative = chin up / leaning back, positive = chin down
      roll  0 = level; magnitude grows as the head tilts onto a shoulder

    The x70 and x60 factors map the ratios onto a degree-like range. They are
    calibration constants, not a true projective solve - good enough to
    threshold against, and the thresholds in config.py were set against them.
    """
    nose = landmarks[NOSE_TIP]

    # Yaw: nose sits midway between the cheeks head-on; the ratio skews as the
    # head turns and one cheek foreshortens.
    d_left  = abs(nose.x - landmarks[LEFT_FACE].x)
    d_right = abs(nose.x - landmarks[RIGHT_FACE].x)
    yaw_ratio = (d_left - d_right) / (d_left + d_right + EPS)
    yaw_deg = float(round(yaw_ratio * 70.0, 2))

    # Pitch: same trick vertically, forehead against chin.
    d_forehead = abs(nose.y - landmarks[FOREHEAD].y)
    d_chin     = abs(nose.y - landmarks[CHIN].y)
    pitch_ratio = (d_forehead - d_chin) / (d_forehead + d_chin + EPS)
    pitch_deg = float(round(pitch_ratio * 60.0, 2))

    # Roll: the true angle of the eye line off horizontal.
    dx = (landmarks[RIGHT_EYE_OUTER].x - landmarks[LEFT_EYE_OUTER].x) * w
    dy = (landmarks[RIGHT_EYE_OUTER].y - landmarks[LEFT_EYE_OUTER].y) * h
    roll_deg = float(round(float(np.degrees(np.arctan2(dy, dx))), 2))

    return yaw_deg, pitch_deg, roll_deg


def face_width_px(landmarks, w):
    """Raw cheek-to-cheek width in pixels. Foreshortens as the head turns."""
    return float(abs(landmarks[RIGHT_FACE].x - landmarks[LEFT_FACE].x) * w)


def face_height_px(landmarks, h):
    """Forehead-to-chin height in pixels. Unaffected by yaw."""
    return float(abs(landmarks[CHIN].y - landmarks[FOREHEAD].y) * h)


def face_size_px(landmarks, w, h):
    """
    How many pixels we have on this face, robust to head turn.

    This is what assigns a resolution tier, so it is measured on the face
    itself rather than inferred from the YOLO person box - a person box
    includes torso and varies with posture, and would mis-tier a student who
    leans forward.

    Cheek width alone is the obvious measure and the wrong one: it projects as
    cos(yaw), so a student turning 40 degrees loses about a quarter of their
    apparent width and drops a tier - losing Sleepy detection at the moment
    they are most worth watching. Face height does not foreshorten with yaw, so
    scaling it by the measured 0.887 width-to-height ratio gives a second
    estimate that survives the turn.

    Taking the larger of the two means a frontal face uses whichever is
    cleaner, and a turned face keeps the height-derived estimate rather than
    being demoted for turning.
    """
    width = face_width_px(landmarks, w)
    from_height = face_height_px(landmarks, h) * FACE_WIDTH_TO_HEIGHT
    return max(width, from_height)


def face_centre_in_frame(landmarks, box, crop_w, crop_h, scale, padding):
    """
    Where this face sits in the original frame, in pixels.

    Landmarks are normalised to their crop, so comparing two students' faces
    means walking each back through its crop's resize and its offset within the
    frame. Without this, every face reports a position near (0.5, 0.4) of its
    own crop and they all look identical.

    This is what makes duplicate students detectable: two tracks can hold quite
    different boxes and still be one person, and only the face position says so.
    """
    nose = landmarks[NOSE_TIP]
    inv = 1.0 / max(scale, EPS)
    x0 = max(0, box[0] - padding)
    y0 = max(0, box[1] - padding)
    return (x0 + nose.x * crop_w * inv,
            y0 + nose.y * crop_h * inv)


def extract_feature_row(landmarks, w, h):
    """
    The full feature vector, in config.FEATURE_COLS order.

    Returns a dict rather than an array so callers cannot silently reorder it.
    """
    left_ear  = eye_aspect_ratio(landmarks, LEFT_EYE,  w, h)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
    avg_ear   = round((left_ear + right_ear) / 2.0, 4)
    mar       = mouth_aspect_ratio(landmarks, w, h)
    yaw, pitch, roll = head_pose_angles(landmarks, w, h)
    return {
        "left_ear":  left_ear,
        "right_ear": right_ear,
        "avg_ear":   avg_ear,
        "mar":       mar,
        "yaw":       yaw,
        "pitch":     pitch,
        "roll":      roll,
    }


def compute_box_iou(box_a, box_b):
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return float(inter / (area_a + area_b - inter + EPS))
