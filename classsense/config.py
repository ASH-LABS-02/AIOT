# classsense/config.py
# Single source of truth for every tunable value.
#
# Training and live inference both import from here, which is what stops the
# two paths from drifting into different units the way they had.

import os

# ── Repo layout ────────────────────────────────
# Resolved from this file's location, so every script works no matter which
# directory the user runs it from.
REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR     = os.path.join(REPO_ROOT, "data")
MODELS_DIR   = os.path.join(REPO_ROOT, "models")
SCRIPTS_DIR  = os.path.join(REPO_ROOT, "scripts")
SNAPSHOT_DIR = os.path.join(DATA_DIR, "snapshots")

MODEL_PATH   = os.path.join(MODELS_DIR, "engagement_classifier.pkl")
SCALER_PATH  = os.path.join(MODELS_DIR, "scaler.pkl")
THRESH_PATH  = os.path.join(MODELS_DIR, "threshold.pkl")
META_PATH    = os.path.join(MODELS_DIR, "model_meta.json")
MP_MODEL     = os.path.join(SCRIPTS_DIR, "face_landmarker.task")
YOLO_WEIGHTS = os.path.join(REPO_ROOT, "yolov8n.pt")

MP_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# ── DAiSEE dataset ─────────────────────────────
# Only the Train split was ever downloaded (13 subjects, 821 clips); there are
# no Validation or Test videos on disk, which is why evaluation is GroupKFold
# over subjects rather than a held-out split.
DAISEE_ROOT   = os.environ.get(
    "DAISEE_ROOT", r"C:\Users\AR\Downloads\DAiSEE\DAiSEE"
)
DAISEE_SPLIT  = "Train"
DAISEE_LABELS = os.path.join(DAISEE_ROOT, "Labels", f"{DAISEE_SPLIT}Labels.csv")
DAISEE_VIDEOS = os.path.join(DAISEE_ROOT, "DataSet", DAISEE_SPLIT)

# Engagement is scored 0-3. Cutting at >= 2 yields 789/32 (3.9% minority from
# only 8 subjects), which is not learnable. Cutting at >= 3 yields 451/370.
# The target therefore means "fully engaged" vs "drifting", not "engaged" vs
# "disengaged" - the more useful signal for a classroom monitor regardless.
ENGAGEMENT_POSITIVE_MIN = 3
CLASS_NAMES             = ["Drifting", "Engaged"]

SAMPLE_EVERY = 15   # frames between samples when extracting clip features

# ── Landmark indices (MediaPipe FaceMesh 478-point) ──
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]
MOUTH      = [61, 291, 13, 14]
NOSE_TIP   = 4
FOREHEAD   = 10
CHIN       = 152
LEFT_FACE  = 234
RIGHT_FACE = 454
LEFT_EYE_OUTER  = 33
RIGHT_EYE_OUTER = 263

# Feature vector the classifier consumes. Order is load-bearing: the same list
# builds training rows and live inference rows.
FEATURE_COLS = ["left_ear", "right_ear", "avg_ear", "mar", "yaw", "pitch", "roll"]

# ── Colours (BGR) ──────────────────────────────
GREEN     = (0, 220, 0)       # Attentive
ORANGE    = (0, 140, 255)     # Sleepy
RED       = (0, 0, 220)       # Distracted
GRAY      = (140, 140, 140)   # Unknown / below resolution floor
YELLOW    = (0, 220, 220)
WHITE     = (255, 255, 255)
BLACK     = (0, 0, 0)
BLUE      = (220, 100, 0)
PANEL_BG  = (25, 25, 25)

# ── Physiological thresholds ───────────────────
EAR_CLOSED_THRESH = 0.20   # below this the eye reads as closed
MAR_YAWN_THRESH   = 0.52   # above this the mouth reads as a yawn

# ── Head pose thresholds (degrees) ─────────────
YAW_DISTRACTED_THRESH = 18.0    # |deg| turned away from the board
ROLL_RECLINED_THRESH  = 20.0    # |deg| head tilted onto a shoulder
PITCH_RECLINED_THRESH = -15.0   # chin up, leaning back
PITCH_NOD_THRESH      = 18.0    # chin down, looking at lap or nodding off

# ── Detection confidence gates ─────────────────
YOLO_CONF_THRESH    = 0.30   # lowered: back-row students are small and dim
MP_FACE_CONF_THRESH = 0.45

# ── Temporal gates (seconds) ───────────────────
# Every gate is wall-clock, not frame-counted, because the analysis rate varies
# with how many students are in frame.
SLEEPY_EYES_DURATION = 1.2
YAWN_DURATION        = 1.5
DISTRACTED_DURATION  = 0.8
FACE_LOST_DURATION   = 0.5
SMOOTHING_WINDOW     = 5

TRACK_IOU_MATCH      = 0.20   # minimum IoU to treat a box as the same person
TRACK_STALE_SECONDS  = 2.0    # drop a track unseen this long
TRACK_UNCONFIRMED_TTL = 1.0   # earliest a never-confirmed box may be dropped
FACE_CONFIRM_HITS    = 2      # landmark hits before a box counts as a student

# Chances an unconfirmed box gets before it is written off as furniture.
# Retirement needs BOTH this many analysis attempts AND the wall-clock TTL,
# never either alone. A wall clock on its own is a trap: confirmation takes
# FACE_CONFIRM_HITS *cycles*, so on hardware where one cycle exceeds the TTL
# no track can ever survive long enough to confirm, and the room reads as
# permanently empty.
UNCONFIRMED_MAX_ATTEMPTS = 6

# ── Resolution tiers ───────────────────────────
# 1080p cannot give 60 faces equal quality. Each student is tiered by measured
# face width and the tier bounds which states they can be assigned, so a label
# never looks more precise than the pixels supporting it.
#
# EAR is suppressed below FULL because at a 50px face the eye landmarks sit
# ~3px apart: one pixel of jitter is a ~10% EAR swing. Head pose spans the whole
# face and degrades gracefully, so it survives a tier lower.
TIER_FULL_MIN_WIDTH   = 64   # full analysis, all states including Sleepy
TIER_COARSE_MIN_WIDTH = 40   # pose only; Attentive / Distracted
# below TIER_COARSE_MIN_WIDTH: presence only, state Unknown

# Typical cheek-width to face-height ratio, measured at 0.887 +/- 0.025 across
# DAiSEE frames. Used to derive a yaw-invariant face size: cheek width
# foreshortens as cos(yaw), so a student turning 40 degrees would lose about a
# quarter of their apparent width and be demoted out of the tier that permits
# Sleepy detection - precisely when they are most worth watching. Face height
# does not foreshorten with yaw, so scaling it by this ratio gives a second
# estimate that holds steady through a turn.
FACE_WIDTH_TO_HEIGHT = 0.887

# ── Throughput ─────────────────────────────────
# Measured on 20 cores, CPU-only torch (see scripts/benchmark.py):
#   detect() @ 96/128/192px crop : 7.13 / 7.21 / 6.71 ms  (flat)
#   detect() @ 256/320px crop    : 14.67 / 14.46 ms       (doubles)
#   8 threads                    : 3.21x speedup, 2.09 ms/face
#   12 threads                   : 3.08x, 26% efficiency  (past the knee)
#   YOLO @ 640/960/1280          : 36.1 / 64.1 / 107.6 ms
#
# 60 students: 60 x 2.09ms = 125ms landmarks + 108ms YOLO = ~233ms/cycle,
# so every student is re-examined ~4.3x per second and the whole cohort fits
# in every cycle. MAX_STUDENTS_PER_CYCLE is the degradation path for slower
# machines, not the normal operating mode here.
MP_POOL_SIZE           = 8
MP_CROP_MAX_SIDE       = 192   # above this MediaPipe costs double for nothing
YOLO_INPUT_WIDTH       = 1280  # 1280 over 960: back-row students are small
MAX_STUDENTS_PER_CYCLE = 60
CROP_PADDING           = 20    # px of context around a YOLO box

# Cycles between YOLO passes. Profiling a 60-student cycle put YOLO at 124ms
# of 263ms - 47% - spent re-finding people who have not moved. Seated students
# keep their positions for seconds at a time, while the things that actually
# change (eyes, mouth, head angle) are read from landmarks every cycle
# regardless. So person detection runs periodically and landmark analysis runs
# continuously.
#
# The cost is admission latency: a student entering the room is picked up
# within DETECT_EVERY cycles, about half a second here. CROP_PADDING absorbs
# the box drift of someone shifting in their seat between passes.
DETECT_EVERY = 3

# Adaptive scheduling: when the cohort exceeds MAX_STUDENTS_PER_CYCLE, sample
# by priority instead of blindly cycling. A student holding one state for
# STABLE_AFTER_SECONDS is cheap to revisit less often; one mid-transition is not.
STABLE_AFTER_SECONDS   = 30.0
STABLE_PRIORITY_PENALTY = 0.25

# ── Capture ────────────────────────────────────
CAPTURE_WIDTH  = 1920
CAPTURE_HEIGHT = 1080
CAPTURE_INDEX  = 0
