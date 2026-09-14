# classsense/detector.py
# Person detection, and which backend does it.
#
# YOLOv8n runs either through PyTorch or through NCNN. NCNN is built for ARM
# NEON and is the reason this is worth having on a Raspberry Pi; on x86 it
# measured slower than PyTorch (53ms against 29ms at 640px), so neither is
# simply "the fast one". Calibration measures both on the machine in question
# and records the winner, which is the only way to be right on both.
#
# The trap this module exists to close:
#
#   An NCNN export has a FIXED input shape. Run a model exported at 640px at
#   960px instead and it returns zero detections - no exception, no warning,
#   an empty list. Measured here: 1 person found at 640, none at 960, same
#   frame. In a classroom that reads as an empty room, which is the worst
#   possible failure because everything downstream keeps working perfectly.
#
# So the export width is recorded beside the model and a mismatch is refused
# at load time rather than discovered in production.

import json
import os

from classsense import config

NCNN_META = "classsense_export.json"


def ncnn_export_width(model_dir):
    """
    The imgsz an NCNN model was exported at, or None if unrecorded.

    Written by scripts/export_ncnn.py. An export made by hand through
    ultralytics will not have it, which is why callers must treat None as
    "cannot verify" rather than "fine".
    """
    meta_path = os.path.join(model_dir, NCNN_META)
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, encoding="utf-8") as fh:
            return int(json.load(fh)["imgsz"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def write_ncnn_meta(model_dir, imgsz, source_weights):
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, NCNN_META), "w", encoding="utf-8") as fh:
        json.dump({
            "imgsz": int(imgsz),
            "source": os.path.basename(source_weights),
            "note": "Run this model ONLY at this imgsz. NCNN exports have a "
                    "fixed input shape and return zero detections at any "
                    "other size, silently.",
        }, fh, indent=2)


def ncnn_available(model_dir=None):
    model_dir = model_dir or config.YOLO_NCNN_DIR
    return os.path.isdir(model_dir) and any(
        f.endswith(".param") for f in os.listdir(model_dir)
    )


def load_detector(backend="pytorch", yolo_width=None, model_dir=None,
                  weights=None, strict=True):
    """
    Load the detector for a backend, refusing configurations that fail quietly.

    Returns (model, backend_actually_used). Falls back to PyTorch, loudly, when
    NCNN is asked for but unusable - a slower detector is recoverable, a
    detector that finds nobody is not.
    """
    from ultralytics import YOLO

    weights = weights or config.YOLO_WEIGHTS
    model_dir = model_dir or config.YOLO_NCNN_DIR

    if backend != "ncnn":
        return YOLO(weights), "pytorch"

    if not ncnn_available(model_dir):
        print(f"NCNN backend requested but no export at {model_dir}. "
              f"Run scripts/export_ncnn.py. Falling back to PyTorch.",
              flush=True)
        return YOLO(weights), "pytorch"

    exported_at = ncnn_export_width(model_dir)
    if yolo_width is not None and exported_at is not None \
            and int(yolo_width) != exported_at:
        message = (
            f"NCNN model was exported at {exported_at}px but the pipeline is "
            f"configured for {yolo_width}px. An NCNN export has a fixed input "
            f"shape and returns ZERO detections at any other size, without "
            f"raising - the room would read as empty. "
            f"Re-export with: python scripts/export_ncnn.py --imgsz {yolo_width}"
        )
        if strict:
            raise ValueError(message)
        print(message + " Falling back to PyTorch.", flush=True)
        return YOLO(weights), "pytorch"

    if exported_at is None:
        print(f"NCNN export at {model_dir} has no recorded imgsz, so the "
              f"width cannot be verified. If detections are always zero, the "
              f"width is wrong - re-export with scripts/export_ncnn.py.",
              flush=True)

    # task= is required: ultralytics cannot infer it from a bare NCNN
    # directory and warns, then guesses.
    return YOLO(model_dir, task="detect"), "ncnn"
