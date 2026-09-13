# scripts/test_setup.py
# Environment smoke test - confirms every heavy dependency imports and that
# the model weights load, before anything else is attempted.
#
#   python scripts/test_setup.py

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import cv2
    import mediapipe
    import numpy
    import sklearn
    import torch
    from ultralytics import YOLO

    from classsense import config

    print("=" * 52)
    print("ClassSense environment check")
    print("=" * 52)
    print(f"python       : {sys.version.split()[0]}")
    print(f"numpy        : {numpy.__version__}")
    print(f"OpenCV       : {cv2.__version__}")
    print(f"MediaPipe    : {mediapipe.__version__}")
    print(f"scikit-learn : {sklearn.__version__}")
    print(f"torch        : {torch.__version__}  CUDA={torch.cuda.is_available()}")
    print(f"CPU cores    : {os.cpu_count()}")

    if not torch.cuda.is_available():
        # Not a problem - every performance figure in the README is a CPU
        # figure - but worth stating so nobody assumes a GPU is in play.
        print("             ^ CPU-only build; this is the measured configuration")

    print()
    ok = True

    if os.path.exists(config.YOLO_WEIGHTS):
        YOLO(config.YOLO_WEIGHTS)
        print(f"YOLO weights : loaded from {config.YOLO_WEIGHTS}")
    else:
        print(f"YOLO weights : MISSING at {config.YOLO_WEIGHTS}")
        ok = False

    if os.path.exists(config.MP_MODEL):
        size_mb = os.path.getsize(config.MP_MODEL) / 1e6
        print(f"Face model   : present ({size_mb:.1f} MB)")
    else:
        print("Face model   : absent - it downloads automatically on first run")

    if os.path.exists(config.MODEL_PATH):
        print(f"Classifier   : present at {config.MODEL_PATH}")
        if os.path.exists(config.META_PATH):
            import json
            with open(config.META_PATH, encoding="utf-8") as fh:
                meta = json.load(fh)
            verdict = ("advisory only - see README"
                       if meta.get("advisory_only") else "cleared for live use")
            print(f"             : AUC {meta.get('oof_roc_auc')}, {verdict}")
    else:
        print("Classifier   : absent - live detection runs on heuristics")

    print()
    print("Ready." if ok else "Setup incomplete - see MISSING above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
