# scripts/export_ncnn.py
# Export YOLOv8n to NCNN, for ARM.
#
#   python scripts/export_ncnn.py                 # export at the calibrated width
#   python scripts/export_ncnn.py --imgsz 640     # export at a specific width
#   python scripts/export_ncnn.py --verify        # export, then check it detects
#
# NCNN is built around ARM NEON and is why this is worth having on a Raspberry
# Pi. It is NOT universally faster: measured on x86 it ran 53ms against
# PyTorch's 29ms at 640px. Calibration measures both backends on whichever
# machine it runs on and records the winner, so nothing here assumes.
#
# The export can be produced anywhere - the .param/.bin files are portable - so
# it is reasonable to export on a desktop and copy the directory to the Pi.
#
# THE ONE THING TO GET RIGHT: an NCNN export has a fixed input shape. Run a
# 640px export at 960px and it returns zero detections, silently. The width is
# recorded alongside the model and the pipeline refuses a mismatch.

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classsense import capacity as cap_mod                      # noqa: E402
from classsense import config                                   # noqa: E402
from classsense.detector import write_ncnn_meta, ncnn_export_width  # noqa: E402

RULE = "=" * 66


def verify(model_dir, imgsz):
    """
    Confirm the export actually detects people at its own width.

    Worth the extra minute: the failure mode is an empty result rather than an
    error, so an export that silently detects nothing looks exactly like a
    successful one until it is deployed.
    """
    import cv2
    from ultralytics import YOLO

    frame = None
    train = os.path.join(config.DAISEE_ROOT, "DataSet", "Train")
    if os.path.isdir(train):
        for person in sorted(os.scandir(train), key=lambda e: e.name):
            if not person.is_dir():
                continue
            for clip in sorted(os.scandir(person.path), key=lambda e: e.name):
                if not clip.is_dir():
                    continue
                for f in os.scandir(clip.path):
                    if f.name.lower().endswith((".avi", ".mp4")):
                        c = cv2.VideoCapture(f.path)
                        ok, frame = c.read()
                        c.release()
                        if ok:
                            break
                if frame is not None:
                    break
            if frame is not None:
                break
    if frame is None:
        c = cv2.VideoCapture(config.CAPTURE_INDEX)
        ok, frame = c.read()
        c.release()
        if not ok:
            print("No frame available to verify against; skipping.")
            return None

    full = cv2.resize(frame, (config.CAPTURE_WIDTH, config.CAPTURE_HEIGHT))

    print("\n" + RULE)
    print("Verification")
    print(RULE)

    pt = YOLO(config.YOLO_WEIGHTS)
    nc = YOLO(model_dir, task="detect")

    def count(model, size):
        out = model(full, classes=[0], imgsz=size, verbose=False)[0]
        confs = [float(b.conf[0]) for b in out.boxes
                 if float(b.conf[0]) >= config.YOLO_CONF_THRESH]
        return len(confs), (max(confs) if confs else 0.0)

    pt_n, pt_c = count(pt, imgsz)
    nc_n, nc_c = count(nc, imgsz)
    print(f"at export width {imgsz}px:")
    print(f"  pytorch : {pt_n} people, best confidence {pt_c:.3f}")
    print(f"  ncnn    : {nc_n} people, best confidence {nc_c:.3f}")

    ok = True
    if nc_n == 0 and pt_n > 0:
        print("\n  FAILED: NCNN found nobody where PyTorch found somebody.")
        print("  Do not deploy this export.")
        ok = False
    elif nc_n != pt_n:
        print(f"\n  NOTE: the backends disagree ({pt_n} vs {nc_n}). Small")
        print("  differences are normal - NCNN quantises slightly differently")
        print("  - but check on a real view before trusting the headcount.")
    else:
        print("\n  Backends agree at the export width.")

    # Demonstrate the failure this guards against, so it is concrete.
    other = 960 if imgsz != 960 else 640
    wrong_n, _ = count(nc, other)
    print(f"\nat the WRONG width {other}px: ncnn finds {wrong_n} people")
    if wrong_n == 0:
        print("  ^ this is why the width is recorded and a mismatch refused.")

    return ok


def main():
    p = argparse.ArgumentParser(description="Export YOLOv8n to NCNN for ARM")
    p.add_argument("--imgsz", type=int, default=None,
                   help="input width to export at (default: from calibration)")
    p.add_argument("--weights", default=config.YOLO_WEIGHTS)
    p.add_argument("--out", default=config.YOLO_NCNN_DIR)
    p.add_argument("--verify", action="store_true",
                   help="check the export detects people before trusting it")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing export")
    args = p.parse_args()

    imgsz = args.imgsz
    if imgsz is None:
        cap = cap_mod.load()
        imgsz = cap.yolo_width
        print(f"Using imgsz {imgsz} from {cap.source} tuning.")
        if cap.source != "measured":
            print("  Note: that is an estimate. Run scripts/calibrate.py on "
                  "this machine first, or pass --imgsz explicitly.")

    print(RULE)
    print("Exporting YOLOv8n to NCNN")
    print(RULE)
    print(f"weights : {args.weights}")
    print(f"imgsz   : {imgsz}")
    print(f"output  : {args.out}")

    if not os.path.exists(args.weights):
        print(f"\nERROR: weights not found at {args.weights}")
        return 1

    if os.path.isdir(args.out):
        existing = ncnn_export_width(args.out)
        if existing == imgsz and not args.force:
            print(f"\nAn export at {imgsz}px already exists. "
                  f"Use --force to rebuild.")
            return 0
        print(f"\nRemoving existing export (was {existing}px).")
        shutil.rmtree(args.out)

    from ultralytics import YOLO
    model = YOLO(args.weights)
    # Export writes next to the weights; move it if a different output was asked.
    produced = model.export(format="ncnn", imgsz=imgsz, half=False)
    produced = str(produced)

    if os.path.abspath(produced) != os.path.abspath(args.out):
        if os.path.isdir(args.out):
            shutil.rmtree(args.out)
        shutil.move(produced, args.out)

    write_ncnn_meta(args.out, imgsz, args.weights)

    size_mb = sum(
        os.path.getsize(os.path.join(args.out, f))
        for f in os.listdir(args.out)
    ) / 1e6
    print(f"\nExported to {args.out} ({size_mb:.1f} MB), width recorded.")

    ok = True
    if args.verify:
        ok = verify(args.out, imgsz)

    print("\n" + RULE)
    print("Next")
    print(RULE)
    print("Re-run calibration so it measures both backends and picks:")
    print("  python scripts/calibrate.py")
    print()
    print("NCNN's advantage is ARM NEON. On x86 it measured slower than")
    print("PyTorch, so calibration choosing PyTorch on a desktop is correct,")
    print("not a failure.")

    return 0 if ok is not False else 1


if __name__ == "__main__":
    sys.exit(main())
