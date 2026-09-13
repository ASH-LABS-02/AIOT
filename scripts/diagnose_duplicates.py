# scripts/diagnose_duplicates.py
# Why is one person being counted as more than one student?
#
#   python scripts/diagnose_duplicates.py --seconds 15
#
# Logs, per cycle: every raw YOLO box with its confidence, the pairwise IoU
# between them, which track each became, and where each track's detected face
# actually landed in frame coordinates.
#
# The last of those is the decisive one. Two boxes can look like two people by
# any box-shaped measure and still resolve to the same face - and a face is a
# point, so two tracks whose faces coincide are one person no matter what their
# boxes say.

import argparse
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classsense import config                                    # noqa: E402
from classsense.geometry import compute_box_iou, face_size_px    # noqa: E402
from classsense.mp_pool import MediaPipePool, prepare_crop       # noqa: E402
from classsense.pipeline import FrameSource                      # noqa: E402
from classsense.tracker import StudentTracker                    # noqa: E402
from classsense.tiers import tier_for_face_width                 # noqa: E402


def face_centre_in_frame(landmarks, box, cw, ch, scale, padding):
    """
    Where this face sits in the ORIGINAL frame, in pixels.

    Landmarks are normalised to the crop, so they must be walked back through
    the crop's resize and its offset within the frame before two tracks'
    faces can be compared.
    """
    nose = landmarks[config.NOSE_TIP]
    x_in_crop = nose.x * cw / max(scale, 1e-6)
    y_in_crop = nose.y * ch / max(scale, 1e-6)
    x0 = max(0, box[0] - padding)
    y0 = max(0, box[1] - padding)
    return x0 + x_in_crop, y0 + y_in_crop


def main():
    p = argparse.ArgumentParser(description="Diagnose duplicate students")
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--yolo-width", type=int, default=960)
    p.add_argument("--conf", type=float, default=config.YOLO_CONF_THRESH)
    p.add_argument("--source", default=None)
    p.add_argument("--save", action="store_true",
                   help="save annotated frames to data/snapshots")
    args = p.parse_args()

    from ultralytics import YOLO
    yolo = YOLO(config.YOLO_WEIGHTS)
    pool = MediaPipePool(4, face_conf=config.MP_FACE_CONF_THRESH)
    tracker = StudentTracker()

    src_arg = args.source if args.source else config.CAPTURE_INDEX
    source = FrameSource(src_arg).start()
    print(f"source {source.actual_size[0]}x{source.actual_size[1]}, "
          f"yolo {args.yolo_width}px, conf gate {args.conf}\n")

    os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
    deadline = time.time() + args.seconds
    cycle = 0
    box_count_hist = Counter()
    confirmed_hist = Counter()
    worst = None

    while time.time() < deadline and source.running:
        frame = source.read()
        if frame is None:
            break
        cycle += 1
        now = time.time()

        out = yolo(frame, classes=[0], imgsz=args.yolo_width, verbose=False)[0]
        raw = []
        for b in out.boxes:
            conf = float(b.conf[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            raw.append(((x1, y1, x2, y2), conf))

        kept = [(box, c) for box, c in raw if c >= args.conf]
        box_count_hist[len(kept)] += 1

        tracker.update([b for b, _ in kept], now)

        faces = {}
        for sid, student in tracker.students.items():
            prepared = prepare_crop(frame, student.box, config.CROP_PADDING)
            if prepared is None:
                continue
            image, cw, ch, scale = prepared
            lm = pool.detect(image)
            if lm is None:
                continue
            student.note_face_hit()
            student.last_seen = now
            student.analysis_count += 1
            cx, cy = face_centre_in_frame(
                lm, student.box, cw, ch, scale, config.CROP_PADDING
            )
            size = face_size_px(lm, cw, ch) / max(scale, 1e-6)
            faces[sid] = (cx, cy, size, tier_for_face_width(size))

        confirmed = tracker.confirmed()
        confirmed_hist[len(confirmed)] += 1

        # Only print cycles that show more than one confirmed student - the
        # symptom under investigation.
        if len(confirmed) > 1 or (cycle % 15 == 1):
            print(f"--- cycle {cycle}: {len(raw)} raw boxes, {len(kept)} above "
                  f"gate, {len(tracker.students)} tracks, "
                  f"{len(confirmed)} CONFIRMED ---")
            for i, (box, conf) in enumerate(raw):
                w = box[2] - box[0]
                h = box[3] - box[1]
                mark = "kept" if conf >= args.conf else "GATED"
                print(f"    box{i}: conf {conf:.3f} {mark:>5}  "
                      f"xy=({box[0]},{box[1]})-({box[2]},{box[3]})  {w}x{h}px")
            for i in range(len(kept)):
                for j in range(i + 1, len(kept)):
                    iou = compute_box_iou(kept[i][0], kept[j][0])
                    print(f"    IoU(box{i},box{j}) = {iou:.3f}"
                          f"{'   <-- overlapping, likely one person' if iou > 0.3 else ''}")
            for sid, (cx, cy, size, tier) in faces.items():
                st = tracker.students[sid]
                print(f"    track {sid}: face at ({cx:.0f},{cy:.0f}) "
                      f"size {size:.0f}px {tier.name} "
                      f"confirmed={st.face_confirmed}")

            ids = list(faces)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    ax, ay, asz, _ = faces[ids[i]]
                    bx, by, bsz, _ = faces[ids[j]]
                    dist = float(np.hypot(ax - bx, ay - by))
                    ref = max(asz, bsz)
                    print(f"    face distance track{ids[i]}<->track{ids[j]}: "
                          f"{dist:.0f}px ({dist / ref:.2f} face-widths)"
                          f"{'   <-- SAME FACE, duplicate student' if dist < ref * 0.6 else ''}")

            if len(confirmed) > 1 and worst is None and args.save:
                annotated = frame.copy()
                for sid, s in confirmed.items():
                    cv2.rectangle(annotated, s.box[:2], s.box[2:], (0, 0, 255), 2)
                    cv2.putText(annotated, f"track {sid}", (s.box[0], s.box[1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                for sid, (cx, cy, size, _) in faces.items():
                    cv2.circle(annotated, (int(cx), int(cy)), 6, (0, 255, 255), -1)
                worst = os.path.join(config.SNAPSHOT_DIR, "duplicate_evidence.jpg")
                cv2.imwrite(worst, annotated)
                print(f"    -> saved {worst}")
            print()

    source.stop()
    pool.close()

    print("=" * 64)
    print("Summary")
    print("=" * 64)
    print(f"cycles: {cycle}")
    print("boxes above gate per cycle : "
          + ", ".join(f"{k} boxes x{v}" for k, v in sorted(box_count_hist.items())))
    print("confirmed students per cycle: "
          + ", ".join(f"{k} students x{v}" for k, v in sorted(confirmed_hist.items())))
    dupes = sum(v for k, v in confirmed_hist.items() if k > 1)
    if dupes:
        print(f"\n{dupes}/{cycle} cycles reported more than one student.")
    else:
        print("\nNo duplicate cycles observed in this run.")


if __name__ == "__main__":
    main()
