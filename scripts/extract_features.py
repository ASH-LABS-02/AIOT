# scripts/extract_features.py
# Stage 3 - turn DAiSEE clips into feature rows.
#
#   python scripts/extract_features.py
#   python scripts/extract_features.py --split Train --workers 8
#
# Uses classsense.geometry, the same module the live pipeline uses. That is the
# point: the previous version carried its own copy of the geometry functions,
# the live path carried another, and the two drifted into different units -
# which is how a model trained on raw normalised head pose ended up being fed
# degrees at inference and became a constant predictor.

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classsense import config                                    # noqa: E402
from classsense.geometry import extract_feature_row              # noqa: E402
from classsense.mp_pool import MediaPipePool, ensure_model       # noqa: E402
import mediapipe as mp                                           # noqa: E402


def resolve_paths(split):
    """
    Locate the labels CSV and the video directory for a split.

    DAiSEE ships nested inconsistently and the two previous scripts disagreed
    about it - check_paths.py resolved DataSet/DataSet/<split> while
    extract_features.py resolved DataSet/<split>. Rather than hard-code either,
    try the known layouts and report which one matched.
    """
    labels = os.path.join(config.DAISEE_ROOT, "Labels", f"{split}Labels.csv")
    candidates = [
        os.path.join(config.DAISEE_ROOT, "DataSet", split),
        os.path.join(config.DAISEE_ROOT, "DataSet", "DataSet", split),
        os.path.join(config.DAISEE_ROOT, split),
    ]
    videos = next((c for c in candidates if os.path.isdir(c)), None)
    return labels, videos, candidates


def clip_path(videos_dir, clip_id):
    """DAiSEE nests as <person>/<clip stem>/<clip file>."""
    stem = Path(clip_id).stem
    return os.path.join(videos_dir, stem[:6], stem, clip_id)


def features_for_clip(video_path, pool, sample_every):
    """
    Average the feature vector over sampled frames of one clip.

    Averaging is what makes a clip-level label meaningful: DAiSEE labels the
    whole 10-second clip, so a per-frame row would attach a clip-level
    judgement to a single instant.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    rows = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % sample_every == 0:
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=np.ascontiguousarray(rgb))
                landmarks = pool.detect(image)
                if landmarks is not None:
                    rows.append(extract_feature_row(landmarks, w, h))
            idx += 1
    finally:
        cap.release()

    if not rows:
        return None

    return {
        col: round(float(np.mean([r[col] for r in rows])), 4)
        for col in config.FEATURE_COLS
    }, len(rows)


def main():
    p = argparse.ArgumentParser(description="Extract DAiSEE features")
    p.add_argument("--split", default=config.DAISEE_SPLIT,
                   help="Train / Validation / Test")
    p.add_argument("--workers", type=int, default=config.MP_POOL_SIZE)
    p.add_argument("--sample-every", type=int, default=config.SAMPLE_EVERY)
    p.add_argument("--limit", type=int, default=None,
                   help="stop after N clips (for a quick check)")
    args = p.parse_args()

    labels_csv, videos_dir, tried = resolve_paths(args.split)

    print("=" * 60)
    print(f"Extracting features - split {args.split}")
    print("=" * 60)

    if not os.path.exists(labels_csv):
        print(f"ERROR: labels not found at {labels_csv}")
        print(f"Set DAISEE_ROOT if the dataset lives elsewhere "
              f"(currently {config.DAISEE_ROOT}).")
        sys.exit(1)

    if videos_dir is None:
        print("ERROR: no video directory found. Tried:")
        for c in tried:
            print(f"  {c}")
        print("\nOnly the Train split was downloaded in this checkout; "
              "Validation and Test have labels but no videos.")
        sys.exit(1)

    print(f"labels : {labels_csv}")
    print(f"videos : {videos_dir}")

    ensure_model()
    pool = MediaPipePool(args.workers, face_conf=config.MP_FACE_CONF_THRESH)

    labels_df = pd.read_csv(labels_csv)
    labels_df.columns = [c.strip() for c in labels_df.columns]
    print(f"clips in labels: {len(labels_df)}\n")

    tasks = []
    for _, row in labels_df.iterrows():
        clip_id = str(row["ClipID"]).strip()
        path = clip_path(videos_dir, clip_id)
        if os.path.exists(path):
            tasks.append((clip_id, path, int(row["Engagement"])))

    print(f"clips present on disk: {len(tasks)} "
          f"({len(labels_df) - len(tasks)} not downloaded)")

    # Limit after filtering, and spread the sample across the dataset. Taking
    # the head of the labels file would draw entirely from subjects near the
    # start of the alphabet - and in this checkout the first several thousand
    # rows belong to subjects that were never downloaded at all.
    if args.limit and len(tasks) > args.limit:
        stride = max(1, len(tasks) // args.limit)
        tasks = tasks[::stride][:args.limit]
        print(f"limited to {len(tasks)} clips, sampled every {stride}")
    print()
    if not tasks:
        print("Nothing to extract.")
        sys.exit(1)

    results = []
    no_face = []
    done = 0

    def work(task):
        clip_id, path, engagement = task
        out = features_for_clip(path, pool, args.sample_every)
        return clip_id, engagement, out

    # Clips are independent and detect() releases the GIL, so this parallelises
    # the same way the live pipeline does.
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for clip_id, engagement, out in ex.map(work, tasks):
            done += 1
            if out is None:
                no_face.append(clip_id)
            else:
                features, n_frames = out
                features["clip_id"] = clip_id
                features["person_id"] = clip_id[:6]
                features["engagement"] = engagement
                features["label"] = int(
                    engagement >= config.ENGAGEMENT_POSITIVE_MIN
                )
                features["frames_used"] = n_frames
                results.append(features)
            if done % 100 == 0:
                print(f"  {done}/{len(tasks)} clips...", flush=True)

    pool.close()

    if not results:
        print("No clips yielded a detectable face.")
        sys.exit(1)

    df = pd.DataFrame(results)
    ordered = (config.FEATURE_COLS
               + ["label", "engagement", "clip_id", "person_id", "frames_used"])
    df = df[ordered]

    os.makedirs(config.DATA_DIR, exist_ok=True)
    out_csv = os.path.join(config.DATA_DIR, f"{args.split.lower()}_features.csv")
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)
    print(f"extracted : {len(df)} clips")
    print(f"no face   : {len(no_face)} clips")
    print(f"subjects  : {df['person_id'].nunique()}")
    print(f"saved     : {out_csv}\n")

    print(f"engagement 0-3:\n{df['engagement'].value_counts().sort_index().to_string()}")
    print(f"\nbinary label (engagement >= {config.ENGAGEMENT_POSITIVE_MIN}):")
    counts = df["label"].value_counts().sort_index()
    for value, count in counts.items():
        print(f"  {config.CLASS_NAMES[value]:<10} {count:>5}  "
              f"({count / len(df) * 100:.1f}%)")

    # Subject count bounds what any honest evaluation can claim, so say it here
    # rather than let train_model.py imply more.
    print(f"\nSubjects available for grouped CV: {df['person_id'].nunique()}")


if __name__ == "__main__":
    main()
