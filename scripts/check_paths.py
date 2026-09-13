# scripts/check_paths.py
# Dataset diagnostic - what DAiSEE is actually on disk, and what that bounds.
#
#   python scripts/check_paths.py
#   python scripts/check_paths.py --split Validation
#
# Worth running before extraction or training, because both are limited by the
# answer far more than by anything in their own code. This checkout has 13 of
# DAiSEE's subjects; no amount of tuning gets past that.

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classsense import config                                    # noqa: E402
from scripts.extract_features import resolve_paths, clip_path    # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Check the DAiSEE dataset on disk")
    p.add_argument("--split", default=config.DAISEE_SPLIT)
    p.add_argument("--examples", type=int, default=5)
    args = p.parse_args()

    print("=" * 64)
    print(f"DAiSEE check - split {args.split}")
    print("=" * 64)
    print(f"DAISEE_ROOT: {config.DAISEE_ROOT}")
    if not os.path.isdir(config.DAISEE_ROOT):
        print("\nERROR: that directory does not exist.")
        print("Set the DAISEE_ROOT environment variable to point at it.")
        sys.exit(1)

    labels_csv, videos_dir, tried = resolve_paths(args.split)

    print(f"labels    : {labels_csv}"
          f"{'' if os.path.exists(labels_csv) else '   [MISSING]'}")
    if videos_dir:
        print(f"videos    : {videos_dir}")
    else:
        print("videos    : NOT FOUND. Tried:")
        for candidate in tried:
            print(f"              {candidate}")

    if not os.path.exists(labels_csv):
        sys.exit(1)

    labels_df = pd.read_csv(labels_csv)
    labels_df.columns = [c.strip() for c in labels_df.columns]
    print(f"\nrows in labels CSV : {len(labels_df)}")

    if videos_dir is None:
        print("\nNo videos for this split. Labels alone cannot be extracted.")
        print("Only the Train split was downloaded in this checkout.")
        sys.exit(1)

    found, missing, missing_examples = 0, 0, []
    found_people, missing_people = Counter(), Counter()

    for _, row in labels_df.iterrows():
        clip_id = str(row["ClipID"]).strip()
        person = clip_id[:6]
        if os.path.exists(clip_path(videos_dir, clip_id)):
            found += 1
            found_people[person] += 1
        else:
            missing += 1
            missing_people[person] += 1
            if len(missing_examples) < args.examples:
                missing_examples.append(clip_path(videos_dir, clip_id))

    total = found + missing
    print(f"clips present      : {found}  ({found / total * 100:.1f}%)")
    print(f"clips not on disk  : {missing}")
    print(f"subjects present   : {len(found_people)}")
    print(f"subjects absent    : {len(missing_people)}")

    if missing_examples:
        print(f"\nexample missing paths:")
        for path in missing_examples:
            print(f"  {path}")

    if found:
        present_ids = sorted(found_people)
        print(f"\nsubjects on disk: {', '.join(present_ids)}")

        subset = labels_df[
            labels_df["ClipID"].astype(str).str.strip().str[:6].isin(found_people)
        ]
        available = subset[subset["ClipID"].astype(str).str.strip().apply(
            lambda c: os.path.exists(clip_path(videos_dir, c))
        )]

        print(f"\nengagement distribution over the {len(available)} available clips:")
        for score, count in available["Engagement"].value_counts().sort_index().items():
            print(f"  {score}: {count:>5}  ({count / len(available) * 100:.1f}%)")

        positive = int((available["Engagement"] >= config.ENGAGEMENT_POSITIVE_MIN).sum())
        negative = len(available) - positive
        print(f"\nbinary split at engagement >= {config.ENGAGEMENT_POSITIVE_MIN}:")
        print(f"  {config.CLASS_NAMES[0]:<10} {negative:>5}  "
              f"({negative / len(available) * 100:.1f}%)")
        print(f"  {config.CLASS_NAMES[1]:<10} {positive:>5}  "
              f"({positive / len(available) * 100:.1f}%)")

        print("\n" + "=" * 64)
        print("What this bounds")
        print("=" * 64)
        n_subjects = len(found_people)
        print(f"{n_subjects} subjects means grouped cross-validation holds out")
        print(f"about {max(1, n_subjects // 5)} people per fold. Fold-to-fold variance")
        print("will be large, and a model cannot be shown to generalise to new")
        print("faces on this base. More subjects is the only fix.")


if __name__ == "__main__":
    main()
