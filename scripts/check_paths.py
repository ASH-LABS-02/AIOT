# scripts/check_paths.py
# Quick diagnostic to check why clips are being skipped

import os
import pandas as pd
from pathlib import Path

DAISEE_ROOT = r"C:\Users\AR\Downloads\DAiSEE\DAiSEE\DataSet"
SPLIT       = "Validation"
LABELS_CSV  = os.path.join(DAISEE_ROOT, "Labels", f"{SPLIT}Labels.csv")
DATASET_DIR = os.path.join(DAISEE_ROOT, "DataSet", SPLIT)

labels_df = pd.read_csv(LABELS_CSV)

missing  = 0
found    = 0
examples = []

for _, row in labels_df.iterrows():
    clip_id   = row["ClipID"]
    clip_name = Path(clip_id).stem
    person_id = clip_name[:6]
    video_path = os.path.join(DATASET_DIR, person_id, clip_name, clip_id)

    if os.path.exists(video_path):
        found += 1
    else:
        missing += 1
        if len(examples) < 5:
            examples.append(video_path)

print(f"Found  : {found}")
print(f"Missing: {missing}")
print(f"\nExample missing paths:")
for p in examples:
    print(f"  {p}")

# Also show what actually exists in the first person folder
print(f"\nWhat actually exists in DataSet/Train (first 2 folders):")
for person in list(os.scandir(DATASET_DIR))[:2]:
    print(f"  {person.name}/")
    for vid in list(os.scandir(person.path))[:2]:
        print(f"    {vid.name}/")
        for f in list(os.scandir(vid.path))[:2]:
            print(f"      {f.name}")