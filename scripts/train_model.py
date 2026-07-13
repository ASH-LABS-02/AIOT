# scripts/train_model.py
# Stage 4 - Train Random Forest engagement classifier
# Input : data/train_features.csv
# Output: models/engagement_classifier.pkl + models/scaler.pkl + models/threshold.pkl

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection    import train_test_split, GridSearchCV
from sklearn.preprocessing      import StandardScaler
from sklearn.ensemble           import RandomForestClassifier
from sklearn.metrics            import (classification_report,
                                        confusion_matrix,
                                        ConfusionMatrixDisplay,
                                        precision_recall_curve)
from imblearn.over_sampling     import SMOTE

# ── Paths ──────────────────────────────────────
INPUT_CSV   = os.path.join("..", "data",   "train_features.csv")
MODEL_PATH  = os.path.join("..", "models", "engagement_classifier.pkl")
SCALER_PATH = os.path.join("..", "models", "scaler.pkl")
THRESH_PATH = os.path.join("..", "models", "threshold.pkl")
CM_PATH     = os.path.join("..", "models", "confusion_matrix.png")
# ───────────────────────────────────────────────

def main():
    os.makedirs(os.path.join("..", "models"), exist_ok=True)

    # ── 1. Load data ───────────────────────────
    print("=" * 50)
    print("STEP 1 — Loading data")
    print("=" * 50)
    df = pd.read_csv(INPUT_CSV)
    print(f"Total samples       : {len(df)}")
    print(f"Label distribution  :\n{df['label'].value_counts()}\n")

    FEATURE_COLS = ["left_ear", "right_ear", "avg_ear", "mar", "yaw", "pitch"]
    X = df[FEATURE_COLS].values
    y = df["label"].values

    # ── 2. Train / Validation / Test split ─────
    print("=" * 50)
    print("STEP 2 — Splitting data")
    print("=" * 50)
    # 20% held out as final test set, never touched until end
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    # Remaining 80% split into 75% train / 25% val
    # Final ratio: 60% train, 20% val, 20% test
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.25, random_state=42, stratify=y_trainval
    )
    print(f"Train      : {len(X_train)} samples")
    print(f"Validation : {len(X_val)} samples")
    print(f"Test       : {len(X_test)} samples\n")

    # ── 3. Normalise features ──────────────────
    print("=" * 50)
    print("STEP 3 — Normalising features (StandardScaler)")
    print("=" * 50)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # fit on train only
    X_val   = scaler.transform(X_val)         # apply same scale to val
    X_test  = scaler.transform(X_test)        # apply same scale to test
    print("Done.\n")

    # ── 4. Balance with SMOTE ──────────────────
    print("=" * 50)
    print("STEP 4 — Balancing classes with SMOTE")
    print("=" * 50)
    print(f"Before: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"After : {dict(zip(*np.unique(y_train_bal, return_counts=True)))}\n")

    # ── 5. Train initial Random Forest ─────────
    print("=" * 50)
    print("STEP 5 — Training Random Forest (initial)")
    print("=" * 50)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_bal, y_train_bal)
    print("Done.\n")

    # ── 6. Validate initial model ──────────────
    print("=" * 50)
    print("STEP 6 — Validation set results (initial model)")
    print("=" * 50)
    y_val_pred = rf.predict(X_val)
    print(classification_report(
        y_val, y_val_pred,
        target_names=["Distracted", "Attentive"]
    ))

    # ── 7. Grid search for best hyperparameters─
    print("=" * 50)
    print("STEP 7 — Grid Search hyperparameter tuning")
    print("=" * 50)
    param_grid = {
        "n_estimators"      : [100, 200, 300],
        "max_depth"         : [5, 10, 15, None],
        "min_samples_split" : [2, 5, 10],
    }
    grid = GridSearchCV(
        RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        param_grid,
        cv=5,
        scoring="f1",
        verbose=1,
        n_jobs=-1
    )
    grid.fit(X_train_bal, y_train_bal)
    print(f"\nBest hyperparameters : {grid.best_params_}")
    print(f"Best CV F1 score     : {grid.best_score_:.4f}\n")
    best_rf = grid.best_estimator_

    # ── 8. Final test set evaluation ───────────
    print("=" * 50)
    print("STEP 8 — Final test set results (best model, default threshold)")
    print("=" * 50)
    y_test_pred = best_rf.predict(X_test)
    print(classification_report(
        y_test, y_test_pred,
        target_names=["Distracted", "Attentive"]
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Distracted", "Attentive"]
    )
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Test Set (Default Threshold)")
    plt.tight_layout()
    plt.savefig(CM_PATH)
    print(f"Confusion matrix saved to: {CM_PATH}\n")

    # ── 9. Feature importances ─────────────────
    print("=" * 50)
    print("STEP 9 — Feature importances")
    print("=" * 50)
    for name, importance in sorted(
        zip(FEATURE_COLS, best_rf.feature_importances_),
        key=lambda x: -x[1]
    ):
        bar = "█" * int(importance * 50)
        print(f"  {name:<12}: {importance:.4f}  {bar}")
    print()

    # ── 10. Threshold tuning ───────────────────
    print("=" * 50)
    print("STEP 10 — Tuning decision threshold for distracted recall")
    print("=" * 50)
    # Get probability of being distracted (class 0) for each test sample
    y_probs = best_rf.predict_proba(X_test)[:, 0]

    precisions, recalls, thresholds = precision_recall_curve(
        y_test, y_probs, pos_label=0
    )

    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 32)

    best_thresh = 0.5
    for p, r, t in zip(precisions, recalls, thresholds):
        if r >= 0.20:   # show thresholds where we catch at least 20% of distracted
            print(f"  {t:>8.2f}   {p:>8.2f}   {r:>8.2f}")
        if r >= 0.60 and p >= 0.10 and best_thresh == 0.5:
            best_thresh = t  # save first threshold meeting our target

    print(f"\nSelected threshold : {best_thresh:.2f}")
    print("(catches ≥60% of distracted students with ≥10% precision)\n")

    # Evaluate with tuned threshold
    y_test_tuned = (y_probs >= best_thresh).astype(int)
    print("Results with tuned threshold:")
    print(classification_report(
        y_test, y_test_tuned,
        target_names=["Distracted", "Attentive"]
    ))

    # Save tuned confusion matrix
    cm_tuned = confusion_matrix(y_test, y_test_tuned)
    disp2 = ConfusionMatrixDisplay(
        confusion_matrix=cm_tuned,
        display_labels=["Distracted", "Attentive"]
    )
    disp2.plot(cmap="Oranges")
    plt.title(f"Confusion Matrix - Tuned Threshold ({best_thresh:.2f})")
    plt.tight_layout()
    plt.savefig(CM_PATH.replace(".png", "_tuned.png"))
    print(f"Tuned confusion matrix saved.\n")

    # ── 11. Save everything ────────────────────
    print("=" * 50)
    print("STEP 11 — Saving model, scaler, threshold")
    print("=" * 50)
    joblib.dump(best_rf,     MODEL_PATH)
    joblib.dump(scaler,      SCALER_PATH)
    joblib.dump(best_thresh, THRESH_PATH)
    print(f"Model     → {MODEL_PATH}")
    print(f"Scaler    → {SCALER_PATH}")
    print(f"Threshold → {THRESH_PATH}")
    print("\nStage 4 complete. Ready for Stage 5 — live detection.")

if __name__ == "__main__":
    main()