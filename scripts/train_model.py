# scripts/train_model.py
# Stage 4 - train the engagement classifier, and report what it is actually worth.
#
#   python scripts/train_model.py
#   python scripts/train_model.py --no-search      # skip the grid search
#
# Two things differ from the previous version, both about honesty rather than
# accuracy:
#
# 1. Evaluation is GroupKFold over subject id. The old 60/20/20 random split
#    drew from only 13 people, so the same face appeared in train and test and
#    the model could recognise the person rather than the behaviour. Every
#    number it printed was inflated by that leak.
#
# 2. The threshold block applied its decision inverted. It computed
#    P(class 0) >= threshold - "is drifting" - and then wrote the result as
#    int(True) == 1, which is the label for Engaged. The two saved confusion
#    matrices were exact transposes of each other, which is the signature of
#    that bug.

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib                                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

from sklearn.ensemble import RandomForestClassifier              # noqa: E402
from sklearn.metrics import (                                    # noqa: E402
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, balanced_accuracy_score, f1_score,
)
from sklearn.model_selection import GroupKFold, GridSearchCV     # noqa: E402
from sklearn.preprocessing import StandardScaler                 # noqa: E402

from classsense import config                                    # noqa: E402

RULE = "=" * 64


def load_features(path):
    df = pd.read_csv(path)
    missing = [c for c in config.FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"ERROR: {path} is missing {missing}.")
        print("It was probably built by an older extractor. Re-run "
              "scripts/extract_features.py.")
        sys.exit(1)
    if "person_id" not in df.columns:
        print(f"ERROR: {path} has no person_id column, so subject-independent "
              "evaluation is impossible. Re-run scripts/extract_features.py.")
        sys.exit(1)
    return df


def out_of_fold_predictions(X, y, groups, params, n_splits):
    """
    Honest predictions: every sample scored by a model that never saw that
    subject.

    Scaling and resampling happen inside the loop, fitted on the training folds
    only. Fitting the scaler on everything first would leak test-fold
    statistics into training - a smaller leak than the subject one, but the
    same kind of mistake.
    """
    from imblearn.over_sampling import SMOTE

    oof_proba = np.zeros(len(y), dtype=float)
    fold_rows = []

    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups), start=1
    ):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train = y[train_idx]

        # SMOTE needs at least a couple of minority samples in the fold.
        counts = np.bincount(y_train, minlength=2)
        if counts.min() >= 6:
            X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

        model = RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1, **params
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]   # P(Engaged)
        oof_proba[test_idx] = proba

        held_out = sorted(set(groups[test_idx]))
        fold_rows.append({
            "fold": fold,
            "subjects": len(held_out),
            "n": len(test_idx),
            "acc": float(((proba >= 0.5).astype(int) == y[test_idx]).mean()),
            "auc": (
                float(roc_auc_score(y[test_idx], proba))
                if len(set(y[test_idx])) > 1 else float("nan")
            ),
        })

    return oof_proba, fold_rows


MIN_AUC_TO_TUNE = 0.65    # below this, a "better" threshold is fitting noise
MIN_GAIN_TO_MOVE = 0.02   # balanced-accuracy gain worth leaving 0.50 for


def pick_threshold(y_true, proba_engaged, auc):
    """
    Choose the cut on P(Engaged) that maximises balanced accuracy - but only
    move off 0.50 when there is real signal to move on.

    The threshold is selected on the same out-of-fold predictions used to
    report the score, so any gain is partly self-congratulation. On a model
    whose AUC is near chance, the "best" threshold is just the one that happens
    to suit this sample's noise, and shipping it makes live behaviour worse
    than the default would be. So: require a discriminating model and a gain
    large enough not to be an artefact, else keep 0.50.
    """
    baseline = balanced_accuracy_score(y_true, (proba_engaged >= 0.5).astype(int))

    best_t, best_score = 0.5, baseline
    for t in np.linspace(0.05, 0.95, 181):
        score = balanced_accuracy_score(y_true, (proba_engaged >= t).astype(int))
        if score > best_score:
            best_t, best_score = float(t), float(score)

    if auc < MIN_AUC_TO_TUNE:
        return 0.5, baseline, (
            f"kept 0.50 - AUC {auc:.3f} is below {MIN_AUC_TO_TUNE}, so the "
            f"apparent best threshold {best_t:.2f} "
            f"(+{best_score - baseline:.3f}) is not distinguishable from noise"
        )
    if best_score - baseline < MIN_GAIN_TO_MOVE:
        return 0.5, baseline, (
            f"kept 0.50 - best alternative {best_t:.2f} gained only "
            f"{best_score - baseline:.3f}"
        )
    return best_t, best_score, f"moved to {best_t:.3f}"


def save_confusion(y_true, y_pred, title, path, cmap):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred),
        display_labels=config.CLASS_NAMES,
    )
    disp.plot(cmap=cmap, values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Train the engagement classifier")
    p.add_argument("--input", default=os.path.join(
        config.DATA_DIR, "train_features.csv"))
    p.add_argument("--no-search", action="store_true",
                   help="skip the grid search and use sane defaults")
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()

    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # ── 1. Load ────────────────────────────────
    print(RULE)
    print("1. Data")
    print(RULE)
    df = load_features(args.input)

    X = df[config.FEATURE_COLS].values
    y = df["label"].values
    groups = df["person_id"].values

    n_subjects = len(set(groups))
    counts = np.bincount(y, minlength=2)
    majority = counts.max() / counts.sum()

    print(f"samples  : {len(df)}")
    print(f"subjects : {n_subjects}")
    print(f"features : {config.FEATURE_COLS}")
    print(f"label    : engagement >= {config.ENGAGEMENT_POSITIVE_MIN}")
    for value, name in enumerate(config.CLASS_NAMES):
        print(f"  {name:<10} {counts[value]:>5}  "
              f"({counts[value] / len(df) * 100:.1f}%)")
    print(f"\nAlways-guess-majority baseline: {majority:.3f} accuracy")

    folds = min(args.folds, n_subjects)
    if folds < args.folds:
        print(f"Only {n_subjects} subjects, so using {folds} folds.")

    # ── 2. Hyperparameters ─────────────────────
    print("\n" + RULE)
    print("2. Hyperparameters")
    print(RULE)
    if args.no_search:
        params = {"n_estimators": 300, "max_depth": 10, "min_samples_split": 5}
        print(f"search skipped, using {params}")
    else:
        # Searched under the same grouping as the evaluation, so the choice of
        # hyperparameters is not itself made on leaked data.
        grid = GridSearchCV(
            RandomForestClassifier(
                class_weight="balanced", random_state=42, n_jobs=-1
            ),
            {
                "n_estimators": [200, 300],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
            },
            cv=GroupKFold(n_splits=folds),
            scoring="balanced_accuracy",
            n_jobs=-1,
        )
        scaled = StandardScaler().fit_transform(X)
        grid.fit(scaled, y, groups=groups)
        params = grid.best_params_
        print(f"best params : {params}")
        print(f"grouped CV  : {grid.best_score_:.3f} balanced accuracy")

    # ── 3. Subject-independent evaluation ──────
    print("\n" + RULE)
    print(f"3. Subject-independent evaluation ({folds}-fold GroupKFold)")
    print(RULE)
    oof_proba, fold_rows = out_of_fold_predictions(X, y, groups, params, folds)

    print(f"{'fold':>5} {'subjects':>9} {'n':>6} {'acc':>7} {'auc':>7}")
    print("-" * 40)
    for row in fold_rows:
        auc = f"{row['auc']:.3f}" if row["auc"] == row["auc"] else "  n/a"
        print(f"{row['fold']:>5} {row['subjects']:>9} {row['n']:>6} "
              f"{row['acc']:>7.3f} {auc:>7}")

    oof_pred = (oof_proba >= 0.5).astype(int)
    oof_acc = float((oof_pred == y).mean())
    oof_bal = balanced_accuracy_score(y, oof_pred)
    oof_auc = roc_auc_score(y, oof_proba)
    oof_f1 = f1_score(y, oof_pred)

    print(f"\nOut-of-fold, default 0.50 threshold:")
    print(classification_report(y, oof_pred, target_names=config.CLASS_NAMES,
                                digits=3, zero_division=0))
    print(f"accuracy          : {oof_acc:.3f}")
    print(f"balanced accuracy : {oof_bal:.3f}")
    print(f"ROC AUC           : {oof_auc:.3f}")
    print(f"F1 (Engaged)      : {oof_f1:.3f}")
    print(f"majority baseline : {majority:.3f}")
    verdict = "beats" if oof_acc > majority else "does NOT beat"
    print(f"\n-> The model {verdict} always guessing the majority class.")

    save_confusion(
        y, oof_pred, "Out-of-fold, threshold 0.50",
        os.path.join(config.MODELS_DIR, "confusion_matrix.png"), "Blues",
    )

    # ── 4. Threshold ───────────────────────────
    print("\n" + RULE)
    print("4. Decision threshold")
    print(RULE)
    best_t, best_bal, thresh_note = pick_threshold(y, oof_proba, oof_auc)
    tuned_pred = (oof_proba >= best_t).astype(int)

    # The comparison the old code got backwards: >= threshold on P(Engaged)
    # means Engaged, which is label 1, so int(True) is correct here only
    # because the probability is of the positive class. Keeping the
    # probability positive-class throughout is what removes the trap.
    print(f"selected threshold : {best_t:.3f} on P(Engaged)")
    print(f"balanced accuracy  : {best_bal:.3f} (0.50 gives {oof_bal:.3f})")
    print(f"decision           : {thresh_note}")
    print()
    print(classification_report(y, tuned_pred, target_names=config.CLASS_NAMES,
                                digits=3, zero_division=0))

    save_confusion(
        y, tuned_pred, f"Out-of-fold, threshold {best_t:.2f}",
        os.path.join(config.MODELS_DIR, "confusion_matrix_tuned.png"), "Oranges",
    )

    # ── 5. Final fit ───────────────────────────
    print(RULE)
    print("5. Final model")
    print(RULE)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    from imblearn.over_sampling import SMOTE
    X_bal, y_bal = SMOTE(random_state=42).fit_resample(X_scaled, y)

    model = RandomForestClassifier(
        class_weight="balanced", random_state=42, n_jobs=-1, **params
    )
    model.fit(X_bal, y_bal)

    print("feature importances:")
    for name, importance in sorted(
        zip(config.FEATURE_COLS, model.feature_importances_),
        key=lambda kv: -kv[1],
    ):
        print(f"  {name:<10} {importance:.3f}  {'#' * int(importance * 60)}")

    # live_detect reads P(class 0) and compares against the stored threshold,
    # so store the threshold in those terms.
    drift_threshold = 1.0 - best_t

    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump(float(drift_threshold), config.THRESH_PATH)

    meta = {
        "features": config.FEATURE_COLS,
        "class_names": config.CLASS_NAMES,
        "engagement_positive_min": config.ENGAGEMENT_POSITIVE_MIN,
        "n_samples": int(len(df)),
        "n_subjects": int(n_subjects),
        "params": params,
        "evaluation": "GroupKFold by person_id",
        "folds": int(folds),
        "oof_accuracy": round(oof_acc, 4),
        "oof_balanced_accuracy": round(float(oof_bal), 4),
        "oof_roc_auc": round(float(oof_auc), 4),
        "majority_baseline": round(float(majority), 4),
        "threshold_p_engaged": round(best_t, 4),
        "threshold_p_drifting": round(float(drift_threshold), 4),
        "threshold_note": thresh_note,
        # Read by scripts/live_detect.py. A model this close to chance would
        # paint red boxes on attentive students often enough to cost the
        # operator their trust in the colours the heuristics get right, so the
        # live pipeline keeps it out of the decision unless asked explicitly.
        "advisory_only": bool(oof_auc < MIN_AUC_TO_TUNE),
        "advisory_reason": (
            f"ROC AUC {oof_auc:.3f} is below {MIN_AUC_TO_TUNE}; accuracy "
            f"{oof_acc:.3f} against a {majority:.3f} majority baseline"
        ),
    }
    with open(config.META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nmodel     -> {config.MODEL_PATH}")
    print(f"scaler    -> {config.SCALER_PATH}")
    print(f"threshold -> {config.THRESH_PATH}  ({drift_threshold:.3f} on P(Drifting))")
    print(f"metadata  -> {config.META_PATH}")

    print("\n" + RULE)
    print("What this model is worth")
    print(RULE)
    print(f"Trained on {len(df)} clips from {n_subjects} subjects.")
    print(f"Out-of-fold accuracy {oof_acc:.3f} against a {majority:.3f} "
          f"majority baseline: a {(oof_acc - majority) * 100:+.1f} point gain.")
    print(f"ROC AUC {oof_auc:.3f}.")
    print()

    if meta["advisory_only"]:
        print("This model is close to chance. Per-fold accuracy ranges")
        lo = min(r["acc"] for r in fold_rows)
        hi = max(r["acc"] for r in fold_rows)
        print(f"{lo:.3f} to {hi:.3f}, and some folds fall below 0.500 - which")
        print("means on some held-out people it is worse than a coin.")
        print()
        print("It is therefore marked advisory_only. live_detect.py will load")
        print("it and show its score, but will NOT let it change a student's")
        print("state unless you pass --use-weak-model. The temporal heuristics")
        print("run regardless and are what the system actually relies on.")
        print()
        print("The limit is the data, not the algorithm: 13 subjects cannot")
        print("support a model that generalises to new faces. Downloading more")
        print("of DAiSEE is the only thing that moves this number.")
    else:
        print("The model discriminates well enough to take part in live")
        print("decisions. It still only speaks when posture and eyes look")
        print("unremarkable - the temporal heuristics outrank it.")


if __name__ == "__main__":
    main()
