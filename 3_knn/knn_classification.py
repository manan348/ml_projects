"""KNN classification: credit card fraud detection.

Dataset: Kaggle "Credit Card Fraud Detection" (creditcard.csv), 284,807
transactions, of which 492 (0.17 %) are fraud. Download it from
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place
creditcard.csv in this folder.

Approach
--------
1. Stratified 70/30 train/test split before any modelling.
2. The training set is split again (80/20) into a "fit" part and a
   "validation" part. Every decision (oversampling, features, k, weighting
   and the decision threshold) is made on the validation part only.
3. Several variants are compared on the validation part:
     A  baseline: StandardScaler -> PCA(15) -> SMOTE(1:1) -> KNN(k=7,
        distance weights), decision threshold 0.5
     B  same model, but k, weighting and threshold tuned on validation
     C  like B, without SMOTE
     D  like B, with light SMOTE (fraud = 10 % of non-fraud)
     E  the best of B/C/D with feature changes (drop Time, log(Amount),
        with and without PCA)
4. The winning variant is refit on the full training set and scored once on
   the test set, next to the baseline, using the validation-chosen threshold.

Why a tuned threshold matters: SMOTE balances the training classes, which
inflates predicted fraud probabilities, so the default 0.5 cut-off raises far
too many false alarms. Choosing the threshold on validation data trades a
little recall for much higher precision.

All neighbour searches are done once per variant with k = 25, and smaller
k values are read off the same neighbour lists, so the k grid is cheap.

Run from this folder:
    python knn_classification.py
    python knn_classification.py --track      # log to MLflow / DagsHub (see README)
    python knn_classification.py --fraction 0.1 --output-dir /tmp/smoke   # quick test
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import tracking

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "creditcard.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
TARGET = "Class"

K_GRID = [3, 5, 7, 11, 15, 25, 40]
K_MAX = max(K_GRID)
WEIGHTS = ["uniform", "distance"]
BASELINE_PARAMS = (7, "distance")  # what the original model used
BASELINE_THRESHOLD = 0.5

BASE_CFG = {"smote": 1.0, "pca": 15, "drop_time": False, "log_amount": False}

START = time.time()


def log(msg=""):
    print(f"[{(time.time() - START) / 60:6.1f} min] {msg}", flush=True)


# --------------------------------------------------------------------------
# Data and features
# --------------------------------------------------------------------------
def load_data(path: Path, fraction: float):
    """Load the dataset and split it into features and target."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Download it from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it in this folder."
        )
    df = pd.read_csv(path)
    if fraction < 1:  # stratified subsample, for quick smoke tests only
        df = df.groupby(TARGET, group_keys=False).sample(
            frac=fraction, random_state=RANDOM_STATE
        )
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    counts = df[TARGET].value_counts()
    print(f"Class counts: non-fraud={counts[0]}, fraud={counts[1]} "
          f"({counts[1] / len(df):.3%} fraud)")
    X = df.drop(columns=TARGET).astype("float32")
    return X, df[TARGET]


def prepare_features(X, cfg):
    """Stateless feature changes (no statistics are learned, so no leakage)."""
    X = X.copy()
    if cfg["log_amount"]:
        X["Amount"] = np.log1p(X["Amount"])
    if cfg["drop_time"]:
        X = X.drop(columns="Time")
    return X


# --------------------------------------------------------------------------
# KNN with a shared neighbour search
# --------------------------------------------------------------------------
def build_reference(X_fit, y_fit, cfg):
    """Fit scaler, PCA and SMOTE on X_fit, then index the reference points."""
    Xf = prepare_features(X_fit, cfg)
    scaler = StandardScaler().fit(Xf)
    Z = scaler.transform(Xf)
    pca = None
    if cfg["pca"]:
        pca = PCA(n_components=cfg["pca"], random_state=RANDOM_STATE).fit(Z)
        Z = pca.transform(Z)
    y_ref = np.asarray(y_fit)
    if cfg["smote"]:
        Z, y_ref = SMOTE(sampling_strategy=cfg["smote"],
                         random_state=RANDOM_STATE).fit_resample(Z, y_ref)
    nn = NearestNeighbors(n_neighbors=K_MAX, algorithm="brute", n_jobs=-1).fit(Z)
    return {"cfg": cfg, "scaler": scaler, "pca": pca, "nn": nn,
            "y_ref": np.asarray(y_ref), "n_ref": len(y_ref)}


def score_grid(ref, X_query):
    """Fraud probabilities for every (k, weights) pair from ONE neighbour search."""
    Z = ref["scaler"].transform(prepare_features(X_query, ref["cfg"]))
    if ref["pca"] is not None:
        Z = ref["pca"].transform(Z)
    dist, idx = ref["nn"].kneighbors(Z)
    labels = ref["y_ref"][idx].astype("float64")
    probas = {}
    for w in WEIGHTS:
        wt = np.ones_like(dist, dtype="float64") if w == "uniform" else 1.0 / (dist + 1e-9)
        for k in K_GRID:
            probas[(k, w)] = (wt[:, :k] * labels[:, :k]).sum(1) / wt[:, :k].sum(1)
    return probas


# --------------------------------------------------------------------------
# Threshold and metric helpers
# --------------------------------------------------------------------------
def pr_points(y, p):
    precision, recall, thr = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return precision, recall, thr, f1


def choose_threshold(y, p):
    """Decision threshold that maximises F1 (chosen on validation data only)."""
    _, _, thr, f1 = pr_points(y, p)
    return float(thr[np.argmax(f1)])


def threshold_for_precision(y, p, target):
    """Lowest threshold (highest recall) whose precision reaches the target."""
    precision, recall, thr, _ = pr_points(y, p)
    ok = np.where(precision >= target)[0]
    if len(ok) == 0:
        return None
    return float(thr[ok[np.argmax(recall[ok])]])


def precision_at_recall(y, p, min_recall=0.8):
    precision, recall, _, _ = pr_points(y, p)
    ok = recall >= min_recall
    return float(precision[ok].max()) if ok.any() else float("nan")


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "alerts": int(pred.sum()),
    }


# --------------------------------------------------------------------------
# Variant comparison on the validation split
# --------------------------------------------------------------------------
def evaluate_variant(name, cfg, X_fit, y_fit, X_val, y_val):
    """Fit one variant, tune k / weights / threshold on validation data."""
    log(f"Variant {name}: fitting and searching neighbours...")
    ref = build_reference(X_fit, y_fit, cfg)
    probas = score_grid(ref, X_val)
    ap = {key: average_precision_score(y_val, p) for key, p in probas.items()}
    best = max(ap, key=ap.get)
    p = probas[best]
    thr = choose_threshold(y_val, p)
    m = metrics_at(y_val, p, thr)
    res = {
        "name": name, "cfg": cfg, "best": best, "ap": ap[best], "ap_grid": ap,
        "threshold": thr, "p_at_r80": precision_at_recall(y_val, p),
        "probas": probas, **m,
    }
    log(f"  reference points: {ref['n_ref']}, best k/weights: {best}, "
        f"val AP = {res['ap']:.4f}, F1 = {res['f1']:.4f} "
        f"(precision {res['precision']:.3f}, recall {res['recall']:.3f}) "
        f"at threshold {thr:.3f}")
    return res


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def save_data_plots(y, X_train, out_dir):
    """Class-imbalance and PCA explained-variance plots."""
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(["Non-fraud", "Fraud"], counts.values, color=["steelblue", "crimson"])
    ax.bar_label(bars, fmt="{:,.0f}".format)
    ax.set_yscale("log")
    ax.set_ylabel("Transactions (log scale)")
    ax.set_title(f"Class imbalance ({counts[1] / len(y):.2%} fraud)")
    fig.tight_layout()
    fig.savefig(out_dir / "classification_class_distribution.png", dpi=120)
    plt.close(fig)

    scaled = StandardScaler().fit_transform(X_train)
    pca = PCA(n_components=15, random_state=RANDOM_STATE).fit(scaled)
    cum = pca.explained_variance_ratio_.cumsum()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, 16), pca.explained_variance_ratio_, color="lightsteelblue",
           label="Per component")
    ax.plot(range(1, 16), cum, marker="o", color="darkblue", label="Cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title(f"PCA with 15 components keeps {cum[-1]:.0%} of the variance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "classification_pca_variance.png", dpi=120)
    plt.close(fig)


def save_variant_plots(results, final, out_dir):
    """Variant comparison bars and the (k, weights) heatmap of the winner."""
    names = [r["name"] for r in results]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, [r["ap"] for r in results], 0.4, label="Validation PR-AUC",
           color="steelblue")
    ax.bar(x + 0.2, [r["f1"] for r in results], 0.4,
           label="Validation F1 (tuned threshold)", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_title("Variant comparison on the validation split")
    ax.set_ylim(0, 1.2)
    ax.legend(ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "classification_variant_comparison.png", dpi=120)
    plt.close(fig)

    table = pd.DataFrame(
        [[final["ap_grid"][(k, w)] for k in K_GRID] for w in WEIGHTS],
        index=WEIGHTS, columns=K_GRID,
    )
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.heatmap(table, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
    ax.set_xlabel("n_neighbors")
    ax.set_ylabel("weights")
    ax.set_title(f"Validation PR-AUC by hyperparameters ({final['name']})")
    fig.tight_layout()
    fig.savefig(out_dir / "classification_grid_search.png", dpi=120)
    plt.close(fig)


def save_threshold_plot(y_val, p_val, thr, out_dir):
    """Precision, recall and F1 against the decision threshold (validation)."""
    precision, recall, thresholds, f1 = pr_points(y_val, p_val)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, precision, label="Precision", color="crimson")
    ax.plot(thresholds, recall, label="Recall", color="steelblue")
    ax.plot(thresholds, f1, label="F1", color="black", linestyle="--")
    ax.axvline(thr, color="green", linestyle=":", label=f"Chosen threshold = {thr:.2f}")
    ax.axvline(BASELINE_THRESHOLD, color="gray", linestyle=":", label="Old threshold = 0.50")
    ax.set_xlabel("Decision threshold on fraud probability")
    ax.set_ylabel("Score")
    ax.set_title("Threshold trade-off on the validation split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "classification_threshold_curve.png", dpi=120)
    plt.close(fig)


def save_test_plots(X_train_final, X_test_final, y_test, y_pred, y_proba, cm,
                    before, after, out_dir):
    """Confusion matrix, ROC, PR, before/after bars and a 3-D PCA view."""
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Non-fraud", "Fraud"], yticklabels=["Non-fraud", "Fraud"])
    ax.set_title("KNN confusion matrix (test set)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(out_dir / "classification_confusion_matrix.png", dpi=120)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("KNN ROC curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "classification_roc_curve.png", dpi=120)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="green", label=f"PR curve (AP = {pr_auc:.3f})")
    ax.scatter([after["recall"]], [after["precision"]], color="red", zorder=3,
               label="Chosen operating point")
    ax.scatter([before["recall"]], [before["precision"]], color="gray", zorder=3,
               label="Old operating point")
    ax.axhline(y_test.mean(), color="gray", linestyle="--", label="Random (fraud rate)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("KNN precision-recall curve (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "classification_pr_curve.png", dpi=120)
    plt.close(fig)

    labels = ["Precision", "Recall", "F1"]
    keys = ["precision", "recall", "f1"]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    b1 = ax.bar(x - 0.2, [before[k] for k in keys], 0.4, label="Before (threshold 0.5)",
                color="lightgray")
    b2 = ax.bar(x + 0.2, [after[k] for k in keys], 0.4, label="After (tuned)",
                color="seagreen")
    ax.bar_label(b1, fmt="%.2f")
    ax.bar_label(b2, fmt="%.2f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Fraud class on the test set: before vs after")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "classification_before_after.png", dpi=120)
    plt.close(fig)

    # 3-D view of test predictions. A separate 3-component PCA is fit on the
    # training data for display only; the classifier is not changed.
    scaler = StandardScaler().fit(X_train_final)
    pca_3d = PCA(n_components=3, random_state=RANDOM_STATE).fit(
        scaler.transform(X_train_final))
    X_test_3d = pca_3d.transform(scaler.transform(X_test_final))
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for label, color, name, size in [(0, "tab:blue", "Predicted non-fraud", 4),
                                     (1, "tab:red", "Predicted fraud", 20)]:
        pts = X_test_3d[y_pred == label]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=size,
                   alpha=0.5 if label == 0 else 0.9, label=name)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("KNN predictions on the test set (first 3 principal components)")
    ax.legend(markerscale=1.5)
    fig.tight_layout()
    fig.savefig(out_dir / "classification_3d_pca.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(args):
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_data(DATA_PATH, args.fraction)

    # Split first, so nothing below ever sees the test set until the end.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
    )
    print(f"Train rows: {len(X_train)} (fit {len(X_fit)}, validation {len(X_val)} "
          f"with {int(y_val.sum())} frauds), test rows: {len(X_test)}")
    save_data_plots(y, X_train, out_dir)

    # ---- Variants A to D on the validation split --------------------------
    results = {}
    base = evaluate_variant("B: SMOTE 1:1 + tuned threshold", BASE_CFG,
                            X_fit, y_fit, X_val, y_val)
    results["B"] = base
    results["C"] = evaluate_variant("C: no SMOTE", {**BASE_CFG, "smote": None},
                                    X_fit, y_fit, X_val, y_val)
    results["D"] = evaluate_variant("D: SMOTE 0.1", {**BASE_CFG, "smote": 0.1},
                                    X_fit, y_fit, X_val, y_val)

    # Variant A: the original model (k=7, distance, threshold 0.5), same fit as B
    a_p = base["probas"][BASELINE_PARAMS]
    a_metrics = metrics_at(y_val, a_p, BASELINE_THRESHOLD)
    a_row = {"name": "A: baseline (thr 0.5)", "ap": average_precision_score(y_val, a_p),
             **a_metrics, "p_at_r80": precision_at_recall(y_val, a_p)}

    # ---- Variant E: feature changes on the best of B/C/D -------------------
    best_key = max("BCD", key=lambda k: results[k]["ap"])
    best_cfg = results[best_key]["cfg"]
    log(f"Best of B/C/D by validation PR-AUC: {best_key}. Testing feature changes on it.")
    results["E1"] = evaluate_variant(
        f"E1: {best_key} + no Time, log Amount, PCA 15",
        {**best_cfg, "drop_time": True, "log_amount": True},
        X_fit, y_fit, X_val, y_val)
    results["E2"] = evaluate_variant(
        f"E2: {best_key} + no Time, log Amount, no PCA",
        {**best_cfg, "drop_time": True, "log_amount": True, "pca": None},
        X_fit, y_fit, X_val, y_val)

    ordered = [a_row] + [results[k] for k in ["B", "C", "D", "E1", "E2"]]
    print("\nValidation comparison (thresholds tuned on validation, so rows B to E "
          "are slightly optimistic):")
    print(f"  {'variant':44s} {'PR-AUC':>7s} {'prec':>6s} {'recall':>7s} {'F1':>6s} "
          f"{'prec@R>=0.8':>12s}")
    for r in ordered:
        print(f"  {r['name']:44s} {r['ap']:7.4f} {r['precision']:6.3f} "
              f"{r['recall']:7.3f} {r['f1']:6.3f} {r['p_at_r80']:12.3f}")

    final_key = max(["B", "C", "D", "E1", "E2"], key=lambda k: results[k]["ap"])
    final = results[final_key]
    thr = final["threshold"]
    p_val = final["probas"][final["best"]]
    thr_p60 = threshold_for_precision(y_val, p_val, 0.6)
    print(f"\nSelected variant: {final['name']}")
    print(f"  k = {final['best'][0]}, weights = {final['best'][1]}, "
          f"F1-optimal threshold = {thr:.3f}")
    if thr_p60 is not None:
        print(f"  Alternative: threshold {thr_p60:.3f} gives precision >= 0.6 on validation")
    save_threshold_plot(y_val, p_val, thr, out_dir)

    # ---- Final evaluation on the test set ---------------------------------
    log("Refitting on the full training set and scoring the test set...")
    test_cache = {}

    def test_probas(cfg):
        key = tuple(sorted(cfg.items(), key=lambda kv: kv[0]))
        if key not in test_cache:
            ref = build_reference(X_train, y_train, cfg)
            test_cache[key] = score_grid(ref, X_test)
            log(f"  scored test set with {ref['n_ref']} reference points")
        return test_cache[key]

    probas_final = test_probas(final["cfg"])[final["best"]]
    probas_base = test_probas(BASE_CFG)[BASELINE_PARAMS]

    before = metrics_at(y_test, probas_base, BASELINE_THRESHOLD)
    after = metrics_at(y_test, probas_final, thr)
    y_pred = (probas_final >= thr).astype(int)

    print("\n=== Test set: baseline (SMOTE 1:1, k=7, distance, threshold 0.5) ===")
    print(confusion_matrix(y_test, (probas_base >= BASELINE_THRESHOLD).astype(int)))
    print(f"  precision {before['precision']:.4f}, recall {before['recall']:.4f}, "
          f"F1 {before['f1']:.4f}, PR-AUC {average_precision_score(y_test, probas_base):.4f}, "
          f"ROC-AUC {roc_auc_score(y_test, probas_base):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n=== Test set: {final['name']} (threshold {thr:.3f}) ===")
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4,
                                target_names=["Non-fraud", "Fraud"]))
    # Both AUC scores use predicted probabilities, not hard labels.
    print(f"ROC-AUC:                     {roc_auc_score(y_test, probas_final):.4f}")
    print(f"PR-AUC (average precision):  {average_precision_score(y_test, probas_final):.4f}")

    if thr_p60 is not None:
        alt = metrics_at(y_test, probas_final, thr_p60)
        print(f"\nAlternative operating point (threshold {thr_p60:.3f}, chosen for "
              f"validation precision >= 0.6): precision {alt['precision']:.4f}, "
              f"recall {alt['recall']:.4f}, F1 {alt['f1']:.4f}, alerts {alt['alerts']}")

    save_variant_plots(ordered, final, out_dir)
    X_train_final = prepare_features(X_train, final["cfg"])
    X_test_final = prepare_features(X_test, final["cfg"])
    save_test_plots(X_train_final, X_test_final, y_test, y_pred, probas_final, cm,
                    before, after, out_dir)
    log(f"Plots saved to: {out_dir}")

    # Experiment tracking (no-op unless started with --track)
    tracking.log_params({
        "variant": final["name"], "n_neighbors": final["best"][0],
        "weights": final["best"][1], "threshold": round(thr, 4),
        "smote": final["cfg"]["smote"], "pca": final["cfg"]["pca"],
        "drop_time": final["cfg"]["drop_time"], "log_amount": final["cfg"]["log_amount"],
        "n_train": len(X_train), "n_test": len(X_test),
        "fraction": args.fraction, "random_state": RANDOM_STATE,
    })
    tracking.log_metrics({
        "test_precision": after["precision"], "test_recall": after["recall"],
        "test_f1": after["f1"],
        "test_pr_auc": average_precision_score(y_test, probas_final),
        "test_roc_auc": roc_auc_score(y_test, probas_final),
        "baseline_test_precision": before["precision"],
        "baseline_test_recall": before["recall"],
        "baseline_test_f1": before["f1"],
        **{f"val_pr_auc_{k}": results[k]["ap"] for k in ["B", "C", "D", "E1", "E2"]},
    })
    tracking.log_artifacts(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="use a stratified fraction of the data (smoke tests)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--track", action="store_true",
                        help="log params, metrics and plots to MLflow / DagsHub")
    cli_args = parser.parse_args()
    with tracking.tracking(cli_args.track, "knn-fraud-classification", "knn_classification"):
        main(cli_args)
