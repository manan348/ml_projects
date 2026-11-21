"""KNN regression: predicting concrete compressive strength.

Dataset: UCI Concrete Compressive Strength (Concrete_Data.xls), 1,030 rows.

Pipeline
--------
1. Load the data and give the columns short names.
2. Merge rows with identical mix recipes (mean target) so the same recipe
   cannot land in both the train and the test set.
3. Make an 80/20 train/test split.
4. Optionally drop training outliers (|z| > 3), using statistics from the
   training set only. The test set is never filtered.
5. Tune StandardScaler + KNeighborsRegressor with 5-fold GridSearchCV on the
   training set.
6. Evaluate the selected model once on the held-out test set.

Run from this folder:  python knn_regression.py
Log to MLflow / DagsHub:  python knn_regression.py --track   (see README)
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import tracking

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Concrete_Data.xls"
OUTPUT_DIR = BASE_DIR / "outputs"

COLUMN_NAMES = [
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_agg",
    "fine_agg",
    "age",
    "strength",
]
TARGET = "strength"


def load_data(path: Path) -> pd.DataFrame:
    """Load the dataset, rename columns, and merge duplicate mix recipes."""
    df = pd.read_excel(path)
    df.columns = COLUMN_NAMES
    print(f"Raw shape: {df.shape}")
    print(f"Exact duplicate rows: {df.duplicated().sum()}")

    # Rows that share the same features (same recipe and age) are averaged
    # into one row. This is done before the split so a recipe cannot appear
    # in both train and test.
    features = [c for c in COLUMN_NAMES if c != TARGET]
    df = df.groupby(features, as_index=False)[TARGET].mean()
    print(f"Shape after merging duplicate recipes: {df.shape}")
    return df


def remove_train_outliers(X_train, y_train, threshold=3.0, verbose=True):
    """Drop training rows where any column has |z| > threshold.

    The mean and standard deviation come from the training data only, so the
    test set is untouched and remains a fair estimate of real-world error.
    """
    train = X_train.assign(**{TARGET: y_train})
    z = np.abs((train - train.mean()) / train.std())
    mask = ~(z > threshold).any(axis=1)
    if verbose:
        print(f"Training outliers removed: {(~mask).sum()} of {len(train)}")
    return X_train[mask], y_train[mask]


def fair_cv_rmse(X, y, params, drop_outliers, n_splits=5):
    """CV RMSE where outliers are dropped from each training fold only.

    Every candidate is validated on the same, unfiltered validation folds, so
    the "with" and "without outlier removal" scores can be compared directly.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        if drop_outliers:
            X_tr, y_tr = remove_train_outliers(X_tr, y_tr, verbose=False)
        model = Pipeline(
            [("scaler", StandardScaler()), ("knn", KNeighborsRegressor())]
        ).set_params(**params)
        model.fit(X_tr, y_tr)
        pred = model.predict(X.iloc[val_idx])
        scores.append(mean_squared_error(y.iloc[val_idx], pred) ** 0.5)
    return float(np.mean(scores))


def tune_knn(X_train, y_train) -> GridSearchCV:
    """Grid-search k, weighting and distance metric with 5-fold CV.

    The scaler is inside the pipeline, so each CV fold is scaled with
    statistics from its own training part only.
    """
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("knn", KNeighborsRegressor())]
    )
    param_grid = {
        "knn__n_neighbors": list(range(1, 21)),
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],  # 1 = Manhattan, 2 = Euclidean
    }
    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid


def save_eda_plots(df):
    """Save data-exploration plots: target distribution and correlations."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df[TARGET], kde=True, color="steelblue", ax=ax)
    ax.set_xlabel("Compressive strength (MPa)")
    ax.set_title("Distribution of concrete compressive strength")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "regression_target_distribution.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, ax=ax)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "regression_correlation_heatmap.png", dpi=120)
    plt.close(fig)


def save_comparison_plots(fair_all, fair_clean, grid, X_test, y_test):
    """Save the outlier-handling comparison and permutation importances."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(["All training rows", "Outliers removed"], [fair_all, fair_clean],
                  color=["gray", "seagreen"])
    ax.bar_label(bars, fmt="%.2f")
    ax.set_ylabel("5-fold CV RMSE (MPa)")
    ax.set_title("Effect of removing training outliers")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "regression_outlier_comparison.png", dpi=120)
    plt.close(fig)

    # Permutation importance on the test set: how much does shuffling each
    # feature increase the error of the final model?
    imp = permutation_importance(
        grid.best_estimator_, X_test, y_test, n_repeats=30,
        random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error",
    )
    order = imp.importances_mean.argsort()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(X_test.columns[order], imp.importances_mean[order],
            xerr=imp.importances_std[order], color="darkorange")
    ax.set_xlabel("Increase in test RMSE when the feature is shuffled (MPa)")
    ax.set_title("Permutation feature importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "regression_feature_importance.png", dpi=120)
    plt.close(fig)
    print("\nPermutation importance (increase in RMSE, MPa):")
    for i in imp.importances_mean.argsort()[::-1]:
        print(f"  {X_test.columns[i]:18s} {imp.importances_mean[i]:6.2f}")


def save_plots(grid, y_test, y_pred):
    """Save diagnostic plots to the outputs folder."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    best = grid.best_params_
    label = f"k={best['knn__n_neighbors']}, {best['knn__weights']}, p={best['knn__p']}"

    # Actual vs predicted
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, color="royalblue", alpha=0.7)
    lims = [y_test.min(), y_test.max()]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_xlabel("Actual strength (MPa)")
    ax.set_ylabel("Predicted strength (MPa)")
    ax.set_title(f"KNN regression: actual vs predicted ({label})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "regression_actual_vs_predicted.png", dpi=120)
    plt.close(fig)

    # Residuals
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_pred, residuals, color="seagreen", alpha=0.6)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Predicted strength (MPa)")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Residuals on the test set")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "regression_residuals.png", dpi=120)
    plt.close(fig)

    # CV RMSE as a function of k, using the best weights and p
    res = pd.DataFrame(grid.cv_results_)
    sel = res[
        (res["param_knn__weights"] == best["knn__weights"])
        & (res["param_knn__p"] == best["knn__p"])
    ].sort_values("param_knn__n_neighbors")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sel["param_knn__n_neighbors"], -sel["mean_test_score"], marker="o")
    ax.axvline(best["knn__n_neighbors"], color="r", linestyle="--", label="Selected k")
    ax.set_xlabel("Number of neighbours (k)")
    ax.set_ylabel("5-fold CV RMSE (MPa)")
    ax.set_title(f"CV error vs k ({best['knn__weights']} weights, p={best['knn__p']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "regression_cv_rmse_vs_k.png", dpi=120)
    plt.close(fig)

    print(f"Plots saved to: {OUTPUT_DIR}")


def main():
    df = load_data(DATA_PATH)

    print("\nCorrelation of each feature with strength:")
    print(df.corr()[TARGET].drop(TARGET).sort_values().round(3).to_string())

    X = df.drop(columns=TARGET)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    print(f"\nTrain rows: {len(X_train)}, test rows: {len(X_test)}")

    # Compare two training sets with cross-validation only (no test set use).
    print("\nTuning on the full training set...")
    grid_all = tune_knn(X_train, y_train)

    print("Tuning on the training set without outliers...")
    X_train_clean, y_train_clean = remove_train_outliers(X_train, y_train)
    grid_clean = tune_knn(X_train_clean, y_train_clean)

    print("\nBest hyperparameters per variant:")
    print(f"  All training rows: {grid_all.best_params_}")
    print(f"  Outliers removed:  {grid_clean.best_params_}")

    # The grid-search CV scores of the two variants are not comparable: the
    # outlier-free folds also have easier validation rows. Re-score both on
    # identical, unfiltered validation folds before choosing.
    fair_all = fair_cv_rmse(X_train, y_train, grid_all.best_params_, drop_outliers=False)
    fair_clean = fair_cv_rmse(X_train, y_train, grid_clean.best_params_, drop_outliers=True)
    print("\nFair CV comparison (same unfiltered validation folds):")
    print(f"  All training rows: CV RMSE = {fair_all:.2f}")
    print(f"  Outliers removed:  CV RMSE = {fair_clean:.2f}")

    grid = grid_clean if fair_clean < fair_all else grid_all
    chosen = "outliers removed" if grid is grid_clean else "all training rows"
    print(f"\nSelected variant: {chosen}")

    # Single evaluation on the untouched test set
    y_pred = grid.predict(X_test)
    test_rmse = mean_squared_error(y_test, y_pred) ** 0.5
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    cv_rmse = min(fair_all, fair_clean)

    print("\nHeld-out test set performance:")
    print(f"  RMSE: {test_rmse:.2f} MPa")
    print(f"  MAE:  {test_mae:.2f} MPa")
    print(f"  R²:   {test_r2:.3f}")

    # Generalisation check. Training R² is not used: with distance weighting
    # every training point is its own nearest neighbour (distance 0), so
    # training R² is always 1.0 and says nothing about overfitting.
    print("\nGeneralisation check (CV vs test):")
    gap = test_rmse - cv_rmse
    print(f"  CV RMSE = {cv_rmse:.2f}, test RMSE = {test_rmse:.2f}, gap = {gap:+.2f} MPa")
    if abs(gap) / cv_rmse < 0.10:
        print("  CV and test scores agree: the model generalises as expected.")
    else:
        print("  CV and test scores differ noticeably: the estimate may be unstable.")

    save_plots(grid, y_test, y_pred)
    save_eda_plots(df)
    save_comparison_plots(fair_all, fair_clean, grid, X_test, y_test)

    # Experiment tracking (no-op unless started with --track)
    tracking.log_params({
        **{k.replace("knn__", ""): v for k, v in grid.best_params_.items()},
        "training_set": chosen,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "random_state": RANDOM_STATE,
    })
    tracking.log_metrics({
        "cv_rmse": cv_rmse, "test_rmse": test_rmse,
        "test_mae": test_mae, "test_r2": test_r2,
    })
    tracking.log_artifacts(OUTPUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KNN regression on concrete strength")
    parser.add_argument("--track", action="store_true",
                        help="log params, metrics and plots to MLflow / DagsHub")
    args = parser.parse_args()
    with tracking.tracking(args.track, "knn-concrete-regression", "knn_regression"):
        main()
