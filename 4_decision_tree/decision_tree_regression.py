"""Decision tree regression: California housing prices.

Leakage-free rewrite of decision_tree_regression.ipynb:
  * the train/test split happens first; imputation, feature engineering and
    encoding live inside a Pipeline, so they are fitted on training folds only;
  * three tree variants are compared with 5-fold cross-validation on the
    training set, and the test set is scored exactly once at the end;
  * figures go to outputs/, metrics to outputs/regression_metrics.json.

Run:  python decision_tree_regression.py [--track]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, plot_tree

import tracking

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
SEED = 42
TARGET = "median_house_value"
CAT = ["ocean_proximity"]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def add_ratios(X: pd.DataFrame) -> pd.DataFrame:
    """Per-household ratios (row-wise, so no statistics are learned)."""
    X = X.copy()
    X["rooms_per_household"] = X["total_rooms"] / X["households"]
    X["bedrooms_per_room"] = X["total_bedrooms"] / X["total_rooms"]
    X["population_per_household"] = X["population"] / X["households"]
    return X


def make_pipeline(tree: DecisionTreeRegressor, numeric_cols) -> Pipeline:
    # Trees are invariant to monotonic transforms, so no scaling, Box-Cox or
    # winsorizing: they cannot change where a split falls, only add leakage risk.
    num = numeric_cols + ["rooms_per_household", "bedrooms_per_room", "population_per_household"]
    pre = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), num),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)],
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("ratios", FunctionTransformer(add_ratios)),
        ("pre", pre),
        ("tree", tree),
    ])


def log_target(model):
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=120)
    plt.close(fig)


def main(track: bool):
    OUT.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = pd.read_csv(HERE / "housing.csv")
    capped = (df[TARGET] >= 500_000).mean()
    print(f"Rows: {len(df):,} | missing total_bedrooms: {df['total_bedrooms'].isna().sum()} "
          f"| target capped at 500,001 for {capped:.1%} of rows")

    X, y = df.drop(columns=[TARGET]), df[TARGET]
    numeric_cols = [c for c in X.columns if c not in CAT]
    # Stratify on income bands so train and test share the income distribution.
    income_band = pd.cut(X["median_income"], [0, 1.5, 3, 4.5, 6, np.inf], labels=False)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=income_band)
    print(f"Train {len(X_tr):,} / test {len(X_te):,}")

    # ---- target distribution -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(y_tr, bins=60, ax=ax)
    ax.axvline(500_001, color="crimson", ls="--", label="cap at $500,001")
    ax.set(title="Median house value (training set)", xlabel="USD")
    ax.legend()
    save(fig, "regression_target_distribution.png")

    cv = KFold(5, shuffle=True, random_state=SEED)
    scoring = {"rmse": "neg_root_mean_squared_error", "r2": "r2"}

    # ---- depth curve: how a tree over-fits as it grows ---------------------------
    depths = list(range(2, 26, 2))
    tr_s, va_s = validation_curve(
        make_pipeline(DecisionTreeRegressor(random_state=SEED), numeric_cols), X_tr, y_tr,
        param_name="tree__max_depth", param_range=depths, cv=cv,
        scoring="neg_root_mean_squared_error", n_jobs=-1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(depths, -tr_s.mean(1), "o-", label="training folds")
    ax.plot(depths, -va_s.mean(1), "o-", label="validation folds")
    ax.set(title="Over-fitting as the tree grows (min_samples_leaf=1)", xlabel="max_depth", ylabel="RMSE (USD)")
    ax.legend()
    save(fig, "regression_depth_curve.png")

    # ---- variants ------------------------------------------------------------
    grid = {
        "max_depth": [8, 10, 12, 15, 20, None],
        "min_samples_leaf": [5, 10, 20, 40],
        "max_features": [None, 0.7],
    }
    variants = {}

    # A: the notebook's hand-picked settings, now inside a leakage-free pipeline
    variants["A: baseline (depth 8)"] = make_pipeline(
        DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, min_samples_split=50, random_state=SEED),
        numeric_cols)

    # B: grid search on the raw target
    gs_b = GridSearchCV(make_pipeline(DecisionTreeRegressor(random_state=SEED), numeric_cols),
                        {f"tree__{k}": v for k, v in grid.items()}, cv=cv,
                        scoring="neg_root_mean_squared_error", n_jobs=-1).fit(X_tr, y_tr)
    variants["B: tuned tree"] = gs_b.best_estimator_
    print("B best params:", gs_b.best_params_)

    # C: grid search with a log-transformed target (still scored in dollars)
    gs_c = GridSearchCV(log_target(make_pipeline(DecisionTreeRegressor(random_state=SEED), numeric_cols)),
                        {f"regressor__tree__{k}": v for k, v in grid.items()}, cv=cv,
                        scoring="neg_root_mean_squared_error", n_jobs=-1).fit(X_tr, y_tr)
    variants["C: tuned tree, log target"] = gs_c.best_estimator_
    print("C best params:", gs_c.best_params_)

    # Re-score every variant on identical folds (fair comparison)
    rows = []
    for name, model in variants.items():
        r = cross_validate(model, X_tr, y_tr, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)
        rows.append({"variant": name,
                     "cv_rmse": -r["test_rmse"].mean(), "cv_rmse_std": r["test_rmse"].std(),
                     "cv_r2": r["test_r2"].mean(), "train_r2": r["train_r2"].mean()})
    comp = pd.DataFrame(rows).set_index("variant")
    print("\nCross-validation comparison (training set only):")
    print(comp.round(3).to_string())

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(comp.index, comp["cv_rmse"], xerr=comp["cv_rmse_std"], color=["#9aa5b1", "#3b82f6", "#10b981"])
    ax.invert_yaxis()
    ax.set(title="5-fold CV RMSE by variant (lower is better)", xlabel="RMSE (USD)")
    save(fig, "regression_variant_comparison.png")

    # ---- final model: best CV variant, scored once on the test set ------------
    best_name = comp["cv_rmse"].idxmin()
    baseline = variants["A: baseline (depth 8)"].fit(X_tr, y_tr)
    final = variants[best_name].fit(X_tr, y_tr)

    def scores(model):
        p_tr, p_te = model.predict(X_tr), model.predict(X_te)
        return {"train_r2": r2_score(y_tr, p_tr), "test_r2": r2_score(y_te, p_te),
                "train_rmse": rmse(y_tr, p_tr), "test_rmse": rmse(y_te, p_te),
                "test_mae": mean_absolute_error(y_te, p_te)}, p_te

    base_scores, _ = scores(baseline)
    final_scores, pred = scores(final)
    print(f"\nSelected: {best_name}")
    print(pd.DataFrame({"baseline A": base_scores, "final": final_scores}).round(3).to_string())

    # ---- diagnostic plots ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_te, pred, s=6, alpha=0.3)
    lim = [0, 520_000]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set(xlim=lim, ylim=lim, title=f"Actual vs predicted (test)  R²={final_scores['test_r2']:.3f}",
           xlabel="Actual (USD)", ylabel="Predicted (USD)")
    save(fig, "regression_actual_vs_predicted.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(pred, y_te - pred, s=6, alpha=0.3)
    ax.axhline(0, color="r", ls="--", lw=1)
    ax.set(title="Residuals (test)", xlabel="Predicted (USD)", ylabel="Actual - predicted (USD)")
    save(fig, "regression_residuals.png")

    pi = permutation_importance(final, X_te, y_te, scoring="neg_root_mean_squared_error",
                                n_repeats=5, random_state=SEED, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=X_te.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    imp.plot.barh(ax=ax, color="#3b82f6")
    ax.set(title="Permutation importance (test): RMSE increase when shuffled", xlabel="USD")
    save(fig, "regression_feature_importance.png")

    pipe = final.regressor_ if isinstance(final, TransformedTargetRegressor) else final
    fig, ax = plt.subplots(figsize=(22, 9))
    plot_tree(pipe.named_steps["tree"], feature_names=pipe.named_steps["pre"].get_feature_names_out(),
              max_depth=3, filled=True, rounded=True, fontsize=9, ax=ax)
    ax.set_title(f"Final tree, top 3 levels of {pipe.named_steps['tree'].get_depth()} "
                 f"({pipe.named_steps['tree'].get_n_leaves()} leaves)")
    save(fig, "regression_tree_top_levels.png")

    # ---- save metrics -----------------------------------------------------------
    metrics = {"selected_variant": best_name,
               "best_params_B": gs_b.best_params_, "best_params_C": gs_c.best_params_,
               "cv_comparison": comp.round(4).reset_index().to_dict(orient="records"),
               "baseline_test": base_scores, "final_test": final_scores,
               "permutation_importance_usd": imp.sort_values(ascending=False).round(0).to_dict()}
    (OUT / "regression_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nFigures and metrics written to {OUT}")

    with tracking.tracking(track, "decision-tree-regression", best_name):
        tracking.log_params({"variant": best_name, **(gs_c.best_params_ if best_name.startswith("C")
                                                      else gs_b.best_params_ if best_name.startswith("B") else {})})
        tracking.log_metrics({f"final_{k}": v for k, v in final_scores.items()})
        tracking.log_metrics({f"baseline_{k}": v for k, v in base_scores.items()})
        tracking.log_artifacts(OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", action="store_true", help="log the run to MLflow (DagsHub if configured)")
    main(ap.parse_args().track)
