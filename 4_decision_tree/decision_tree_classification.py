"""Decision tree classification: late deliveries and bad reviews (Olist).

Leakage-free rewrite of decision_tree_classification.ipynb. Two targets:
  * is_late    - the order arrives after the promised date.
                 Predicted at purchase time: only purchase-time features.
  * bad_review - the customer gives 1-3 stars.
                 Predicted after delivery: purchase-time features plus the
                 delivery outcome (days late, delivery time). Never the score.

What changed versus the notebook:
  * one row per ORDER (the notebook's merge gave one row per item x payment,
    so the same order landed in both train and test), on all delivered orders
    instead of the first 15,000 merged rows;
  * no outcome columns as inputs (review_score, delivery dates, review dates);
  * encoding / imputation / SMOTE live inside a pipeline fitted on CV folds;
  * the decision threshold is tuned on out-of-fold training predictions, and
    the test set is scored once.
MultiOutputClassifier just fits one independent tree per target, so the two
targets are modelled separately here, each with its own valid feature set.

Run:  python decision_tree_classification.py [--track]
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
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate,
                                     train_test_split, validation_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

import tracking

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
SEED = 42

CAT = ["customer_state", "seller_state", "main_category", "payment_type"]
PURCHASE_NUM = [
    "purchase_month", "purchase_dow", "purchase_hour", "approval_hours", "promised_days",
    "seller_deadline_days", "n_items", "n_products", "n_sellers", "total_price", "total_freight",
    "freight_ratio", "total_weight_g", "total_volume_cm3", "payment_value", "max_installments",
    "n_payments", "same_state", "distance_km",
]
DELIVERY_NUM = ["carrier_days", "delivery_days", "days_late", "is_late"]
FEATURES = {"is_late": PURCHASE_NUM, "bad_review": PURCHASE_NUM + DELIVERY_NUM}


# ---------------------------------------------------------------------------
# Data: one row per delivered order
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def build_orders() -> pd.DataFrame:
    rd = lambda name: pd.read_csv(HERE / f"olist_{name}_dataset.csv")
    dates = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
             "order_delivered_customer_date", "order_estimated_delivery_date"]
    orders = rd("orders")
    orders[dates] = orders[dates].apply(pd.to_datetime)
    orders = orders[(orders.order_status == "delivered")].dropna(
        subset=["order_approved_at", "order_delivered_customer_date"])

    # Items + products, aggregated per order
    items = rd("order_items").merge(rd("products"), on="product_id", how="left")
    items["shipping_limit_date"] = pd.to_datetime(items["shipping_limit_date"])
    items["volume"] = items.product_length_cm * items.product_height_cm * items.product_width_cm
    tr = pd.read_csv(HERE / "product_category_name_translation.csv")
    items = items.merge(tr, on="product_category_name", how="left")
    items["category"] = items.product_category_name_english.fillna("unknown")
    agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"), n_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"), total_price=("price", "sum"),
        total_freight=("freight_value", "sum"), total_weight_g=("product_weight_g", "sum"),
        total_volume_cm3=("volume", "sum"), shipping_limit=("shipping_limit_date", "max"))
    main = (items.sort_values("price", ascending=False).drop_duplicates("order_id")
            .set_index("order_id")[["seller_id", "category"]].rename(columns={"category": "main_category"}))

    pay = rd("order_payments")
    pay_agg = pay.groupby("order_id").agg(payment_value=("payment_value", "sum"),
                                          max_installments=("payment_installments", "max"),
                                          n_payments=("payment_sequential", "count"))
    pay_agg["payment_type"] = (pay.sort_values("payment_value", ascending=False)
                               .drop_duplicates("order_id").set_index("order_id")["payment_type"])

    reviews = rd("order_reviews").groupby("order_id")["review_score"].mean()

    # Customer / seller location -> state and straight-line distance
    geo = (rd("geolocation").query("-35 < geolocation_lat < 6 and -75 < geolocation_lng < -33")
           .groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].median())
    cust = rd("customers").set_index("customer_id")[["customer_zip_code_prefix", "customer_state"]]
    sell = rd("sellers").set_index("seller_id")[["seller_zip_code_prefix", "seller_state"]]

    df = (orders.set_index("order_id").join(agg, how="inner").join(main).join(pay_agg)
          .join(reviews).join(cust, on="customer_id").join(sell, on="seller_id"))
    c = geo.reindex(df.customer_zip_code_prefix).to_numpy()
    s = geo.reindex(df.seller_zip_code_prefix).to_numpy()
    df["distance_km"] = haversine_km(c[:, 0], c[:, 1], s[:, 0], s[:, 1])

    p = df.order_purchase_timestamp
    day = pd.Timedelta(days=1)
    out = pd.DataFrame(index=df.index)
    # --- purchase-time features
    out["purchase_month"], out["purchase_dow"], out["purchase_hour"] = p.dt.month, p.dt.dayofweek, p.dt.hour
    out["approval_hours"] = (df.order_approved_at - p) / pd.Timedelta(hours=1)
    out["promised_days"] = (df.order_estimated_delivery_date - p) / day
    out["seller_deadline_days"] = (df.shipping_limit - p) / day
    for col in ["n_items", "n_products", "n_sellers", "total_price", "total_freight", "total_weight_g",
                "total_volume_cm3", "payment_value", "max_installments", "n_payments", "distance_km",
                "customer_state", "seller_state", "main_category", "payment_type"]:
        out[col] = df[col]
    out["freight_ratio"] = df.total_freight / df.total_price
    out["same_state"] = (df.customer_state == df.seller_state).astype(int)
    # --- delivery outcome (only used for bad_review)
    out["carrier_days"] = (df.order_delivered_carrier_date - p) / day
    out["delivery_days"] = (df.order_delivered_customer_date - p) / day
    out["days_late"] = (df.order_delivered_customer_date.dt.normalize() - df.order_estimated_delivery_date) / day
    # --- targets
    out["is_late"] = (out["days_late"] > 0).astype(int)
    out["review_score"] = df.review_score
    out = out.dropna(subset=["review_score"])
    out["bad_review"] = (out["review_score"] <= 3).astype(int)
    return out


# ---------------------------------------------------------------------------
# Modelling helpers
# ---------------------------------------------------------------------------
def preprocessor(num_cols):
    return ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), num_cols),
         ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200,
                               sparse_output=False), CAT)],
        verbose_feature_names_out=False)


def tree_pipe(num_cols, smote=False, **tree_kw):
    tree = DecisionTreeClassifier(random_state=SEED, **tree_kw)
    if smote:
        return ImbPipeline([("pre", preprocessor(num_cols)), ("smote", SMOTE(random_state=SEED)), ("tree", tree)])
    return Pipeline([("pre", preprocessor(num_cols)), ("tree", tree)])


def best_f1_threshold(y, proba):
    prec, rec, thr = precision_recall_curve(y, proba)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i]), (prec, rec, thr, f1)


def test_scores(y, proba, thr):
    pred = (proba >= thr).astype(int)
    return {"threshold": thr, "pr_auc": average_precision_score(y, proba), "roc_auc": roc_auc_score(y, proba),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred), "positives_flagged": int(pred.sum())}, pred


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=120)
    plt.close(fig)


def run_target(target, X_tr, X_te, y_tr, y_te):
    print(f"\n================ {target} ================")
    num = FEATURES[target]
    cols = num + CAT
    Xtr, Xte = X_tr[cols], X_te[cols]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    # Depth curve: training vs validation PR-AUC
    depths = [2, 4, 6, 8, 10, 12, 15, 20, 25]
    tr_s, va_s = validation_curve(tree_pipe(num, min_samples_leaf=1), Xtr, y_tr, param_name="tree__max_depth",
                                  param_range=depths, cv=cv, scoring="average_precision", n_jobs=-1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(depths, tr_s.mean(1), "o-", label="training folds")
    ax.plot(depths, va_s.mean(1), "o-", label="validation folds")
    ax.set(title=f"{target}: over-fitting as the tree grows", xlabel="max_depth", ylabel="PR-AUC")
    ax.legend()
    save(fig, f"classification_{target}_depth_curve.png")

    variants = {}
    # A: notebook settings (depth 8, balanced classes), threshold 0.5
    variants["A: baseline (depth 8, balanced)"] = tree_pipe(num, max_depth=8, class_weight="balanced")
    # B: grid search, PR-AUC
    gs_b = GridSearchCV(tree_pipe(num), {
        "tree__criterion": ["gini", "entropy"],
        "tree__max_depth": [4, 6, 8, 10, 12, 15],
        "tree__min_samples_leaf": [10, 50, 100, 200],
        "tree__class_weight": [None, "balanced"]}, cv=cv, scoring="average_precision", n_jobs=-1).fit(Xtr, y_tr)
    variants["B: tuned tree"] = gs_b.best_estimator_
    print("B best params:", gs_b.best_params_)
    # C: SMOTE inside the CV pipeline (synthetic rows never reach a validation fold)
    gs_c = GridSearchCV(tree_pipe(num, smote=True), {
        "tree__max_depth": [6, 8, 10, 12],
        "tree__min_samples_leaf": [50, 100, 200]}, cv=cv, scoring="average_precision", n_jobs=-1).fit(Xtr, y_tr)
    variants["C: tuned tree + SMOTE"] = gs_c.best_estimator_
    print("C best params:", gs_c.best_params_)

    rows = []
    for name, model in variants.items():
        r = cross_validate(model, Xtr, y_tr, cv=cv, scoring=["average_precision", "roc_auc"], n_jobs=-1)
        rows.append({"variant": name, "cv_pr_auc": r["test_average_precision"].mean(),
                     "cv_pr_auc_std": r["test_average_precision"].std(), "cv_roc_auc": r["test_roc_auc"].mean()})
    comp = pd.DataFrame(rows).set_index("variant")
    print(comp.round(3).to_string())

    best_name = comp["cv_pr_auc"].idxmax()
    final = variants[best_name]
    # Threshold from out-of-fold training predictions (test set untouched)
    oof = cross_val_predict(final, Xtr, y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    thr, (prec, rec, thr_arr, f1) = best_f1_threshold(y_tr, oof)

    baseline = variants["A: baseline (depth 8, balanced)"].fit(Xtr, y_tr)
    final.fit(Xtr, y_tr)
    base_s, _ = test_scores(y_te, baseline.predict_proba(Xte)[:, 1], 0.5)
    proba = final.predict_proba(Xte)[:, 1]
    fin_s, pred = test_scores(y_te, proba, thr)
    print(f"Selected: {best_name}, threshold {thr:.3f}")
    print(pd.DataFrame({"baseline A (0.5)": base_s, "final": fin_s}).round(3).to_string())

    # ---- figures
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thr_arr, prec[:-1], label="precision")
    ax.plot(thr_arr, rec[:-1], label="recall")
    ax.plot(thr_arr, f1[:-1], label="F1")
    ax.axvline(thr, color="k", ls="--", lw=1, label=f"chosen {thr:.2f}")
    ax.set(title=f"{target}: threshold trade-off (out-of-fold, training set)", xlabel="threshold")
    ax.legend()
    save(fig, f"classification_{target}_threshold.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    p_b = baseline.predict_proba(Xte)[:, 1]
    for lbl, p in [("baseline A", p_b), (f"final ({best_name[:2]})", proba)]:
        pr, rc, _ = precision_recall_curve(y_te, p)
        axes[0].plot(rc, pr, label=f"{lbl}  AP={average_precision_score(y_te, p):.3f}")
    axes[0].axhline(y_te.mean(), color="grey", ls=":", label=f"chance ({y_te.mean():.3f})")
    axes[0].set(title="Precision-recall (test)", xlabel="recall", ylabel="precision")
    axes[0].legend(fontsize=8)
    ConfusionMatrixDisplay(confusion_matrix(y_te, pred), display_labels=["no", "yes"]).plot(
        ax=axes[1], colorbar=False, cmap="Blues", values_format=",d")
    axes[1].set_title(f"Confusion matrix (test, threshold {thr:.2f})")
    fig.suptitle(target)
    save(fig, f"classification_{target}_test_results.png")

    tree = final.named_steps["tree"]
    names = final.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(tree.feature_importances_, index=names).nlargest(15).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot.barh(ax=ax, color="#3b82f6")
    ax.set(title=f"{target}: top features (impurity importance)")
    save(fig, f"classification_{target}_feature_importance.png")

    fig, ax = plt.subplots(figsize=(22, 9))
    plot_tree(tree, feature_names=names, class_names=["no", "yes"], max_depth=3, filled=True,
              rounded=True, fontsize=9, ax=ax)
    ax.set_title(f"{target}: top 3 levels of a depth-{tree.get_depth()} tree ({tree.get_n_leaves()} leaves)")
    save(fig, f"classification_{target}_tree_top_levels.png")

    return {"selected_variant": best_name, "best_params_B": gs_b.best_params_, "best_params_C": gs_c.best_params_,
            "cv_comparison": comp.round(4).reset_index().to_dict(orient="records"),
            "baseline_test": base_s, "final_test": fin_s,
            "top_features": imp.sort_values(ascending=False).round(4).to_dict()}, pred, comp


def main(track: bool):
    OUT.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")
    df = build_orders()
    print(f"Orders (delivered, reviewed): {len(df):,} | late {df.is_late.mean():.1%} "
          f"| bad review {df.bad_review.mean():.1%}")
    joint = pd.crosstab(df.is_late, df.bad_review, normalize=True)
    print("Joint distribution:\n", joint.round(3))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df.groupby("is_late").bad_review.mean().plot.bar(ax=axes[0], color=["#10b981", "#ef4444"])
    axes[0].set(title="Bad-review rate: on time vs late", xticklabels=["on time", "late"], ylabel="share 1-3 stars")
    axes[0].tick_params(axis="x", rotation=0)
    sns.heatmap(joint, annot=True, fmt=".1%", cmap="Blues", cbar=False, ax=axes[1])
    axes[1].set_title("Joint share of orders")
    save(fig, "classification_target_distribution.png")

    strat = df.is_late.astype(str) + df.bad_review.astype(str)
    tr_idx, te_idx = train_test_split(df.index, test_size=0.2, random_state=SEED, stratify=strat)
    X_tr, X_te = df.loc[tr_idx], df.loc[te_idx]
    print(f"Train {len(tr_idx):,} / test {len(te_idx):,} (no order in both)")

    results, preds = {}, {}
    for target in ["is_late", "bad_review"]:
        results[target], preds[target], _ = run_target(target, X_tr, X_te, X_tr[target], X_te[target])

    both = ((preds["is_late"] == X_te.is_late.to_numpy()) & (preds["bad_review"] == X_te.bad_review.to_numpy()))
    results["both_targets_correct_test"] = float(both.mean())
    print(f"\nShare of test orders with BOTH targets predicted correctly: {both.mean():.3f}")

    (OUT / "classification_metrics.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"Figures and metrics written to {OUT}")

    with tracking.tracking(track, "decision-tree-classification", "is_late + bad_review"):
        for t in ["is_late", "bad_review"]:
            tracking.log_params({f"{t}_variant": results[t]["selected_variant"]})
            tracking.log_metrics({f"{t}_final_{k}": v for k, v in results[t]["final_test"].items()})
            tracking.log_metrics({f"{t}_baseline_{k}": v for k, v in results[t]["baseline_test"].items()})
        tracking.log_artifacts(OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", action="store_true", help="log the run to MLflow (DagsHub if configured)")
    main(ap.parse_args().track)
