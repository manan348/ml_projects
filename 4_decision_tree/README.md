# Decision Tree Projects

Two end-to-end decision tree projects, each a standalone, leakage-free Python script:

1. **Regression:** California housing prices (`decision_tree_regression.py`).
2. **Classification:** late deliveries and bad reviews on the Olist e-commerce data (`decision_tree_classification.py`).

Both scripts split the data first, fit every preprocessing step inside a scikit-learn pipeline (so only on training folds), compare several tree variants with cross-validation on the training set, and use the test set exactly once. Figures and a metrics JSON go to `outputs/`.

The original notebooks (`decision_tree_regression.ipynb`, `decision_tree_classification.ipynb`) are kept for reference. The scripts replace them. The section [What was wrong in the notebooks](#what-was-wrong-in-the-notebooks) explains why.

---

## Project 1: California Housing (Regression)

**Script:** `decision_tree_regression.py`
**Dataset:** `housing.csv`, 20,640 districts, target `median_house_value`. 207 rows are missing `total_bedrooms`, and the target is **capped at $500,001** for 4.8 % of rows, so the model cannot predict above that cap.

### Methodology

1. **Split first:** 80/20 train/test, stratified on income bands.
2. **Pipeline:** per-household ratios (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`), median imputation, then one-hot `ocean_proximity`, then the tree. There is no scaling, Box-Cox or winsorizing: a tree's splits do not change under monotonic transforms, so these steps add nothing.
3. **Variants**, each scored with 5-fold CV on the training set:

   | Variant | Description |
   |---|---|
   | A | the notebook's settings (`max_depth=8`, `min_samples_leaf=20`, `min_samples_split=50`) |
   | B | grid search over `max_depth`, `min_samples_leaf`, `max_features` |
   | C | as B, but trained on `log1p(price)` (`TransformedTargetRegressor`), still scored in dollars |

4. **Final model:** the variant with the lowest CV RMSE, refit on the full training set, scored once on the test set.

An unpruned tree keeps lowering training error, while validation error bottoms out around depth 8 to 10. That gap is why `min_samples_leaf` and `max_depth` matter so much for single trees.

### Results

| Variant | CV RMSE | CV R² |
|---|---|---|
| A: baseline | $61,444 ± 1,346 | 0.718 |
| B: tuned (depth 20, leaf 20) | $59,143 ± 1,052 | 0.739 |
| **C: tuned + log target (depth 15, leaf 20)** | **$58,888 ± 2,064** | **0.741** |

| Test set (4,128 districts) | Baseline A | Final (C) |
|---|---|---|
| R² | 0.730 | **0.751** |
| RMSE | $59,362 | **$57,008** |
| MAE | $41,717 | **$37,674** |
| Train R² | 0.764 | 0.821 |

### Key findings

- Tuning gains about $2,500 RMSE over the baseline. B and C are within one standard deviation of each other, so the log target is a tie rather than a clear win.
- **`median_income` dominates:** shuffling it raises RMSE by about $59k. `population` and `households` rank high because they feed the per-household ratio features.
- The horizontal band at $500k in `outputs/regression_actual_vs_predicted.png` is the capped target: the model cannot learn values above the cap.
- A single tree tops out at about R² 0.75. Averaging many trees (see `5_random_forest`, `6_gradient_boosting`) is the usual next step.

---

## Project 2: Olist Deliveries and Reviews (Classification)

**Script:** `decision_tree_classification.py`
**Dataset:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce): 9 CSVs (orders, items, payments, reviews, products, customers, sellers, geolocation, category translation), about 120 MB in total and versioned with DVC.

### The two targets

| Target | Definition | When it is predicted | Allowed inputs |
|---|---|---|---|
| `is_late` | delivered after the promised date (6.7 % of orders) | at purchase | purchase-time features only |
| `bad_review` | review score of 1 to 3 (21.1 %) | after delivery | purchase-time features + delivery outcome (days late, delivery time) |

The data holds 95,810 delivered and reviewed orders, at **one row per order**. Features: promised delivery days, seller shipping deadline, approval delay, purchase month/weekday/hour, item count, sellers, price, freight, weight, volume, payment type and installments, customer and seller state, and the **customer-to-seller distance** (from zip-code geolocation).

Late orders get a bad review 73 % of the time, against 17 % for on-time orders.

### Methodology

1. 80/20 stratified split on the joint label (no order appears in both).
2. Per target, three variants with 5-fold stratified CV, scored by **PR-AUC** (the right metric for a rare positive class):

   | Variant | Description |
   |---|---|
   | A | the notebook's settings (`max_depth=8`, `class_weight="balanced"`), threshold 0.5 |
   | B | grid search over criterion (gini/entropy), depth, `min_samples_leaf`, `class_weight` |
   | C | SMOTE **inside** the CV pipeline, then a tuned tree |

3. The decision threshold maximises F1 on **out-of-fold training predictions**, then the test set is scored once.

### Results

| CV PR-AUC | is_late | bad_review |
|---|---|---|
| A: baseline | 0.190 | 0.475 |
| **B: tuned tree** | **0.219** | **0.492** |
| C: tuned + SMOTE | 0.131 | 0.469 |

| Test set (19,162 orders) | is_late: A (0.5) | is_late: **B** (0.17) | bad_review: A (0.5) | bad_review: **B** (0.27) |
|---|---|---|---|---|
| PR-AUC (chance) | 0.169 | **0.205** (0.067) | 0.478 | **0.493** (0.211) |
| ROC-AUC | 0.708 | **0.731** | 0.688 | **0.692** |
| Precision | 0.143 | **0.225** | 0.390 | **0.458** |
| Recall | **0.633** | 0.382 | **0.504** | 0.426 |
| F1 | 0.233 | **0.284** | 0.440 | **0.441** |
| Orders flagged | 5,657 | 2,165 | 5,214 | 3,758 |

### Key findings

- **These scores are lower than the notebook's, and that is correct.** The notebook's numbers came from leaked information (see below). Predicting lateness at purchase time is genuinely hard: the tree gets 3× the chance precision (PR-AUC 0.205 vs 0.067).
- **is_late drivers:** purchase month (the Nov 2017 and Feb–Mar 2018 delivery crunches), how many days were promised, Rio de Janeiro customers, and customer–seller distance.
- **bad_review drivers:** `days_late` dominates, followed by multi-item and multi-seller orders (split shipments).
- **SMOTE hurt both targets** once it was applied correctly inside CV. `class_weight` or a tuned threshold is enough for a tree.
- Gini vs entropy made no measurable difference: both searches picked gini, and the gap was within CV noise.
- Both targets were predicted correctly for 67.8 % of test orders.

---

## What was wrong in the notebooks

### `decision_tree_classification.ipynb`

| # | Problem | Effect |
|---|---|---|
| 1 | **Wrong column:** the `bad_review` reports use `y_pred[:, 0]` (the is_late predictions) | the reported bad_review F1 of 0.63 was a different model's output |
| 2 | **Target leakage:** `review_score` is a feature, and `bad_review = review_score <= 3` | with the correct column, bad_review F1 = **1.00**, a trivial model |
| 3 | **Post-outcome features** for is_late: delivered-date and review-date year/month/day/hour, plus `review_score` | `review_score` was the top is_late feature (importance 0.46) |
| 4 | Merging items × payments gives one row **per item per payment**, then `head(15000)` | 22.6 % of test rows belonged to orders also in the training set; only the first 15k rows were used, not a random sample |
| 5 | SMOTE applied **before** cross-validation | synthetic copies leak into validation folds, so CV picked an unpruned tree (`max_depth=None`, `min_samples_leaf=1`) |
| 6 | SMOTE over label-encoded categories and date parts | creates impossible values (category 37.4, hour 13.6) |
| 7 | Gini vs entropy compared on the **test set** | the test set is used for model selection |
| 8 | The last cell was never run; `!pip install plotly` sits inside the code; `roc_auc_score` and `cross_val_score` are imported but unused | the notebook is not reproducible top to bottom |

### `decision_tree_regression.ipynb`

| # | Problem | Effect |
|---|---|---|
| 1 | Imputation, winsorizing, Box-Cox and KMeans fitted on the **full data before the split** | mild leakage. These transforms are also pointless for trees (splits do not change under monotonic transforms) |
| 2 | **Swapped labels:** `train_mse = mean_squared_error(y_test, …)` and the reverse | "Train RMSE 60,228 / Test RMSE 55,268" is really Test 60,228 / Train 55,268 |
| 3 | First metric cell prints "Train MSE" as an RMSE and "Test MSE" as an MSE | the numbers cannot be compared |
| 4 | "GridSearchCV" with **one value per parameter** | no search actually happens |
| 5 | Three models compared on the **test set** | the reported test score is optimistic |
| 6 | Winsorizing the one-hot `ocean_proximity_ISLAND` column | turns it into a constant column (skew 0) |
| 7 | `distance_to_coast = max(longitude) - longitude` | just a shifted copy of longitude, not a distance |
| 8 | `plt.Figure(figsize=…)` instead of `plt.figure` | the figure size is silently ignored |
| 9 | The `root_mean_squared_error` function redefined twice, shadowing the sklearn import | confusing code |
| 10 | The $500,001 target cap is never mentioned | it explains the band of large errors at the top |

---

## Requirements

```bash
pip install -r requirements.txt            # numpy, pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn
pip install -r requirements-dagshub.txt    # optional: dvc, mlflow, dagshub
```

## Project structure

```
4_decision_tree/
├── decision_tree_regression.py       # California housing, 3 tree variants
├── decision_tree_classification.py   # Olist is_late + bad_review, 3 variants each
├── tracking.py                       # optional MLflow / DagsHub tracking (--track)
├── housing.csv                       # 1.4 MB, kept in Git
├── olist_*.csv.dvc, product_category_name_translation.csv.dvc   # DVC pointers (data on DagsHub)
├── decision_tree_*.ipynb             # original notebooks (superseded, see above)
├── outputs/                          # figures + regression_metrics.json + classification_metrics.json
├── requirements.txt / requirements-dagshub.txt
└── README.md
```

## Usage

```bash
python decision_tree_regression.py        # about 2 minutes
python decision_tree_classification.py    # about 40 minutes (6 grid searches on 77k orders)
```

All seeds are fixed to 42.

## DagsHub: data versioning and experiment tracking

### Get the Olist data

```bash
pip install dvc
dvc pull          # downloads the Olist CSVs from DagsHub storage
```

Alternatively, download them from Kaggle (link above) into this folder.

### Track experiments (MLflow)

Without configuration, `--track` logs to a local `mlflow.db`. To log to DagsHub (PowerShell; create a token under DagsHub *Settings → Tokens*, never commit it):

```powershell
$env:DAGSHUB_REPO_OWNER = "<your-dagshub-username>"
$env:DAGSHUB_REPO_NAME  = "ml_projects"
$env:DAGSHUB_USER_TOKEN = "<your-dagshub-token>"

python decision_tree_regression.py --track
python decision_tree_classification.py --track
```

Each run logs the selected variant, test metrics for the baseline and final model, and every file in `outputs/`.
