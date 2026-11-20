# Logistic Regression Projects

This repository contains five classification projects built with logistic regression:

| Notebook | Task | Dataset |
|---|---|---|
| `simple_logistic_regression.ipynb` | Binary classification | Social Network Ads |
| `binary_regression.ipynb` | Data cleaning + binary classification | Messy healthcare records |
| `multinomial_regression.ipynb` | Multiclass (3-class) classification | Students Performance |
| `multivariate-regression.ipynb` | Multi-output (3 targets × 3 classes) classification | UCI Air Quality |
| `Perform logistic regression.ipynb` | Binary classification | Invistico Airline (not included) |

All preprocessing (imputation, encoding, scaling) is fitted inside scikit-learn pipelines on training data only, so the reported results are free of leakage.

---

## Project 1: Social Network Ads Purchase Prediction

**Notebook:** `simple_logistic_regression.ipynb`
**Dataset:** `social_network_ads.csv`, 400 users

Predicts whether a user buys a product after seeing a social network advertisement, using gender, age, and estimated salary.

### Methodology
- Stratified 80/20 train/test split. The classes are 64% non-buyers and 36% buyers.
- `StandardScaler` and `LogisticRegression` inside a pipeline, so the scaler is refit on every cross-validation fold.
- Grid search over regularization strength `C` and `class_weight`, using 10-fold stratified cross-validation and selecting the model by F1 score.
- Coefficients reported as odds ratios per standard deviation.

### Results

| Metric | 10-fold CV (train) | Hold-out test |
|---|---|---|
| Accuracy | 0.844 | 0.800 |
| ROC AUC | 0.930 | 0.906 |
| Recall (Purchased) | 0.816 | 0.793 |

### Key Findings
- Age is the strongest predictor. Each additional 10.7 years (one standard deviation) multiplies the odds of purchase by about 4.6.
- Estimated salary is the second predictor (about 2.4× the odds per standard deviation). Gender has no meaningful effect.
- Balanced class weights raise recall for buyers from about 0.69 to 0.82, at the cost of a few extra false positives.

---

## Project 2: Hypertension Prediction from Messy Healthcare Records

**Notebook:** `binary_regression.ipynb`
**Dataset:** `healthcare_messy_data.csv`, 1,000 records (794 with a recorded condition)

A data-cleaning exercise followed by a binary logistic regression that predicts whether a patient's condition is hypertension.

### Data Cleaning
- Removed the patient name, email, and phone columns (identifiers with no predictive meaning).
- Converted text ages to numbers (`'forty'` → 40), and split blood pressure text (`'140/90'`) into systolic and diastolic values.
- Parsed five different date formats into a visit year.
- Kept `'Other'` gender and `'NONE'` medication as real categories rather than overwriting them.
- Dropped rows with a missing target instead of filling them, which would have invented labels.

### Results

| Model | CV ROC AUC | Test ROC AUC |
|---|---|---|
| Random baseline (DummyClassifier) | 0.511 | — |
| Logistic Regression | 0.493 | 0.455 |

### Key Findings
- The model performs at chance level. This is the correct outcome: chi-square tests show no feature is associated with the target (all p-values between 0.32 and 0.86).
- The dataset is synthetic, with values generated independently of each other, and was designed for practicing data cleaning rather than prediction.
- The comparison with a dummy baseline shows why accuracy alone is misleading here: about 79% of patients are not hypertensive.

---

## Project 3: Student Math Performance Category

**Notebook:** `multinomial_regression.ipynb`
**Dataset:** `StudentsPerformance.csv`, 1,000 students

Classifies a student's math score as Low (0–60), Medium (61–80), or High (81–100), using demographics, lunch type, test preparation, and reading and writing scores.

### Methodology
- One-hot encoding for nominal features (gender, ethnicity group, parental education, lunch, test preparation), and standardization for reading and writing scores.
- Multinomial (softmax) logistic regression.
- Stratified train/test split and 5-fold stratified cross-validation.
- Comparison against a demographics-only model and a majority-class baseline.

### Results

| Metric | 5-fold CV (train) | Hold-out test |
|---|---|---|
| Accuracy | 0.823 | 0.810 |
| Macro F1 | 0.820 | 0.813 |
| ROC AUC (one-vs-rest) | 0.942 | 0.953 |

### Key Findings
- Recall is balanced across the three classes (0.77–0.85). Errors occur only between neighboring categories; Low and High are never confused.
- Reading and writing scores (correlation with math ≈ 0.80) provide almost all of the predictive power. Demographics alone reach about 50% accuracy, close to the 48.5% majority baseline.

---

## Project 4: Air Quality Pollutant Level Classification

**Notebook:** `multivariate-regression.ipynb`
**Dataset:** `AirQualityUCI.csv`, hourly readings from March 2004 to April 2005. Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/360/air+quality)

Classifies three pollutants at once (CO, benzene, and NOx) into Low / Medium / High levels, using only low-cost metal-oxide sensor readings and weather data. Each target has its own logistic regression, wrapped in `MultiOutputClassifier`.

### Methodology
- Parsed timestamps, removed empty padding rows, and treated `-200` as missing.
- Dropped `NMHC(GT)` (90% missing). Excluded `NO2(GT)` from the features because it is a component of the NOx target.
- Dropped rows with a missing target instead of imputing them.
- **Chronological split:** the first 80% of the timeline is used for training and the final 20% for testing, so the model is evaluated on future data.
- Class cut points (tertiles) are computed on the training period only.

### Results (future test period)

| Target | Accuracy | Macro F1 | ROC AUC (OvR) |
|---|---|---|---|
| CO | 0.842 | 0.844 | 0.951 |
| Benzene (C6H6) | 0.982 | 0.981 | 0.999 |
| NOx | 0.839 | 0.821 | 0.955 |

### Key Findings
- The low-cost sensors classify pollutant levels with about 84% accuracy for CO and NOx on unseen future data, despite a seasonal shift in the class mix.
- Benzene is near-perfect because the `PT08.S2(NMHC)` sensor tracks benzene almost exactly (correlation 0.98).
- A random split would inflate accuracy by only 1–2 points here. The chronological split is still the correct choice for time-series data.

---

## Project 5: Airline Customer Satisfaction

**Notebook:** `Perform logistic regression.ipynb`
**Dataset:** `Invistico_Airline.csv` (**not included** in this repository)

A binomial logistic regression that predicts airline customer satisfaction from the in-flight entertainment rating, evaluated with a confusion matrix. To run it, download the Invistico Airline customer satisfaction dataset (available on Kaggle) and place `Invistico_Airline.csv` in this folder.

---

## Requirements

- Python 3.8 or later
- pandas, numpy, scipy
- scikit-learn
- matplotlib, seaborn

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Project Structure

```
2-logistic-regression/
├── social_network_ads.csv                # Social Network Ads dataset (400 rows)
├── healthcare_messy_data.csv             # Messy healthcare records (1,000 rows)
├── StudentsPerformance.csv               # Students Performance dataset (1,000 rows)
├── AirQualityUCI.csv                     # UCI Air Quality dataset (hourly, 2004-2005)
├── simple_logistic_regression.ipynb      # Binary: purchase prediction
├── binary_regression.ipynb               # Data cleaning + binary: hypertension
├── multinomial_regression.ipynb          # Multiclass: math performance category
├── multivariate-regression.ipynb         # Multi-output: pollutant levels
├── Perform logistic regression.ipynb     # Binary: airline satisfaction (dataset not included)
├── requirements.txt                      # Project dependencies
└── README.md
```

## Usage

1. Install the dependencies listed in `requirements.txt`.
2. Make sure the CSV files are in the same directory as the notebooks.
3. Open any notebook in Jupyter Lab or Jupyter Notebook and run all cells.
