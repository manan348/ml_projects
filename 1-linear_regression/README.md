# Linear Regression Projects

This repository contains two end-to-end regression projects:

1. **Student Performance:** multiple linear regression with scikit-learn (OLS, Ridge, Lasso).
2. **Marketing Sales:** multiple linear regression with statsmodels (statistical inference, assumption diagnostics, business interpretation).

Both notebooks run top to bottom, preprocess the data without leakage, and validate on out-of-sample data.

---

## Project 1: Student Performance

**Notebook:** `linear_regression_student_performance.ipynb`
**Dataset:** `Student_Performance.csv`, 10,000 rows (9,873 after removing duplicates). Source: [Kaggle](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)

| Variable | Type | Description |
|----------|------|-------------|
| `Hours Studied` | Integer | Daily study hours |
| `Previous Scores` | Integer | Scores from previous exams |
| `Extracurricular Activities` | Binary (Yes = 1, No = 0) | Participation in extracurricular activities |
| `Sleep Hours` | Integer | Average sleep per night |
| `Sample Question Papers Practiced` | Integer | Number of practice papers completed |
| `Performance Index` | Continuous (target) | Overall performance score, 10 to 100 |

### Methodology

- **Data quality:** 127 exact duplicate rows are removed before the train/test split, so that no test row also appears in the training data.
- **Exploratory analysis:** pairplot, Pearson correlation matrix, and variance inflation factors (VIF).
- **Modeling:**
  - Ordinary least squares as the baseline.
  - Ridge and Lasso with feature scaling fit on the training data only, and the regularization strength tuned by cross-validation.
- **Validation:** an 80/20 hold-out split and 5-fold cross-validation, with scaling inside a pipeline.

### Results

| Model | Test RMSE | Test MAE | Test R² | 5-Fold CV R² |
|---|---|---|---|---|
| Linear Regression | 2.075 | 1.647 | 0.9884 | 0.9887 ± 0.0002 |
| Ridge (α = 0.001) | 2.075 | 1.647 | 0.9884 | 0.9887 ± 0.0002 |
| Lasso (α = 0.0003) | 2.075 | 1.647 | 0.9884 | 0.9887 ± 0.0002 |

### Key Findings

- `Previous Scores` (r = 0.915) and `Hours Studied` (+2.85 points per additional hour) are the dominant predictors.
- Adding `Extracurricular Activities` lowers test MSE from 4.449 to 4.306. Participation is associated with a gain of about 0.57 points.
- Regularization gives no improvement: there is no multicollinearity (all VIF ≈ 1.00), and every predictor carries signal.
- **Note:** an R² near 0.99 reflects a synthetic, almost perfectly linear dataset. Real-world educational data would not usually be this predictable.

---

## Project 2: Marketing Sales

**Notebook:** `multiple_linear_regression_marketing.ipynb`
**Dataset:** `marketing_sales_data.csv`, 572 rows (569 after removing 3 rows with missing values)

| Variable | Type | Description |
|----------|------|-------------|
| `TV` | Categorical | TV promotional budget: Low, Medium, High |
| `Radio` | Continuous | Radio promotional budget (USD millions) |
| `Social Media` | Continuous | Social media promotional budget (USD millions) |
| `Influencer` | Categorical | Influencer tier: Mega, Macro, Micro, Nano |
| `Sales` | Continuous (target) | Sales revenue (USD millions) |

### Methodology

1. **Data cleaning and splitting:** rows with missing values are dropped. The 80/20 train/test split is made before any exploration or feature selection, to prevent selection leakage.
2. **Exploratory analysis and feature selection (training data only):** a pairplot, mean sales by category, and a comparison of candidate models by adjusted R², AIC, and BIC.
3. **Assumption diagnostics:**
   - Linearity and independence.
   - Normality: Shapiro-Wilk, Jarque-Bera, and Omnibus tests.
   - Homoscedasticity: Breusch-Pagan test.
   - Multicollinearity: VIF.
   - Autocorrelation: Durbin-Watson.
   - A confounding check between the TV and Radio budgets.
4. **Inference:** the selected model is refit on the full dataset with HC3 heteroscedasticity-robust standard errors.
5. **Validation:** 5-fold cross-validation on the training data, then a single evaluation on the held-out test set.
6. **Scenario analysis:** sales forecasts with 95% prediction intervals, flagged when a scenario falls outside the observed data range.

### Results

| Metric | Value |
|---|---|
| Selected model | `Sales ~ C(TV) + Radio` |
| Train R² / Adjusted R² | 0.901 / 0.900 |
| 5-fold CV R² (training data) | 0.900 ± 0.009 |
| **Hold-out test R²** | **0.915** |
| Test RMSE / MAE | 28.3 / 24.0 (USD millions) |

### Coefficient Estimates

Full dataset, with HC3 robust 95% confidence intervals:

| Term | Estimate | 95% Confidence Interval |
|---|---|---|
| Intercept (High TV) | 217.64 | [206.51, 228.76] |
| TV = Low | -152.09 | [-160.79, -143.39] |
| TV = Medium | -73.48 | [-79.96, -67.00] |
| Radio (per USD 1M) | 2.89 | [2.52, 3.25] |

### Key Findings

- **TV budget level is the dominant driver of sales.** Holding Radio spend constant, campaigns with a Low TV budget generate about USD 152M less in sales than campaigns with a High TV budget.
- **Radio:** each additional USD 1M of Radio spend is associated with about USD 2.89M of additional sales.
- **Social Media and Influencer** add no explanatory power once TV and Radio are in the model.
- **Assumptions:** normality is mildly violated (light-tailed residuals) and constant variance is borderline, so HC3 robust standard errors are used for inference.
- **Note:** TV and Radio budgets are strongly correlated in this data; High TV campaigns almost always also have high Radio spend. The coefficients therefore describe associations, not causal returns on investment.

### Scenario Forecasts

| Scenario | Predicted Sales | 95% Prediction Interval |
|----------|----------------|-------------------------|
| Low TV, USD 5M Radio | USD 80.0M | [24.8, 135.1] |
| Medium TV, USD 15M Radio | USD 187.4M | [132.3, 242.6] |
| High TV, USD 25M Radio | USD 289.8M | [234.6, 345.0] |
| High TV, USD 35M Radio | USD 318.7M | [263.4, 373.9] |

---

## Requirements

- Python 3.8 or later
- pandas, numpy, scipy
- statsmodels, scikit-learn
- matplotlib, seaborn

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Project Structure

```
1-linear-regression/
├── Student_Performance.csv                     # Student performance dataset (10,000 rows)
├── marketing_sales_data.csv                    # Marketing promotion and sales dataset (572 rows)
├── linear_regression_student_performance.ipynb # Regression with scikit-learn (OLS, Ridge, Lasso)
├── multiple_linear_regression_marketing.ipynb  # Regression and inference with statsmodels
├── run_analysis.py                             # Standalone marketing analysis script
├── improve_model.py                            # Marketing model improvement experiments
├── requirements.txt                            # Project dependencies
└── LICENSE                                     # MIT License
```

## Usage

1. Install the dependencies listed in `requirements.txt`.
2. Make sure both CSV files are in the same directory as the notebooks.
3. Open either notebook in Jupyter Lab or Jupyter Notebook and run all cells.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
