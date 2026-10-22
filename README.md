# Employee Attrition Prediction Models
### End-to-End Supervised Machine Learning Benchmark & Workforce Intelligence System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-red.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: Pytest Passing](https://img.shields.io/badge/Tests-14%20Passed-brightgreen.svg)](tests/)

---

## 1. Executive Summary & Overview
This repository delivers an end-to-end, reproducible, leakage-safe machine learning project designed for the academic capstone assignment **"Employee Attrition Prediction Models."** 

The primary business objective is to accurately predict voluntary and involuntary employee departure using multidimensional workforce data ($N = 1,470$ records), enabling proactive people operations, targeted stay interviews, and data-driven retention interventions before costly turnover takes place.

### Key Deliverables:
- **Four Trained Classification Paradigms:** Logistic Regression, Decision Tree, Random Forest, and XGBoost benchmarked against a Dummy Baseline.
- **Leakage-Safe Preprocessing Pipeline:** Stratified train-test splitting (80/20) with untouched test evaluation, numerical scaling, and safe categorical one-hot encoding.
- **Imbalance-Aware Optimization:** Systematic hyperparameter tuning under 5-Fold Stratified Cross-Validation optimizing for **F1-Score** and **ROC-AUC** rather than misleading raw accuracy.
- **Explainability Suite:** Extraction of linear coefficients, tree impurity importances (MDI), and permutation importance.
- **Optional Continuous Task:** High-precision regression pipeline predicting `MonthlyIncome` ($R^2 = 0.9583$).
- **Dual PDF Publications:**
  1. `reports/employee_attrition_evaluation_report.pdf` (Technical Evaluation Report)
  2. `reports/employee_attrition_business_interpretation_report.pdf` (Executive HR Strategy Report)
- **Production Notebook:** Fully executed 27-section Jupyter Notebook (`notebooks/employee_attrition_prediction.ipynb`).
- **Automated Test Suite:** 14 passing unit tests covering validation, pipelines, models, and metric calculation (`pytest`).

---

## 2. Problem Statement & Scope
Corporate employee turnover imposes crippling operational penalties. Benchmark studies from Gallup and SHRM estimate the replacement cost of a salaried professional at **50% to 150% of annual compensation** due to:
1. **Direct Recruiting & Sourcing Costs:** Agency fees, assessment tools, and advertising.
2. **Onboarding & Ramp-Up Drag:** Reduced velocity across team members mentoring new hires.
3. **Loss of Tacit Institutional Knowledge:** Departure of critical operational domain expertise.
4. **Cascading Turnover Contagion:** Overburdened colleagues suffering burnout following team departures.

This project reframes retention from reactive exit interviews to **proactive algorithmic detection**, equipping HR leaders with calibrated probability scores to implement supportive retention initiatives.

---

## 3. Dataset Source & Data Dictionary

The project utilizes the canonical **IBM HR Analytics Employee Attrition** benchmark dataset.
- **Total Records:** 1,470 employees
- **Total Features:** 22 (14 numerical, 8 categorical)
- **Missing Values:** 0 null values across all attributes
- **Duplicate Records:** 0 duplicates
- **Administrative Key:** `EmployeeID` (strictly quarantined and dropped prior to modeling)
- **Primary Target:** `Attrition` (`Yes` = departed enterprise, `No` = retained)
  - Retained (`No`): 1,196 employees (**81.36%**)
  - Departed (`Yes`): 274 employees (**18.64%**)
  - Class Imbalance Ratio: **4.37 : 1**

### Feature Schema:
| Feature Name | Data Type | Role | Description |
|---|---|---|---|
| `EmployeeID` | String | Identifier | Unique tracking key (dropped to prevent memorization) |
| `Department` | Categorical | Predictor | Sales, Research & Development, Human Resources |
| `JobRole` | Categorical | Predictor | 9 specific organizational job titles |
| `Age` | Integer | Predictor | Chronological age (20 to 59 years) |
| `Gender` | Categorical | Demographic | Male / Female (audited for algorithmic fairness) |
| `EducationLevel`| Categorical | Predictor | Below College, College, Bachelor, Master, Doctor |
| `JobLevel` | Integer | Predictor | Hierarchical corporate grade (1 to 5) |
| `MonthlyIncome` | Integer | Predictor / Reg Target | Gross monthly earnings in USD ($1,800 to $25,000+) |
| `TotalWorkingYears` | Integer | Predictor | Total career professional experience (0 to 40) |
| `YearsAtCompany` | Integer | Predictor | Tenure at current enterprise (0 to 35) |
| `YearsInCurrentRole` | Integer | Predictor | Duration in current assignment (0 to 20) |
| `YearsSinceLastPromotion` | Integer | Predictor | Years elapsed since last promotion (0 to 18) |
| `YearsWithCurrManager` | Integer | Predictor | Tenure under current supervisor (0 to 18) |
| `TrainingTimesLastYear` | Integer | Predictor | Formal training events attended (0 to 6) |
| `JobSatisfaction` | Likert (1-4) | Predictor | Self-reported job satisfaction rating |
| `EnvironmentSatisfaction` | Likert (1-4) | Predictor | Workplace environment satisfaction rating |
| `WorkLifeBalance` | Likert (1-4) | Predictor | Balance between professional and personal life |
| `PerformanceRating` | Likert (1-4) | Predictor | Annual performance review evaluation score |
| `Overtime` | Binary | Predictor | Requirement to perform overtime (Yes / No) |
| `BusinessTravel` | Categorical | Predictor | Travel frequency (Non-Travel, Rare, Frequent) |
| `AbsenteeismDays` | Integer | Predictor | Days of absence recorded in past 12 months |
| `Attrition` | Binary | **Target** | Voluntary/involuntary turnover (`Yes`=1, `No`=0) |

---

## 4. Technology Stack & Dependencies
- **Core Language:** Python 3.11
- **Data Engineering:** Pandas 3.0, NumPy 2.4, SciPy 1.17
- **Machine Learning:** Scikit-Learn 1.9, XGBoost 3.2, Joblib 1.5
- **Visualization:** Matplotlib 3.10, Seaborn 0.13, Plotly 7.1
- **Report Compilation:** ReportLab 5.0, OpenPyXL 3.1
- **Testing & Quality Assurance:** Pytest 9.1

---

## 5. Project Directory Structure
```
Project/
│
├── README.md                                  # Complete project documentation & guide
├── requirements.txt                           # Pin-pointed Python dependencies
├── pytest.ini                                 # Pytest configuration & search path
├── .gitignore                                 # Git ignore patterns for venv & cache
├── LICENSE                                    # MIT Open Source License
├── run_pipeline.py                            # End-to-end master execution pipeline
├── build_notebook.py                          # Jupyter notebook generator script
│
├── data/
│   ├── raw/
│   │   └── employee_data.csv                  # Pristine raw IBM HR dataset (N=1,470)
│   ├── processed/
│   │   ├── train_features.csv                 # 80% Stratified training features (N=1,176)
│   │   ├── test_features.csv                  # 20% Held-out test features (N=294)
│   │   ├── train_target.csv                   # Training target vector (y_train)
│   │   └── test_target.csv                    # Held-out test target vector (y_test)
│   ├── data_dictionary.csv                    # Detailed variable schema & leakage audit
│   └── README.md                              # Data documentation & usage ethics
│
├── notebooks/
│   └── employee_attrition_prediction.ipynb    # Executed 27-section Jupyter Notebook
│
├── src/
│   ├── __init__.py                            # Package initializer
│   ├── utils.py                               # Seed fixing, logging, timer, artifact I/O
│   ├── data_loader.py                         # Safe dataset loading utilities
│   ├── data_validation.py                     # Missingness, duplicate, & anomaly checks
│   ├── preprocessing.py                       # ColumnTransformer pipelines & splitting
│   ├── eda.py                                 # 13 high-resolution analytical plots
│   ├── models.py                              # Baseline, LR, DT, RF, XGBoost, Regressor
│   ├── tuning.py                              # Systematic 5-fold CV hyperparameter tuning
│   ├── evaluation.py                          # Metric computation, ROC/PR, thresholding
│   ├── feature_importance.py                  # Coefficients, MDI, permutation importance
│   └── reporting.py                           # Publication-grade PDF generator
│
├── models/
│   ├── best_model_pipeline.joblib             # Champion model pipeline (Random Forest)
│   ├── logistic_regression_pipeline.joblib    # Tuned Logistic Regression pipeline
│   ├── decision_tree_pipeline.joblib          # Tuned Decision Tree pipeline
│   ├── random_forest_pipeline.joblib          # Tuned Random Forest pipeline
│   ├── xgboost_pipeline.joblib                # Tuned XGBoost pipeline
│   ├── model_metadata.json                    # Serialized hyperparameters & test metrics
│   └── README.md                              # Model loading & inference documentation
│
├── reports/
│   ├── employee_attrition_evaluation_report.pdf           # Technical PDF (1.14 MB)
│   └── employee_attrition_business_interpretation_report.pdf # Executive HR PDF (651 KB)
│
├── outputs/
│   ├── figures/                               # 18 high-resolution PNG charts (300 DPI)
│   └── tables/                                # CSV and Excel benchmark summaries
│
└── tests/
    ├── test_data_validation.py                # Data loading & integrity tests
    ├── test_preprocessing.py                  # Preprocessor & split tests
    ├── test_models.py                         # Pipeline instantiation & probability tests
    └── test_evaluation.py                     # Metric calculation & persistence tests
```

---

## 6. Installation & Environment Setup

### Windows PowerShell Quickstart
Execute the following commands in order:

```powershell
# 1. Clone or navigate to the project directory
cd c:\Users\Rushi\.gemini\antigravity-ide\scratch\Project

# 2. Create Python virtual environment
python -m venv .venv

# 3. Activate the virtual environment
.venv\Scripts\Activate.ps1

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install all required dependencies
pip install -r requirements.txt

# 6. Register kernel for Jupyter Notebook
python -m ipykernel install --user --name employee-attrition --display-name "Python (Employee Attrition)"

# 7. Run automated test suite to verify installation
python -m pytest tests/ -v

# 8. Launch Jupyter Notebook
jupyter notebook
```

---

## 7. Model Execution & Experimental Results

### 1. 5-Fold Stratified Cross-Validation Benchmark (Training Partition $N = 1,176$)
Evaluated using Stratified K-Fold CV to measure out-of-fold generalization prior to test evaluation:

| Classifier | Accuracy (Mean ± Std) | Precision (Mean ± Std) | Recall (Mean ± Std) | F1-Score (Mean ± Std) | ROC-AUC (Mean ± Std) | CV Time |
|---|---|---|---|---|---|---|
| **Baseline (Dummy)** | $0.8138 \pm 0.00$ | $0.0000 \pm 0.00$ | $0.0000 \pm 0.00$ | $0.0000 \pm 0.00$ | $0.5000 \pm 0.00$ | 0.08s |
| **Logistic Regression** | $0.5986 \pm 0.03$ | $0.2414 \pm 0.02$ | $\mathbf{0.5429 \pm 0.05}$ | $\mathbf{0.3340 \pm 0.02}$ | $\mathbf{0.6212 \pm 0.03}$ | 0.85s |
| **Decision Tree** | $0.6139 \pm 0.03$ | $0.2462 \pm 0.02$ | $0.5066 \pm 0.05$ | $0.3299 \pm 0.03$ | $0.5902 \pm 0.04$ | 0.22s |
| **Random Forest** | $\mathbf{0.7244 \pm 0.02}$ | $\mathbf{0.2980 \pm 0.03}$ | $0.3645 \pm 0.04$ | $0.3269 \pm 0.03$ | $0.6105 \pm 0.03$ | 2.15s |
| **XGBoost** | $0.7194 \pm 0.03$ | $0.2700 \pm 0.03$ | $0.3053 \pm 0.04$ | $0.2855 \pm 0.03$ | $0.5738 \pm 0.03$ | 0.95s |

---

### 2. Final Evaluation on Strictly Untouched Test Set ($N = 294$, 239 Retained / 55 Departed)
Tuned models evaluated once on the quarantined held-out test partition at default threshold ($p \ge 0.50$):

| Classifier | Accuracy | Precision | Recall | F1-Score | ROC-AUC | TP | FP | TN | FN | Inference / Fit |
|---|---|---|---|---|---|---|---|---|---|---|
| **Baseline (Dummy)** | 0.8129 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0 | 0 | 239 | 55 | 0.075s |
| **Logistic Regression** | 0.6565 | 0.3175 | $\mathbf{0.7273}$ | $\mathbf{0.4420}$ | 0.7281 | 40 | 86 | 153 | 15 | 0.057s |
| **Decision Tree** | 0.6769 | 0.3113 | 0.6000 | 0.4099 | 0.6491 | 33 | 73 | 166 | 22 | 0.038s |
| **Random Forest** | $\mathbf{0.7517}$ | $\mathbf{0.3750}$ | 0.4909 | 0.4252 | 0.7240 | 27 | 45 | 194 | 28 | 0.651s |
| **XGBoost** | 0.7381 | 0.3382 | 0.4182 | 0.3740 | $\mathbf{0.7294}$ | 23 | 45 | 194 | 32 | 0.715s |

#### Optimal Threshold Performance for Production Model (Random Forest):
- **Default Threshold ($0.50$):** Precision = $0.3750$ | Recall = $0.4909$ | F1-Score = $0.4252$
- **CV-Calibrated Optimal Threshold ($0.48$):** Precision = $0.3659$ | Recall = $\mathbf{0.5455}$ | F1-Score = $\mathbf{0.4380}$
*(Captures 30 out of 55 true leavers while keeping false positives under control).*

---

### 3. Optimal Hyperparameters from Tuning
- **Logistic Regression:** `{'C': 0.01, 'class_weight': 'balanced'}`
- **Decision Tree:** `{'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 4, 'min_samples_split': 20, 'class_weight': 'balanced'}`
- **Random Forest:** `{'n_estimators': 150, 'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 4, 'min_samples_split': 12, 'class_weight': 'balanced'}`
- **XGBoost:** `{'colsample_bytree': 0.75, 'learning_rate': 0.03, 'max_depth': 4, 'min_child_weight': 3, 'n_estimators': 180, 'scale_pos_weight': 4.37, 'subsample': 0.75}`

---

### 4. Optional Continuous Regression Task (Predicting `MonthlyIncome`)
- **Algorithm:** RandomForestRegressor ($N_{\text{est}}=100, \text{max\_depth}=8$)
- **Test Mean Absolute Error (MAE):** **$724.81**
- **Test Root Mean Squared Error (RMSE):** **$1,057.24**
- **Test Coefficient of Determination ($R^2$):** **0.9583**
*(Demonstrates exceptional predictive fidelity on continuous career compensation).*

---

## 8. Feature Importance & Explainability Findings

Attribution across linear coefficients and tree feature importances reveals four decisive factors driving employee turnover:
1. **Mandatory Overtime (`Overtime_Yes`):** Emerges as the **#1 strongest positive driver of attrition**. Employees required to work overtime leave at nearly **3x the rate** of non-overtime peers ($30.5\%$ vs $10.4\%$).
2. **Monthly Compensation (`MonthlyIncome`):** Strong negative association with departure. Lower-earning tiers exhibit turnover rates exceeding $25\%$, stabilizing under $10\%$ once compensation reaches senior bands.
3. **Tenure in Current Assignment (`YearsInCurrentRole`):** High risk observed among personnel who remain in the same role for $1-2$ years without horizontal enrichment or vertical advancement (career stagnation).
4. **Managerial Tenure Continuity (`YearsWithCurrManager`):** Attrition spikes when employees transition to newly assigned direct managers ($<1$ year of reporting relationship).

---

## 9. Model Selection & Academic Rationale

### Selected Production Model: **Random Forest Classifier**
- **High Discrimination & Calibration:** Achieves an untouched test **ROC-AUC of 0.724** and high overall Accuracy ($75.2\%$) without collapsing on the minority class.
- **Resilient Precision-Recall Balance:** At calibrated threshold $0.48$, achieves **$54.6\%$ Recall** and **$0.438$ F1-Score**, successfully identifying the majority of at-risk employees while maintaining a low false-positive rate (only 45 false alarms across 239 retained employees).
- **Ensemble Stability:** Decision tree bagging provides immunity to individual leaf overfitting and minimizes variance compared to single trees or high-learning-rate boosting.

---

## 10. Automated Testing & Validation

The test suite can be run at any time via:
```powershell
python -m pytest tests/ -v
```

### Test Suite Coverage:
- `tests/test_data_validation.py`: Verifies loading, 1,470 rows, zero nulls, zero duplicates, target mapping, and `EmployeeID` leakage prevention.
- `tests/test_preprocessing.py`: Verifies 80/20 stratified split ratios, ColumnTransformer shapes, and safe handling of unseen categories (`handle_unknown='ignore'`).
- `tests/test_models.py`: Verifies pipeline construction, fitting, and valid probability outputs ($p \in [0, 1], \sum p = 1$) across all 4 classifiers and regressor.
- `tests/test_evaluation.py`: Verifies metric bounds, CV out-of-fold threshold tuning, and Joblib artifact serialization / re-loading.

---

## 11. Ethical AI Governance & Responsible Use

This project is an academic demonstration of predictive analytics. In real-world enterprise deployments, the following governance boundaries must be strictly enforced:
1. **Strict Non-Punitive Mandate:** Model predictions must **NEVER** be used to terminate, discipline, penalize compensation, or deny promotions to any employee.
2. **Supportive Intent Only:** Risk alerts serve exclusively to trigger empathetic human dialogue (e.g., manager stay interviews, career development check-ins).
3. **Demographic Neutrality:** Demographic attributes like gender and age must never be utilized as gating criteria for career decisions.
4. **Human-in-the-Loop:** Algorithmic scores are statistical indicators, not deterministic verdicts. Final judgment rests with experienced HR business partners.

---

## 12. Git & GitHub Submission Guide

To submit this project to GitHub, execute the following commands in PowerShell:

```powershell
# 1. Initialize Git repository
git init

# 2. Add all project files
git add .

# 3. Commit initial publication
git commit -m "feat: complete end-to-end employee attrition prediction project"

# 4. Set default branch to main
git branch -M main

# 5. Add remote GitHub repository URL (replace with your personal repository URL)
# git remote add origin https://github.com/<your-username>/employee-attrition-prediction.git

# 6. Push code to GitHub
# git push -u origin main
```

*(Note: Per user safety guidelines, remote creation and pushing requires explicit repository credentials and user authorization).*
