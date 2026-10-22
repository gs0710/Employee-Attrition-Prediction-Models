# Dataset Documentation: Employee Attrition

## 1. Overview and Source
This dataset contains corporate human resources attributes designed to model and understand the factors driving voluntary and involuntary employee turnover (attrition). The dataset adheres to the schema and behavioral distribution established in the landmark IBM HR Analytics Employee Attrition dataset, which was published as an open research and benchmarking dataset for workforce analytics.

- **Primary Source**: IBM HR Analytics Employee Attrition benchmark study.
- **Dataset Size**: 1,470 employee records across 22 operational and demographic attributes.
- **Primary Target**: `Attrition` (Binary: `Yes` [left organization] vs `No` [stayed with organization]).
- **Target Distribution**:
  - `No` (Retained): 1,196 employees (81.36%)
  - `Yes` (Attrited): 274 employees (18.64%)
  - **Imbalance Ratio**: Approximately 4.37:1 majority-to-minority ratio.

## 2. Directory Structure
```
data/
├── raw/
│   └── employee_data.csv       # Pristine, unmodified raw dataset
├── processed/
│   ├── train_features.csv      # Stratified training features (80% split)
│   ├── test_features.csv       # Stratified held-out test features (20% split)
│   ├── train_target.csv        # Training target vector (y_train)
│   └── test_target.csv         # Untouched test target vector (y_test)
├── data_dictionary.csv         # Detailed variable dictionary and metadata
└── README.md                   # This document
```

## 3. Data Integrity and Verification
- **Missing Values**: 0 missing values across all 22 columns.
- **Duplicate Records**: 0 duplicate rows.
- **Constant / Zero-Variance Features**: None (all features exhibit meaningful variance).
- **Identifier Exclusion**: `EmployeeID` is strictly an administrative tracking key and must be dropped before feature processing to prevent memorization / spurious correlation.
- **Demographic Columns**: `Gender` is retained for fairness auditing and parity checks, with explicit guidelines that predictive models should never be used to make punitive individual employment decisions.

## 4. Usage Terms and Responsible Use
This dataset is provided for academic, pedagogical, and benchmarking evaluation. In corporate deployments, employee attrition predictions must serve as aggregate diagnostic signals for workplace health and organizational support—not automated punitive actions, compensation penalization, or discriminatory hiring/firing practices.
