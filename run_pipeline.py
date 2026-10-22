"""
Master Orchestration Script for Employee Attrition Prediction Pipeline.
Executes end-to-end data validation, EDA, model training, cross-validation,
hyperparameter tuning, held-out evaluation, explainability, and report generation.
"""

from __future__ import annotations
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.utils import set_seed, setup_logger, save_joblib_artifact, ExecutionTimer
from src.data_loader import load_raw_data
from src.data_validation import validate_dataset, get_column_summary
from src.preprocessing import split_and_save_data
from src.eda import run_full_eda
from src.models import (
    get_all_classifiers,
    get_baseline_pipeline,
    get_optional_regression_pipeline,
)
from src.tuning import tune_all_models
from src.evaluation import (
    evaluate_cross_validation,
    evaluate_model_on_test,
    find_optimal_f1_threshold,
    plot_evaluation_charts,
    export_evaluation_tables,
)
from src.feature_importance import compare_feature_importances
from src.reporting import generate_evaluation_report, generate_business_report

logger = setup_logger("master_pipeline")


def main():
    set_seed(42)
    print("\n" + "="*80)
    print("   EMPLOYEE ATTRITION PREDICTION: END-TO-END MACHINE LEARNING PIPELINE")
    print("="*80 + "\n")

    # Step 1: Data Loading & Validation
    print(">>> [1/9] Loading and Validating Raw Dataset...")
    df = load_raw_data()
    val_report = validate_dataset(df)
    print(f"    - Total records: {val_report['total_records']}, Total features: {val_report['total_columns']}")
    print(f"    - Missing values: {val_report['has_missing']} | Duplicates: {val_report['duplicate_rows']}")
    print(f"    - Target class distribution: {val_report['target_distribution_counts']}")
    print(f"    - Target proportions: {val_report['target_distribution_pct']}")

    # Step 2: Exploratory Data Analysis (EDA)
    print("\n>>> [2/9] Generating All 13 Exploratory Visualizations...")
    eda_fig_paths = run_full_eda(df)
    print(f"    - Successfully generated and saved {len(eda_fig_paths)} EDA charts to outputs/figures/")

    # Step 3: Train-Test Split (80% Train, 20% Test)
    print("\n>>> [3/9] Performing Stratified Train-Test Split (80/20)...")
    X_train, X_test, y_train, y_test = split_and_save_data(df, test_size=0.20, random_state=42)
    pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"    - X_train: {X_train.shape}, y_train: {y_train.shape} (Positive ratio: {pos_ratio:.2f})")
    print(f"    - X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    # Step 4: Cross-Validation on Untuned Base Models
    print("\n>>> [4/9] Running 5-Fold Stratified Cross-Validation on Base Classifiers...")
    base_models = get_all_classifiers(scale_pos_weight=pos_ratio)
    cv_base_df = evaluate_cross_validation(base_models, X_train, y_train, cv_folds=5, random_state=42)
    print("\n--- 5-Fold Stratified Cross-Validation Benchmark (Training Partition) ---")
    print(cv_base_df[["Model", "Accuracy_Mean", "Precision_Mean", "Recall_Mean", "F1_Mean", "ROC_AUC_Mean"]].to_string(index=False))

    # Step 5: Systematic Hyperparameter Tuning
    print("\n>>> [5/9] Performing Systematic Hyperparameter Tuning (Stratified 5-Fold CV)...")
    tuning_results = tune_all_models(
        base_models, X_train, y_train,
        search_type="randomized", n_iter=15, cv_folds=5, scoring="f1", random_state=42
    )

    # Compile final pipeline candidates (including Baseline)
    final_models = {"Baseline": get_baseline_pipeline()}
    for name, tune_res in tuning_results.items():
        final_models[name] = tune_res["best_estimator"]

    # Step 6: Evaluation on Untouched Held-Out Test Set (20%)
    print("\n>>> [6/9] Evaluating Tuned Models on Untouched Test Set (N=294)...")
    test_eval_results = {}
    for name, pipeline in final_models.items():
        eval_res = evaluate_model_on_test(pipeline, X_train, y_train, X_test, y_test, name, threshold=0.5)
        test_eval_results[name] = eval_res

    # Build Test comparison summary
    test_rows = []
    for name, res in test_eval_results.items():
        test_rows.append({
            "Model": name,
            "Accuracy": res["accuracy"],
            "Precision": res["precision"],
            "Recall": res["recall"],
            "F1_Score": res["f1_score"],
            "ROC_AUC": res["roc_auc"],
            "TP": res["true_positives"],
            "FP": res["false_positives"],
            "TN": res["true_negatives"],
            "FN": res["false_negatives"],
            "Duration (s)": res["fit_duration_seconds"],
        })
    test_summary_df = pd.DataFrame(test_rows)
    print("\n--- Final Held-Out Test Set Performance Comparison ---")
    print(test_summary_df.to_string(index=False))

    # Threshold Optimization Analysis for Best Model
    best_candidate_name = "Random Forest"
    best_candidate_model = final_models[best_candidate_name]
    opt_thresh, opt_cv_f1 = find_optimal_f1_threshold(best_candidate_model, X_train, y_train, cv_folds=5)
    opt_eval_res = evaluate_model_on_test(
        best_candidate_model, X_train, y_train, X_test, y_test,
        f"{best_candidate_name} (Threshold={opt_thresh:.2f})", threshold=opt_thresh
    )
    print(f"\n--- Threshold Optimization for {best_candidate_name} ---")
    print(f"    - Default Threshold (0.50): F1={test_eval_results[best_candidate_name]['f1_score']:.4f}, Recall={test_eval_results[best_candidate_name]['recall']:.4f}, Precision={test_eval_results[best_candidate_name]['precision']:.4f}")
    print(f"    - Optimal Threshold ({opt_thresh:.2f}): F1={opt_eval_res['f1_score']:.4f}, Recall={opt_eval_res['recall']:.4f}, Precision={opt_eval_res['precision']:.4f}")

    # Step 7: Plotting Evaluation Figures & Tables
    print("\n>>> [7/9] Generating Evaluation Curves and Exporting Result Tables...")
    eval_fig_paths = plot_evaluation_charts(test_eval_results, y_test)
    test_csv, excel_path = export_evaluation_tables(cv_base_df, test_eval_results)

    # Feature Importance Comparison
    df_feat_comp, feat_fig_path = compare_feature_importances(final_models, X_test, y_test)
    all_fig_paths = {**eda_fig_paths, **eval_fig_paths, "feature_importance": feat_fig_path}

    # Step 8: Optional Regression Task (MonthlyIncome)
    print("\n>>> [8/9] Executing Optional Continuous Regression Task (Predicting MonthlyIncome)...")
    reg_pipeline = get_optional_regression_pipeline()
    y_reg_train = X_train["MonthlyIncome"]
    X_reg_train = X_train.drop(columns=["MonthlyIncome"])
    y_reg_test = X_test["MonthlyIncome"]
    X_reg_test = X_test.drop(columns=["MonthlyIncome"])

    reg_pipeline.fit(X_reg_train, y_reg_train)
    y_reg_pred = reg_pipeline.predict(X_reg_test)
    reg_mae = mean_absolute_error(y_reg_test, y_reg_pred)
    reg_rmse = root_mean_squared_error(y_reg_test, y_reg_pred)
    reg_r2 = r2_score(y_reg_test, y_reg_pred)

    print(f"    - Regression Model: RandomForestRegressor")
    print(f"    - Test MAE:  ${reg_mae:,.2f}")
    print(f"    - Test RMSE: ${reg_rmse:,.2f}")
    print(f"    - Test R²:   {reg_r2:.4f}")

    # Persist Models & Metadata
    print("\n>>> Persisting Serialized Pipelines and Technical Metadata...")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    save_joblib_artifact(final_models["Random Forest"], models_dir / "best_model_pipeline.joblib")
    save_joblib_artifact(final_models["Logistic Regression"], models_dir / "logistic_regression_pipeline.joblib")
    save_joblib_artifact(final_models["Decision Tree"], models_dir / "decision_tree_pipeline.joblib")
    save_joblib_artifact(final_models["Random Forest"], models_dir / "random_forest_pipeline.joblib")
    if "XGBoost" in final_models:
        save_joblib_artifact(final_models["XGBoost"], models_dir / "xgboost_pipeline.joblib")

    # Metadata JSON
    metadata = {
        "project_name": "Employee Attrition Prediction Models",
        "dataset_name": "IBM HR Analytics Employee Attrition",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_records": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "target_positive_class": "Yes (1)",
        "selected_production_model": "Random Forest Classifier",
        "selected_model_test_metrics": {
            "accuracy": test_eval_results["Random Forest"]["accuracy"],
            "precision": test_eval_results["Random Forest"]["precision"],
            "recall": test_eval_results["Random Forest"]["recall"],
            "f1_score": test_eval_results["Random Forest"]["f1_score"],
            "roc_auc": test_eval_results["Random Forest"]["roc_auc"],
        },
        "best_hyperparameters": {
            name: t_res["clean_best_params"] for name, t_res in tuning_results.items()
        },
        "optimal_threshold": opt_thresh,
        "optional_regression_metrics": {
            "target": "MonthlyIncome",
            "MAE": round(reg_mae, 2),
            "RMSE": round(reg_rmse, 2),
            "R2": round(reg_r2, 4)
        }
    }
    with open(models_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    # Step 9: PDF Report Generation
    print("\n>>> [9/9] Generating Professional PDF Reports...")
    eval_pdf = generate_evaluation_report(test_eval_results, cv_base_df, all_fig_paths)
    biz_pdf = generate_business_report(test_eval_results, all_fig_paths)
    print(f"    - Generated Technical Evaluation Report: {eval_pdf}")
    print(f"    - Generated Business Interpretation Report: {biz_pdf}")

    print("\n" + "="*80)
    print("   PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
