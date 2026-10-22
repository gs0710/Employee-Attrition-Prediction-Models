"""
Exploratory Data Analysis (EDA) module.
Generates all 13 required analytical visualizations for employee attrition.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import get_project_root, setup_logger

logger = setup_logger("eda")

# Global aesthetic styling
PALETTE_ATTRITION = {"No": "#2b5c8f", "Yes": "#d9534f", 0: "#2b5c8f", 1: "#d9534f"}
ACCENT_COLOR = "#1f77b4"
HIGHLIGHT_COLOR = "#d9534f"


def set_plotting_style() -> None:
    """Set clean, publication-grade styling for Matplotlib and Seaborn."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "figure.autolayout": True,
    })


def run_full_eda(
    df: pd.DataFrame,
    output_dir: Optional[str | Path] = None,
    target_col: str = "Attrition"
) -> Dict[str, str]:
    """
    Generate all 13 required exploratory visualizations and save them to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Complete employee dataset.
    output_dir : Optional[str | Path]
        Directory path to store figure artifacts. Defaults to 'outputs/figures'.
    target_col : str
        Target column name (e.g. 'Attrition').

    Returns
    -------
    Dict[str, str]
        Dictionary mapping figure names to saved file paths.
    """
    set_plotting_style()
    if output_dir is None:
        save_path = get_project_root() / "outputs" / "figures"
    else:
        save_path = Path(output_dir)
        if not save_path.is_absolute():
            save_path = get_project_root() / save_path
    save_path.mkdir(parents=True, exist_ok=True)

    figure_paths: Dict[str, str] = {}
    logger.info(f"Generating EDA figures in: {save_path}")

    # Standardized string representation of target for plots
    target_str = df[target_col].astype(str)

    # -------------------------------------------------------------
    # 1 & 2. Overall Employee Attrition & Imbalance Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = target_str.value_counts()
    pcts = target_str.value_counts(normalize=True) * 100
    bars = ax.bar(counts.index, counts.values, color=["#2b5c8f", "#d9534f"], width=0.55, edgecolor="black", linewidth=0.8)
    for bar, pct in zip(bars, pcts.values):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 20,
            f"{int(yval):,} ({pct:.1f}%)",
            ha="center", va="bottom", fontweight="bold", fontsize=11
        )
    ax.set_title("Overall Employee Attrition Distribution (Class Imbalance)", pad=15)
    ax.set_xlabel("Attrition Status")
    ax.set_ylabel("Employee Headcount")
    ax.set_ylim(0, counts.max() * 1.15)
    p1 = save_path / "01_attrition_distribution.png"
    fig.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figure_paths["attrition_distribution"] = str(p1)

    # -------------------------------------------------------------
    # 3. Department vs Attrition
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    dept_df = df.groupby(["Department", target_col]).size().unstack(fill_value=0)
    dept_rates = (dept_df["Yes"] / dept_df.sum(axis=1) * 100).round(1)
    dept_df.plot(kind="bar", stacked=False, color=["#2b5c8f", "#d9534f"], ax=ax, edgecolor="black", linewidth=0.8)
    ax.set_title("Department Headcount and Attrition Breakdown", pad=15)
    ax.set_xlabel("Department")
    ax.set_ylabel("Employee Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.legend(title="Attrition", labels=["No (Stayed)", "Yes (Left)"])
    for i, (dept, rate) in enumerate(dept_rates.items()):
        ax.text(i, dept_df.loc[dept].max() * 0.95, f"Turnover:\n{rate}%", ha="center", fontweight="bold", color="#a94442", fontsize=10)
    p3 = save_path / "03_department_vs_attrition.png"
    fig.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figure_paths["department_vs_attrition"] = str(p3)

    # -------------------------------------------------------------
    # 4. Job Role vs Attrition
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    role_df = df.groupby("JobRole")[target_col].apply(lambda x: (x == "Yes").mean() * 100).sort_values(ascending=True)
    bars = ax.barh(role_df.index, role_df.values, color="#e67e22", edgecolor="black", linewidth=0.8)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.8, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center", fontweight="bold")
    ax.set_title("Attrition Rate (%) by Job Role", pad=15)
    ax.set_xlabel("Attrition Rate (%)")
    ax.set_ylabel("Job Role")
    ax.set_xlim(0, max(role_df.values) * 1.18)
    p4 = save_path / "04_job_role_vs_attrition.png"
    fig.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figure_paths["job_role_vs_attrition"] = str(p4)

    # -------------------------------------------------------------
    # 5. Overtime vs Attrition
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ot_df = df.groupby(["Overtime", target_col]).size().unstack(fill_value=0)
    ot_rates = (ot_df["Yes"] / ot_df.sum(axis=1) * 100).round(1)
    ot_df.plot(kind="bar", color=["#2b5c8f", "#d9534f"], ax=ax, edgecolor="black", linewidth=0.8)
    ax.set_title("Impact of Mandatory OverTime on Employee Turnover", pad=15)
    ax.set_xlabel("Overtime Requirement")
    ax.set_ylabel("Headcount")
    ax.set_xticklabels(["No Overtime", "Works Overtime"], rotation=0)
    ax.legend(title="Attrition", labels=["No", "Yes"])
    for i, (ot, rate) in enumerate(ot_rates.items()):
        ax.text(i, ot_df.loc[ot].max() * 0.9, f"Turnover Rate:\n{rate}%", ha="center", fontweight="bold", color="#a94442", fontsize=10)
    p5 = save_path / "05_overtime_vs_attrition.png"
    fig.savefig(p5, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figure_paths["overtime_vs_attrition"] = str(p5)

    # -------------------------------------------------------------
    # 6. Job Satisfaction vs Attrition
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sat_df = df.groupby(["JobSatisfaction", target_col]).size().unstack(fill_value=0)
    sat_rates = (sat_df["Yes"] / sat_df.sum(axis=1) * 100).round(1)
    sat_df.plot(kind="bar", color=["#2b5c8f", "#d9534f"], ax=ax, edgecolor="black", linewidth=0.8)
    ax.set_title("Job Satisfaction (1=Low, 4=Very High) vs Turnover", pad=15)
    ax.set_xlabel("Job Satisfaction Rating")
    ax.set_ylabel("Employee Count")
    ax.set_xticklabels(["1 - Low", "2 - Medium", "3 - High", "4 - Very High"], rotation=0)
    ax.legend(title="Attrition", labels=["No", "Yes"])
    for i, (sat, rate) in enumerate(sat_rates.items()):
        ax.text(i, sat_df.loc[sat].max() * 0.9, f"{rate}% Left", ha="center", fontweight="bold", color="#a94442", fontsize=9)
    p6 = save_path / "06_job_satisfaction_vs_attrition.png"
    fig.savefig(p6, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figure_paths["job_satisfaction_vs_attrition"] = str(p6)

    # -------------------------------------------------------------
    # 7. Years at Company vs Attrition
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x=target_col, y="YearsAtCompany", hue=target_col, palette=["#2b5c8f", "#d9534f"], ax=ax, width=0.45, legend=False)
    ax.set_title("Tenure at Company Distribution by Attrition Status", pad=15)
    ax.set_xlabel("Attrition Status")
    ax.set_ylabel("Years at Current Company")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Retained (No)", "Departed (Yes)"])
    p7 = save_path / "07_years_at_company_vs_attrition.png"
    fig.savefig(p7, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figure_paths["years_at_company_vs_attrition"] = str(p7)

    # -------------------------------------------------------------
    # 8. Monthly Income vs Attrition
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x=target_col, y="MonthlyIncome", hue=target_col, palette=["#2b5c8f", "#d9534f"], ax=ax, width=0.45, legend=False)
    ax.set_title("Monthly Income Distribution by Attrition Status", pad=15)
    ax.set_xlabel("Attrition Status")
    ax.set_ylabel("Monthly Income (USD)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Retained (No)", "Departed (Yes)"])
    p8 = save_path / "08_monthly_income_vs_attrition.png"
    fig.savefig(p8, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figure_paths["monthly_income_vs_attrition"] = str(p8)

    # -------------------------------------------------------------
    # 9. Age Distribution by Attrition
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.kdeplot(data=df[df[target_col] == "No"], x="Age", fill=True, color="#2b5c8f", label="Retained (No)", alpha=0.4, ax=ax)
    sns.kdeplot(data=df[df[target_col] == "Yes"], x="Age", fill=True, color="#d9534f", label="Departed (Yes)", alpha=0.4, ax=ax)
    ax.set_title("Age Density Estimation by Attrition Status", pad=15)
    ax.set_xlabel("Age (Years)")
    ax.set_ylabel("Density")
    ax.legend()
    p9 = save_path / "09_age_distribution_by_attrition.png"
    fig.savefig(p9, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figure_paths["age_distribution_by_attrition"] = str(p9)

    # -------------------------------------------------------------
    # 10. Years Since Last Promotion vs Attrition
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x=target_col, y="YearsSinceLastPromotion", hue=target_col, palette=["#2b5c8f", "#d9534f"], ax=ax, width=0.45, legend=False)
    ax.set_title("Years Since Last Promotion by Attrition Status", pad=15)
    ax.set_xlabel("Attrition Status")
    ax.set_ylabel("Years Elapsed Since Promotion")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Retained (No)", "Departed (Yes)"])
    p10 = save_path / "10_promotion_years_vs_attrition.png"
    fig.savefig(p10, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figure_paths["promotion_years_vs_attrition"] = str(p10)

    # -------------------------------------------------------------
    # 11. Correlation Heatmap for Numerical Variables
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 9))
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[num_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, square=True, linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Correlation Heatmap: Numerical Workforce Features", pad=15)
    p11 = save_path / "11_correlation_heatmap.png"
    fig.savefig(p11, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figure_paths["correlation_heatmap"] = str(p11)

    # -------------------------------------------------------------
    # 12. Numerical Feature Distributions (Multi-panel)
    # -------------------------------------------------------------
    key_numerics = ["Age", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany", "YearsWithCurrManager", "AbsenteeismDays"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(key_numerics):
        sns.histplot(df[col], kde=True, color="#2b5c8f", ax=axes[i], bins=20)
        axes[i].set_title(f"Distribution: {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
    fig.suptitle("Key Numerical Feature Distributions", fontsize=15, fontweight="bold")
    fig.tight_layout()
    p12 = save_path / "12_numerical_distributions.png"
    fig.savefig(p12, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figure_paths["numerical_distributions"] = str(p12)

    # -------------------------------------------------------------
    # 13. Selected Categorical Feature Distributions (Multi-panel)
    # -------------------------------------------------------------
    key_categoricals = ["Department", "EducationLevel", "BusinessTravel", "Gender"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()
    for i, col in enumerate(key_categoricals):
        counts = df[col].value_counts()
        sns.barplot(x=counts.index, y=counts.values, palette="Blues_r", ax=axes[i], edgecolor="black", linewidth=0.8)
        axes[i].set_title(f"Composition: {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        axes[i].tick_params(axis="x", rotation=20)
    fig.suptitle("Categorical Workforce Demographic & Context Features", fontsize=15, fontweight="bold")
    fig.tight_layout()
    p13 = save_path / "13_categorical_distributions.png"
    fig.savefig(p13, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figure_paths["categorical_distributions"] = str(p13)

    logger.info(f"Successfully generated and saved all {len(figure_paths)} EDA visualizations.")
    return figure_paths
