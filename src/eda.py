import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def run_basic_eda(df: pd.DataFrame, output_dir: Path):
    """
    Perform research-grade exploratory data analysis.
    Saves visualizations for research reporting.
    """

    output_dir.mkdir(exist_ok=True)

    print("\n===== DATASET OVERVIEW =====")
    print("Shape:", df.shape)
    print("\nColumn types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())

    # Depression distribution
    plt.figure()
    df["depression"].value_counts().plot(kind="bar")
    plt.title("Depression Distribution")
    plt.savefig(output_dir / "depression_distribution.png")
    plt.close()

    # Age distribution
    plt.figure()
    sns.histplot(df["age"], bins=20, kde=True)
    plt.title("Age Distribution")
    plt.savefig(output_dir / "age_distribution.png")
    plt.close()

    # Sleep vs Depression
    plt.figure()
    sns.boxplot(x="depression", y="sleepduration", data=df)
    plt.title("Sleep Duration vs Depression")
    plt.savefig(output_dir / "sleep_vs_depression.png")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.savefig(output_dir / "correlation_heatmap.png")
    plt.close()

    print("\nEDA visualizations saved to:", output_dir)
