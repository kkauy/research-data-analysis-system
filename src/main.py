from pathlib import Path

from src.data_loader import load_student_data
from src.eda import run_basic_eda
from src.stats_analysis import run_statistical_analysis
from src.model import run_logistic_pipeline, run_cross_validation_auc
from src.utils.data_quality import print_data_quality



def main():
    # project root
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "student_depression_dataset.csv"

    # output folders
    artifacts_dir = project_root / "artifacts"
    results_dir = project_root / "results"

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) Load ----
    print("Loading dataset...")
    df = load_student_data(data_path)

    features = ["sleep_duration", "age", "cgpa", "academic_pressure", "work_study_hours"]

    print("\n=== Dataset Overview ===")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Preview:\n", df.head())

    # ---- 2) Data Quality (high-level) ----
    print("\n=== Data Quality: full dataset ===")
    print_data_quality(df[features + ["depression"]])

    # quick helper: show sleep columns
    sleep_cols = [c for c in df.columns if "sleep" in c]
    print("\nSleep-like columns:", sleep_cols)

    # ---- 3) EDA ----
    print("\nRunning research EDA...")
    run_basic_eda(df, artifacts_dir)
    print(f"EDA artifacts saved to: {artifacts_dir}")



    # ---- 4) Statistical inference ----
    print("\nRunning statistical analysis...")
    run_statistical_analysis(df, results_dir)
    print(f"Statistical results saved to: {results_dir}")


    missing_cols = [c for c in features + ["depression"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print("\n=== Data Quality: model features only ===")
    print_data_quality(df[features + ["depression"]])

    print("\nRunning single split baseline (Logistic Regression)...")
    ml_results = run_logistic_pipeline(df, features)
    print("\nML Results dict:")
    print(ml_results)

    print("\nRunning cross-validation...")
    cv_res = run_cross_validation_auc(df, features, n_splits=5)
    print("\nCV Results dict:")
    print(cv_res)


if __name__ == "__main__":
    main()
