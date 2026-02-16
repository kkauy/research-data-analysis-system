from pathlib import Path
from data_loader import load_student_data
from eda import run_basic_eda
from stats_analysis import run_statistical_analysis
from model import run_logistic_pipeline

def main():
    # project root
    project_root = Path(__file__).resolve().parent.parent

    data_path = Path(__file__).resolve().parent.parent / "data" / "student_depression_dataset.csv"

    # output folder for research figures
    output_dir = project_root / "artifacts"

    print("Loading dataset...")
    df = load_student_data(data_path)

    print(df["depression"].dtype)

    print("Running research EDA...")
    run_basic_eda(df, output_dir)

    results_dir = project_root / "results"
    run_statistical_analysis(df, results_dir)

    print("\nDataset loaded successfully.")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nPreview:")
    print(df.head())
    print("Research EDA completed.")

    print(f"Artifacts saved to: {output_dir}")

    sleep_col_candidates = [c for c in df.columns if "sleep" in c]
    print("Sleep-like columns:", sleep_col_candidates)

    features = ["sleep_duration", "age", "cgpa", "academic_pressure", "work_study_hours"]

    ml_results = run_logistic_pipeline(df, features)
    print("\nML Results dict:")
    print(ml_results)

if __name__ == "__main__":
    main()