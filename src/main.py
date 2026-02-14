from pathlib import Path
from data_loader import load_student_data
from eda import run_basic_eda

def main():
    # project root
    project_root = Path(__file__).resolve().parent.parent

    data_path = Path(__file__).resolve().parent.parent / "data" / "student_depression_dataset.csv"

    # output folder for research figures
    output_dir = project_root / "artifacts"

    print("Loading dataset...")
    df = load_student_data(data_path)

    print("Running research EDA...")
    run_basic_eda(df, output_dir)

    print("\nDataset loaded successfully.")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nPreview:")
    print(df.head())
    print("Research EDA completed.")

    print(f"Artifacts saved to: {output_dir}")

if __name__ == "__main__":
    main()