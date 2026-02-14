from pathlib import Path
import pandas as pd



def load_student_data(path: Path) -> pd.DataFrame:
    """
        Load and clean the student mental health dataset.

        Steps:
        - Read CSV file
        - Drop completely empty columns
        - Standardize column names
        - Remove missing or invalid rows

        Returns
        -------
        pd.DataFrame
            Cleaned dataset ready for analysis.
        """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)

    #Drop empty columns
    df = df.dropna(axis = 1, how="all")

    #Standarze column names
    df.columns = df.columns.str.strip().str.lower().str.replace("","_")

    #Remove rows with missing vallues
    df = df.dropna()

    df.columns = [clean_col(c) for c in df.columns]

    return df

def clean_col(col: str) -> str:
    col = col.strip().lower()
    col = col.replace(" ", "_")
    col = col.replace("/", "_")
    col = col.replace("?", "")
    # remove duplicate underscore
    col = col.replace("__", "_")
    col = col.strip("_")

    parts = col.split("_")
    if all(len(p) == 1 for p in parts if p):
        col = "".join([p for p in parts if p])

    return col
