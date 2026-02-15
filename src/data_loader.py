from pathlib import Path
import pandas as pd


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

def parse_sleep(x):
    """
    Convert sleepdurationb values into numeric hours
    suvey formats like:
    -'5 - 6 hours'
    -'Less than 5 hours'
    -'More than 8 hours'
    -'7 hours'
    """
    if pd.isna(x):
        return None

    #Numeric
    if isinstance(x,(int, float)):
        return float(x)

    s = str(x).strip().lower()

    #normalize
    s = s.replace("hours","").replace("hours","").strip()

    if "less than" in s:
        return 4.0
    if "more than" in s:
        return 9.0
    if  "_" in s:
        parts = [p.strip() for p in s.split("_")]
        try:
            a = float(parts[0])
            b = float(parts[1])
            return (a + b) / 2.0
        except Exception:
            return None

    try:
        return float(s)
    except Exception:
        return None

def load_student_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)

    # drop  empty columns
    df = df.dropna(axis=1, how="all")

    # clean column names
    df.columns = [clean_col(c) for c in df.columns]


    if "depression" in df.columns:
        df = df.dropna(subset=["depression"])

    # parse sleepduration to numeric if present
    if "sleep_duration" in df.columns:
        df["sleep_duration"] = df["sleep_duration"].apply(parse_sleep)

    return df


