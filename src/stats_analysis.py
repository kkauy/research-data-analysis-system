import pandas as pd
from scipy.stats import mannwhitneyu
import numpy as np
from pathlib import Path


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(
        ((nx - 1)*np.var(x, ddof=1) + (ny - 1)*np.var(y, ddof=1)) / (nx + ny - 2)
    )
    return (np.mean(x) - np.mean(y)) / pooled_std


def sleep_vs_depression(df: pd.DataFrame,
                        sleep_col: str = "sleep_duration",
                        dep_col: str = "depression") -> dict:
    # keep only needed cols and drop missing
    tmp = df[[sleep_col, dep_col]].copy()
    #make sure no NaN for Mann–Whitney U
    tmp = tmp.dropna(subset=[sleep_col, dep_col])

    # groups
    g0 = tmp.loc[tmp[dep_col] == 0, sleep_col].astype(float)
    g1 = tmp.loc[tmp[dep_col] == 1, sleep_col].astype(float)

    # Mann–Whitney U (robust for non-normal)
    u_stat, p_value = mannwhitneyu(g0, g1, alternative="two-sided")

    d = cohens_d(g0, g1)

    return {
        "factor": sleep_col,
        "outcome": dep_col,
        "test": "Mann-Whitney U",
        "u_stat": float(u_stat),
        "p_value": float(p_value),
        "effect_size_cohens_d": float(d),
        "n_depr0": int(len(g0)),
        "n_depr1": int(len(g1)),
        "mean_depr0": float(g0.mean()) if len(g0) else np.nan,
        "mean_depr1": float(g1.mean()) if len(g1) else np.nan,
    }


def run_statistical_analysis(df: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = [sleep_vs_depression(df)]
    res = pd.DataFrame(rows)

    out_path = results_dir / "stat_summary.csv"
    res.to_csv(out_path, index=False)

    print("\n Statistical results saved to:", out_path)
    print(res)

    return res
