import pandas as pd
from scipy.stats import mannwhitneyu
import numpy as np

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(
        ((nx - 1)*np.var(x, ddof=1) + (ny - 1)*np.var(y, ddof=1)) / (nx + ny - 2)
    )
    return (np.mean(x) - np.mean(y)) / pooled_std


def sleep_vs_depression(df):
    g0 = df[df["depression"] == 0]["sleepduration"]
    g1 = df[df["depression"] == 1]["sleepduration"]

    stat, p = mannwhitneyu(g0, g1, alternative="two-sided")

    d = cohens_d(g0, g1)

    return {
        "p_value": p,
        "effect_size_d": d,
        "n0": len(g0),
        "n1": len(g1),
    }
