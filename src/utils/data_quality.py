import numpy as np
import pandas as pd

def print_data_quality(X: pd.DataFrame, return_dict: bool = False) -> dict:
    print("\n=== Data Quality Report ===")

    # Missing values
    missing_pct = (X.isna().mean() * 100).round(2)
    print("\nMissing % by feature:")
    print(missing_pct[missing_pct > 0])

    # Infinite values
    has_inf = np.isinf(X.to_numpy()).any()
    print("\nContains inf:", has_inf)

    # Extreme values
    abs_max = X.abs().max().sort_values(ascending=False)
    print("\nMax absolute value (top 10):")
    print(abs_max.head(10))

    # Distribution tail check
    quantile_999 = X.quantile(0.999, numeric_only=True)
    print("\n99.9% quantile:")
    print(quantile_999)

    if return_dict:
        return {
            'missing_pct': missing_pct[missing_pct > 0].to_dict(),
            'has_inf': bool(has_inf),
            'max_values': abs_max.head(10).to_dict(),
            'quantile_999': quantile_999.to_dict()
        }
