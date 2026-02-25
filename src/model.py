from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
    Schema for capturing comprehensive model performance and metadata.
    Includes tracking for data loss (dropped rows) and class prevalence to 
    monitor potential sampling bias during the filtering process.
    """

@dataclass
class ModelResult:
    model: str
    features: List[str]
    auc: float
    accuracy: float
    n_train: int
    n_test: int
    pos_rate_train: float
    pos_rate_test: float
    tn: int
    fp: int
    fn: int
    tp: int
    # sample / prevalence
    n_rows_original: int | None = None
    n_rows_used: int | None = None
    dropped_rows: int | None = None
    pos_rate_original: float | None = None
    pos_rate_overall: float | None = None
    # interpretability
    coefficients: dict | None = None
    odds_ratios: dict | None = None
    # artifacts
    roc_curve_path: str | None = None


"""
    Executes an end-to-end training and evaluation pipeline.

    Technical Highlights:
    1. Audits data prevalence before and after filtering to ensure representativeness.
    2. Encapsulates Imputer, Scaler, and Model in a single Pipeline object. 
       This ensures that preprocessing parameters (e.g., median, scale) are 
       learned ONLY from the training set, preventing data leakage.
    3. Exports ROC-AUC visualizations for academic validation.
    """

def run_logistic_pipeline(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "depression",
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
    artifacts_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    Train + evaluate a Logistic Regression binary classifier using a Pipeline.

    Pipeline used for
    - Prevents data leakage (scaler fitted ONLY on train)
    - Reproducible preprocessing
    - Deployable as one object
    """
    # --- STEP 1: INITIAL DATA AUDIT (Pre-processing Baseline) ---
    # Record the total number of samples before any filtering to track data loss
    n_rows_original = int(len(df))
    # Calculate baseline prevalence (class balance) to detect potential sampling bias later
    pos_rate_original = float(pd.to_numeric(df[target_col], errors="coerce").mean())

    # --- STEP 2: DATA CLEANING & SUBSET SELECTION ---
    # Define features to be used; MUST be initialized before creating the 'needed' list
    needed = feature_cols + [target_col]
    data = df[needed].copy()
    before = len(data)

    # Handle extreme numerical edge cases: replace Infinity with NaN to allow dropna() to catch them
    data = data.replace([np.inf, -np.inf], np.nan)

    # numeric dtype (robust)
    for c in feature_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")

    # Complete-case filtering
    data = data.dropna(subset=[target_col])
    after = len(data)
    dropped_rows = before -after

    if after == 0:
        raise ValueError("After complete-case filtering, no rows remain. Check missingness / column names.")

        # filtered prevalence
    y_all = data[target_col].astype(int)
    pos_rate_overall = float(y_all.mean())

    if verbose:
        print("\n=== Logistic Regression Pipeline: Data validity ===")
        print(f"Rows before dropna: {before}")
        print(f"Rows after dropna:  {after}")
        print(f"Dropped rows:       {before - after}")
        print(f"Prevalence (original df): {pos_rate_original:.4f}")
        print(f"Prevalence (filtered):    {pos_rate_overall:.4f}")

    # 2) Split X/y
    X = data[feature_cols]
    y = y_all

    # 3) Train/test split (stratify keeps class ratio stable)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 4) Build pipeline
    # IterativeImputer implements MICE (Multivariate Imputation by Chained Equations).
    # This is preferred over simple mean/median imputation as it preserves the
    # statistical relationships between lifestyle features (e.g., Sleep vs. Pressure).
    pipeline = Pipeline([
        ("imputer", IterativeImputer(max_iter=10, random_state=42)),
        ("scaler", RobustScaler()),
        ("model", LogisticRegression(max_iter=5000, solver="liblinear"))
    ])
    # 5) Fit
    pipeline.fit(X_train, y_train)

    # 6) Predict
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    # 7) Evaluate
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.grid(alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Logistic Regression")
    plt.legend(loc="lower right")

    out_dir = artifacts_dir if artifacts_dir is not None else Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    res = ModelResult(
        model="LogisticRegression+RobustScaler(Pipeline)",
        features=feature_cols,
        auc=float(auc),
        accuracy=float(acc),
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
        pos_rate_train=float(y_train.mean()),
        pos_rate_test=float(y_test.mean()),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )

    if verbose:
        print("\n=== Model metrics ===")
        print(f"AUC:      {res.auc:.4f}")
        print(f"Accuracy: {res.accuracy:.4f}")
        print(f"Train pos rate: {res.pos_rate_train:.4f}")
        print(f"Test  pos rate: {res.pos_rate_test:.4f}")
        print("\nConfusion matrix:")
        print(f"TN={res.tn}, FP={res.fp}, FN={res.fn}, TP={res.tp}")

    out = res.__dict__
    # metadata for research traceability
    out.update({
        "n_rows_original": n_rows_original,
        "pos_rate_original": pos_rate_original,
        "n_rows_used": after,
        "dropped_rows": dropped_rows,
        "pos_rate_overall": pos_rate_overall,
        "roc_curve_path": str(out_dir / "roc_curve.png"),
    })

    # return as dictionary
    return out

def run_cross_validation_auc(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "depression",
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    K-fold cross-validation for Logistic Regression Pipeline.
    Returns mean/std AUC across folds.
    """

    needed = feature_cols + [target_col]
    data = df[needed].copy()
    before = len(data)

    # Defensive Cleaning: Handle numerical edge cases (Inf) and force numeric casting.
    # This prevents pipeline failure during matrix operations in Logistic Regression.
    data = data.replace([np.inf, -np.inf], np.nan)

    for c in feature_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")

    data = data.dropna(subset=[target_col])
    after = len(data)

    X = data[feature_cols]
    y = data[target_col].astype(int)

    if verbose:
        # sanity check : check Nan, inf and missing values
        X_num = X.copy()
        print("\n=== Numeric sanity ===")
        print("Any NaN:", X_num.isna().any().to_dict())
        print("Any inf:", np.isinf(X_num.to_numpy()).any())
        abs_max_by_col = X_num.abs().max().sort_values(ascending=False)
        print("\nMax abs value by col:\n", abs_max_by_col.head(10))
        print("\n99.9% quantile by col:\n", X_num.quantile(0.999, numeric_only=True))

    if verbose:
        print("\n=== Cross-Validation: Data validity ===")
        print(f"Rows before dropna: {before}")
        print(f"Rows after dropna:  {after}")
        print(f"Dropped rows:       {before - after}")
        print(f"Positive rate overall: {y.mean():.4f}")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    auc_scores: List[float] = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline = Pipeline([
            ("imputer", IterativeImputer(max_iter=10, random_state=42)),  # 加入 MICE
            ("scaler", RobustScaler()),
            ("model", LogisticRegression(max_iter=5000, solver="liblinear"))
        ])

        pipeline.fit(X_train, y_train)

        y_prob = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        auc_scores.append(float(auc))

        if verbose:
            print(f"Fold {fold}/{n_splits}: AUC={auc:.4f} | "
                  f"train_pos={y_train.mean():.4f} test_pos={y_test.mean():.4f}")

    mean_auc = float(np.mean(auc_scores))
    std_auc = float(np.std(auc_scores, ddof=1)) if len(auc_scores) > 1 else 0.0

    if verbose:
        print("\n=== CV Summary ===")
        print("AUC scores:", [round(a, 4) for a in auc_scores])
        print(f"Mean AUC: {mean_auc:.4f}")
        print(f"Std  AUC: {std_auc:.4f}")

    return {
        "model": "LogisticRegression+RobustScaler(Pipeline)",
        "features": feature_cols,
        "cv": f"StratifiedKFold(n_splits={n_splits})",
        "auc_scores": auc_scores,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "n_rows_used": int(after),
        "dropped_rows": int(before - after),
        "pos_rate_overall": float(y.mean()),
    }
