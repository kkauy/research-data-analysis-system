from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix


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


def run_logistic_pipeline(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "depression",
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train + evaluate a Logistic Regression binary classifier using a Pipeline.

    Pipeline used for
    - Prevents data leakage (scaler fitted ONLY on train)
    - Reproducible preprocessing
    - Deployable as one object
    """

    # 1) Select needed columns + drop missing
    needed = feature_cols + [target_col]
    data = df[needed].copy()
    before = len(data)

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=needed)
    after = len(data)

    # numeric dtype (robust)
    for c in feature_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")

    data = data.dropna(subset=needed)
    after = len(data)

    if verbose:
        print("\n=== Logistic Regression Pipeline: Data validity ===")
        print(f"Rows before dropna: {before}")
        print(f"Rows after dropna:  {after}")
        print(f"Dropped rows:       {before - after}")

    # 2) Split X/y
    X = data[feature_cols]
    y = data[target_col].astype(int)

    # 3) Train/test split (stratify keeps class ratio stable)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 4) Build pipeline
    pipeline = Pipeline([
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

    return res.__dict__

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

    data = data.replace([np.inf, -np.inf], np.nan)

    for c in feature_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")

    data = data.dropna(subset=needed)
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
