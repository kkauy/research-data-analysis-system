from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

    Why Pipeline:
    - Prevents data leakage (scaler fitted ONLY on train)
    - Reproducible preprocessing
    - Deployable as one object
    """

    # 1) Select needed columns + drop missing
    needed = feature_cols + [target_col]
    data = df[needed].copy()

    before = len(data)
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
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
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
        model="LogisticRegression+StandardScaler(Pipeline)",
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
