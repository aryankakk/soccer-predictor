from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class TrainedModel:
    pipeline: Pipeline
    label_order: list[str]


def build_pipeline(feature_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    num_cols = [c for c in feature_cols if c not in categorical_cols]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=1,
        C=1.0,
    )

    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def train_multiclass(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    test_fraction: float = 0.2,
) -> tuple[TrainedModel, dict[str, Any]]:
    """Train on played matches only; time-based split."""
    data = df.dropna(subset=["result", "date"]).sort_values("date").copy()
    y = data["result"].astype(str)
    X = data[feature_cols].copy()

    n = len(data)
    split = int(round(n * (1 - test_fraction)))
    split = max(1, min(n - 1, split))

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    pipe = build_pipeline(feature_cols=feature_cols, categorical_cols=categorical_cols)
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)
    pred = pipe.predict(X_test)

    labels = list(pipe.named_steps["clf"].classes_)

    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "labels": labels,
        "accuracy": float(accuracy_score(y_test, pred)),
        "log_loss": float(log_loss(y_test, proba, labels=labels)),
        "report": classification_report(y_test, pred, digits=4),
    }

    return TrainedModel(pipeline=pipe, label_order=labels), metrics


def predict_proba(model: TrainedModel, X: pd.DataFrame) -> pd.DataFrame:
    proba = model.pipeline.predict_proba(X)
    return pd.DataFrame(proba, columns=[f"p_{c}" for c in model.label_order])


def save_model(model: TrainedModel, path: str) -> None:
    dump({"pipeline": model.pipeline, "label_order": model.label_order}, path)


def load_model(path: str) -> TrainedModel:
    obj = load(path)
    return TrainedModel(pipeline=obj["pipeline"], label_order=list(obj["label_order"]))
