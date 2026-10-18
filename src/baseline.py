"""Reproducible validation baselines for the chess-opening project."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

TARGET = "Result"
CLASS_LABELS = ("0-1", "1-0", "1/2-1/2")
NUMERIC_FEATURES = ("AverageElo", "EloDiff")
CATEGORICAL_FEATURES = ("ECO", "TimeControl")
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RANDOM_STATE = 42


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a cleaned Parquet or CSV dataset."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Add the cleaned M2 dataset or pass "
            "its path with --input."
        )

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, engine="pyarrow")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset format {suffix!r}; use .parquet or .csv.")


def prepare_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select only pre-game predictors and validate the target and feature values."""
    required = [*FEATURES, TARGET]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Dataset contains no rows.")

    X = frame.loc[:, FEATURES].copy()
    y = frame[TARGET].astype("string").str.strip()

    for column in NUMERIC_FEATURES:
        X[column] = pd.to_numeric(X[column], errors="coerce")
        finite = np.isfinite(X[column].to_numpy(dtype=float, na_value=np.nan))
        if not finite.all():
            raise ValueError(f"Feature {column!r} contains missing or non-numeric values.")

    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].astype("string")

    if X.isna().any().any() or y.isna().any():
        raise ValueError("Dataset contains missing values in required features or target.")

    observed_labels = set(y.unique())
    unexpected_labels = observed_labels - set(CLASS_LABELS)
    if unexpected_labels:
        labels = ", ".join(sorted(str(label) for label in unexpected_labels))
        raise ValueError(f"Unexpected {TARGET} labels: {labels}")
    missing_labels = set(CLASS_LABELS) - observed_labels
    if missing_labels:
        labels = ", ".join(sorted(missing_labels))
        raise ValueError(f"Dataset does not contain all outcome classes: {labels}")

    return X, y.astype(str)


def split_dataset(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create the fixed, stratified 80%/15%/5% train/validation/test split."""
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_remaining,
    )
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_models() -> dict[str, Any]:
    """Build the agreed naive and course-level baseline models."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ],
        remainder="passthrough",
    )
    tree = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=8,
                    min_samples_leaf=50,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return {
        "Majority Class": DummyClassifier(strategy="most_frequent"),
        "Decision Tree": tree,
    }


def _probabilities_in_label_order(model: Any, X: pd.DataFrame) -> np.ndarray:
    model_classes = list(model.classes_)
    missing_classes = set(CLASS_LABELS) - set(model_classes)
    if missing_classes:
        labels = ", ".join(sorted(missing_classes))
        raise ValueError(f"Model was not trained on all outcome classes: {labels}")
    class_indices = [model_classes.index(label) for label in CLASS_LABELS]
    return model.predict_proba(X)[:, class_indices]


def evaluate_baselines(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit both models on training data and report validation-only metrics."""
    X, y = prepare_dataset(frame)
    X_train, X_validation, X_test, y_train, y_validation, y_test = split_dataset(X, y)
    models = build_models()

    metric_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_validation)
        probabilities = _probabilities_in_label_order(model, X_validation)
        metric_rows.append(
            {
                "model": name,
                "split": "validation",
                "n": len(y_validation),
                "accuracy": accuracy_score(y_validation, predictions),
                "macro_f1": f1_score(
                    y_validation,
                    predictions,
                    labels=CLASS_LABELS,
                    average="macro",
                    zero_division=0,
                ),
                "log_loss": log_loss(
                    y_validation,
                    probabilities,
                    labels=CLASS_LABELS,
                ),
            }
        )

        report = classification_report(
            y_validation,
            predictions,
            labels=CLASS_LABELS,
            output_dict=True,
            zero_division=0,
        )
        for label in CLASS_LABELS:
            class_rows.append(
                {
                    "model": name,
                    "class": label,
                    "precision": report[label]["precision"],
                    "recall": report[label]["recall"],
                    "f1": report[label]["f1-score"],
                    "support": int(report[label]["support"]),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    class_metrics = pd.DataFrame(class_rows)
    metadata = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "features": list(FEATURES),
        "target": TARGET,
        "class_labels": list(CLASS_LABELS),
        "split_rows": {
            "train": len(y_train),
            "validation": len(y_validation),
            "test_reserved": len(y_test),
        },
        "test_evaluated": False,
        "decision_tree": {
            "max_depth": 8,
            "min_samples_leaf": 50,
            "random_state": RANDOM_STATE,
        },
    }
    return metrics, class_metrics, metadata


def write_outputs(
    output_dir: Path,
    metrics: pd.DataFrame,
    class_metrics: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "baseline_table.csv", index=False, float_format="%.6f")
    class_metrics.to_csv(
        output_dir / "classification_report.csv",
        index=False,
        float_format="%.6f",
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/lichess_clean_2016_08.parquet"),
        help="Cleaned M2 Parquet or CSV dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/generated"),
        help="Directory for the baseline table, class report, and run metadata.",
    )
    args = parser.parse_args()

    frame = load_dataset(args.input)
    metrics, class_metrics, metadata = evaluate_baselines(frame)
    metadata["input"] = str(args.input)
    write_outputs(args.output_dir, metrics, class_metrics, metadata)

    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nReports written to {args.output_dir}")


if __name__ == "__main__":
    main()
