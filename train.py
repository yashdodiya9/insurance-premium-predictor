"""Train, compare and export the insurance premium category model.

Usage (from the project root):

    uv run python train.py
    uv run python train.py --data data/insurance.csv --output app/model/model.pkl

What it does:
  1. Loads the raw CSV and builds features with app/features.py (the same code the API uses).
  2. Holds out a stratified test set that is never used for model selection.
  3. Compares several models with repeated stratified k-fold cross-validation on the
     remaining training data, reporting accuracy, precision, recall and F1 (macro-averaged).
  4. Picks the model with the best cross-validated macro F1, evaluates it once on the
     held-out test set, then refits it on ALL the data and saves it for the API.
  5. Writes the results to a JSON file next to the model.
"""
import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RAW_COLUMNS,
    TARGET,
    add_features,
)

DEFAULT_DATA = Path("data/insurance.csv")
DEFAULT_MODEL = Path("app/model/model.pkl")
DEFAULT_METRICS = Path("app/model/metrics.json")

SCORING = {
    "accuracy": "accuracy",
    "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
    "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
}


def make_pipeline(classifier, scale_numeric: bool = False) -> Pipeline:
    """Preprocessing + classifier in one object, so the API only ever needs the pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler() if scale_numeric else "passthrough", NUMERIC_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def candidate_models(seed: int) -> dict:
    return {
        "Baseline (most frequent class)": make_pipeline(DummyClassifier(strategy="most_frequent")),
        "Logistic Regression": make_pipeline(
            LogisticRegression(max_iter=1000), scale_numeric=True
        ),
        "Random Forest": make_pipeline(RandomForestClassifier(random_state=seed)),
        "Gradient Boosting": make_pipeline(GradientBoostingClassifier(random_state=seed)),
    }


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in RAW_COLUMNS + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    before = len(df)
    df = df.dropna(subset=RAW_COLUMNS + [TARGET])
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with missing values.")
    return df


def compare_models(models: dict, X, y, n_splits: int, n_repeats: int, seed: int) -> pd.DataFrame:
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=SCORING)
        row = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)


def print_comparison(results: pd.DataFrame, n_splits: int, n_repeats: int) -> None:
    print(f"\nCross-validation ({n_splits}-fold x {n_repeats} repeats), mean +/- std:\n")
    header = f"{'Model':<32}{'Accuracy':>16}{'Precision':>16}{'Recall':>16}{'F1':>16}"
    print(header)
    print("-" * len(header))
    for _, r in results.iterrows():
        cells = [
            f"{r[f'{m}_mean']:.3f} +/- {r[f'{m}_std']:.3f}"
            for m in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
        ]
        print(f"{r['model']:<32}" + "".join(f"{c:>16}" for c in cells))
    print("\n(Precision, recall and F1 are macro-averaged over the three classes.)")


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Raw training CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL, help="Where to save the model")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS, help="Where to save metrics JSON")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction held out for the final test")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    df = add_features(load_data(args.data))
    X, y = df[FEATURE_COLUMNS], df[TARGET]
    print(f"Loaded {len(df)} rows from {args.data}. Class counts:\n{y.value_counts().to_string()}")

    # 1) Hold out a test set that plays no part in model selection.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    # 2) Compare models with cross-validation on the training portion only.
    n_splits = min(args.cv_folds, int(y_train.value_counts().min()))
    if n_splits < 2:
        raise ValueError("Not enough samples in the smallest class for cross-validation.")
    models = candidate_models(args.seed)
    results = compare_models(models, X_train, y_train, n_splits, args.cv_repeats, args.seed)
    print_comparison(results, n_splits, args.cv_repeats)

    best_name = results.loc[0, "model"]
    print(f"\nSelected model (best cross-validated macro F1): {best_name}")

    # 3) One-time evaluation of the selected model on the untouched test set.
    best = clone(models[best_name]).fit(X_train, y_train)
    y_pred = best.predict(X_test)
    labels = sorted(y.unique())
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    print(f"\nHeld-out test set ({len(X_test)} rows):\n")
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(matrix, index=labels, columns=labels).to_string())

    # 4) Refit on all available data for the model the API will serve.
    final_model = clone(models[best_name]).fit(X, y)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(final_model, f)
    print(f"\nSaved model to {args.output}")

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "n_rows": int(len(df)),
        "class_counts": {k: int(v) for k, v in y.value_counts().items()},
        "selected_model": best_name,
        "cv": {"folds": n_splits, "repeats": args.cv_repeats, "results": results.to_dict(orient="records")},
        "test": {
            "n_rows": int(len(X_test)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_macro": float(report["macro avg"]["precision"]),
            "recall_macro": float(report["macro avg"]["recall"]),
            "f1_macro": float(report["macro avg"]["f1-score"]),
            "confusion_matrix": {"labels": labels, "matrix": matrix.tolist()},
        },
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {args.metrics}")
    return metrics


if __name__ == "__main__":
    main()
