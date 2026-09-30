import json
import pickle
from pathlib import Path
import pandas as pd
import pytest
import train
from app.features import FEATURE_COLUMNS, add_features

DATA = Path(__file__).resolve().parent.parent / "data" / "insurance.csv"


def test_training_pipeline_end_to_end(tmp_path):
    model_path = tmp_path / "model.pkl"
    metrics_path = tmp_path / "metrics.json"

    train.main([
        "--data", str(DATA),
        "--output", str(model_path),
        "--metrics", str(metrics_path),
        "--cv-repeats", "1",  # keep the test quick
    ])

    assert model_path.exists() and metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())

    # The chosen model must clearly beat always predicting the majority class.
    results = {r["model"]: r for r in metrics["cv"]["results"]}
    baseline = results["Baseline (most frequent class)"]["f1_macro_mean"]
    chosen = results[metrics["selected_model"]]["f1_macro_mean"]
    assert chosen > baseline + 0.2

    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        assert 0 <= metrics["test"][key] <= 1

    # The saved artifact must be loadable and usable exactly the way the API uses it.
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    X = add_features(pd.read_csv(DATA).head(5))[FEATURE_COLUMNS]
    assert len(model.predict(X)) == 5
    assert model.predict_proba(X).sum(axis=1).tolist() == pytest.approx([1.0] * 5)
    assert set(model.classes_) == {"High", "Low", "Medium"}
