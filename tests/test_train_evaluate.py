"""Unit tests for training selection and evaluation metrics (synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.evaluate import ModelEvaluator
from src.train import ModelTrainer


@pytest.fixture
def tiny_xy():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 8))
    # Linear-ish target so Ridge can fit something non-trivial
    y = pd.Series(X[:, 0] * 0.4 + X[:, 1] * 0.2 + rng.normal(0, 0.1, size=120) + 5.5)
    y = y.clip(3, 9)
    return X, y


def test_evaluate_predictions_keys(tiny_xy):
    X, y = tiny_xy
    model = Ridge(random_state=42).fit(X, y)
    pred = model.predict(X)
    metrics = ModelEvaluator.evaluate_predictions(y, pred)
    assert set(metrics) == {"rmse", "mae", "r2", "mape"}
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0
    assert metrics["r2"] <= 1.0


def test_print_evaluation_report_runs(tiny_xy, capsys):
    X, y = tiny_xy
    pred = Ridge(random_state=42).fit(X, y).predict(X)
    ModelEvaluator.print_evaluation_report(y, pred)
    out = capsys.readouterr().out
    assert "RMSE" in out
    assert "Model Evaluation Report" in out


def test_model_trainer_selects_and_fits(tiny_xy, tmp_path, monkeypatch):
    """Trainer should pick a model by CV and persist it."""
    X, y = tiny_xy
    # Split manually
    X_train, X_test = X[:90], X[90:]
    y_train, y_test = y.iloc[:90], y.iloc[90:]

    model_path = tmp_path / "model.joblib"
    monkeypatch.setattr("src.train.MODEL_FILE", str(model_path))

    # Speed up: shrink model zoo for this unit test
    trainer = ModelTrainer()
    trainer.models = {
        "Ridge": Ridge(random_state=42),
        "LinearRegression": __import__("sklearn.linear_model", fromlist=["LinearRegression"]).LinearRegression(),
    }

    best = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
    assert best is not None
    assert trainer.best_model_name in trainer.models
    assert model_path.exists()
    pred = best.predict(X_test)
    assert len(pred) == len(y_test)


def test_quality_bins_length(tiny_xy):
    _, y = tiny_xy
    trainer = ModelTrainer()
    bins = trainer._create_quality_bins(y)
    assert len(bins) == len(y)
