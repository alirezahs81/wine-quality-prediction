"""Train a lightweight model for CI / local smoke tests.

Uses a reduced model set and fewer estimators so GitHub Actions stays fast,
while still writing MODEL_FILE and SCALER_FILE for WineQualityPredictor tests.
"""

from __future__ import annotations

import os

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from src.config import MODEL_FILE, RANDOM_STATE, SCALER_FILE
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor


def main() -> None:
    n_estimators = int(os.getenv("CI_N_ESTIMATORS", "40"))
    print(f"CI fast-train: n_estimators={n_estimators}")

    df = DataLoader(wine_type="combined").load_data()
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, _ = preprocessor.preprocess_pipeline(df)

    candidates = {
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_estimators=n_estimators,
            n_jobs=-1,
        ),
    }

    best_name, best_model, best_rmse = None, None, float("inf")
    from sklearn.metrics import mean_squared_error
    import numpy as np

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        print(f"{name} test RMSE: {rmse:.4f}")
        if rmse < best_rmse:
            best_rmse, best_name, best_model = rmse, name, model

    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(best_model, MODEL_FILE)
    # SCALER_FILE already written by preprocessor
    assert os.path.exists(SCALER_FILE), "scaler missing after preprocess"
    print(f"Saved best model ({best_name}) -> {MODEL_FILE}")
    print(f"Preprocessor -> {SCALER_FILE}")


if __name__ == "__main__":
    main()
