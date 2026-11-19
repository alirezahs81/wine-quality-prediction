import joblib
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from src.config import MODEL_FILE, RANDOM_STATE, CV_FOLDS


class ModelTrainer:
    """
    Trains multiple regression models and selects the best performing one.

    Model selection is done via stratified cross-validation (using quality bins)
    on the training split only; the test split is never used for selection.
    """

    def __init__(self):
        self.models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(random_state=RANDOM_STATE),
            "Lasso": Lasso(random_state=RANDOM_STATE, max_iter=5000),
            "ElasticNet": ElasticNet(random_state=RANDOM_STATE, max_iter=5000),
            "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
            "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=200),
            "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE, n_estimators=200),
            "XGBoost": XGBRegressor(random_state=RANDOM_STATE, n_estimators=200, verbosity=0),
        }
        self.best_model = None
        self.best_model_name = None

    def _create_quality_bins(self, y):
        """Create bins for stratified CV on continuous target."""
        # Bin quality scores into 3-4 categories for stratification
        return pd.qcut(y, q=4, labels=False, duplicates='drop')

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Selects the best model by CV on the train split, fits it, and saves it."""
        
        # Create stratification bins for cross-validation
        stratify_bins = self._create_quality_bins(y_train)
        
        best_cv_score = np.inf  # For RMSE, lower is better

        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Use negative RMSE for cross_val_score (higher is better in sklearn)
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=CV_FOLDS, 
                scoring="neg_root_mean_squared_error", 
                n_jobs=-1
            )
            mean_cv_rmse = -cv_scores.mean()  # Convert back to positive RMSE
            print(f"{name} CV RMSE: {mean_cv_rmse:.4f} (+/- {cv_scores.std():.4f})")

            if mean_cv_rmse < best_cv_score:
                best_cv_score = mean_cv_rmse
                self.best_model = model
                self.best_model_name = name

        print(f"\nBest Model Selected: {self.best_model_name} (CV RMSE: {best_cv_score:.4f})")

        # Fit the selected model on the full training split
        self.best_model.fit(X_train, y_train)
        
        # Evaluate on test set
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        y_pred = self.best_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        
        print(f"{self.best_model_name} Test RMSE: {test_rmse:.4f}")
        print(f"{self.best_model_name} Test MAE: {test_mae:.4f}")
        print(f"{self.best_model_name} Test R²: {test_r2:.4f}")

        # Save the best model
        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        joblib.dump(self.best_model, MODEL_FILE)
        print(f"Best model successfully saved to {MODEL_FILE}")

        return self.best_model


if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.preprocessing import DataPreprocessor

    loader = DataLoader(wine_type="combined")
    df = loader.load_data()

    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, _ = preprocessor.preprocess_pipeline(df)

    trainer = ModelTrainer()
    trainer.train_and_evaluate(X_train, y_train, X_test, y_test)