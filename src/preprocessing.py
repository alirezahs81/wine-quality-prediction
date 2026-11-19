import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os
from src.config import RANDOM_STATE, TEST_SIZE, SCALER_FILE, TARGET_COLUMN, FEATURE_COLUMNS

class DataPreprocessor:
    """
    Handles data preprocessing for Wine Quality prediction:
    - Encodes categorical variable (wine_type)
    - Scales numerical features
    - Splits data (train/test) with no leakage
    - Saves the fitted preprocessing pipeline for inference
    """
    
    # Numerical features (all chemical properties)
    NUMERICAL_FEATURES = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol"
    ]
    
    # Categorical features
    CATEGORICAL_FEATURES = ["wine_type"]

    def __init__(self):
        # Create column transformer: scale numerical, encode categorical
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.NUMERICAL_FEATURES),
                ("cat", OneHotEncoder(drop="first", sparse_output=False), self.CATEGORICAL_FEATURES),
            ],
            remainder="drop"  # Drop any columns not specified (like target)
        )

    def preprocess_pipeline(self, df: pd.DataFrame):
        """
        Runs the full preprocessing pipeline:
        1. Select features and target
        2. Train-test split (before any fitting, to avoid leakage)
        3. Fit ColumnTransformer on training split only
        4. Transform both train and test
        Returns X_train_scaled, X_test_scaled, y_train, y_test, feature_names.
        """
        # Step 1: Select features and target
        missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns in DataFrame: {missing_cols}")
        
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame.")

        X = df[FEATURE_COLUMNS]
        y = df[TARGET_COLUMN]

        # Step 2: Train-Test Split (before any fitting, to avoid leakage)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=None
        )

        # Step 3: Fit preprocessor on training split only
        X_train_scaled = self.preprocessor.fit_transform(X_train)
        X_test_scaled = self.preprocessor.transform(X_test)

        # Get feature names after transformation
        num_features = self.NUMERICAL_FEATURES
        cat_encoder = self.preprocessor.named_transformers_["cat"]
        cat_features = cat_encoder.get_feature_names_out(self.CATEGORICAL_FEATURES).tolist()
        feature_names = num_features + cat_features

        # Step 4: Save the fitted preprocessing pipeline for inference use
        os.makedirs(os.path.dirname(SCALER_FILE), exist_ok=True)
        joblib.dump(self.preprocessor, SCALER_FILE)
        print(f"Preprocessing pipeline successfully saved to {SCALER_FILE}")

        return X_train_scaled, X_test_scaled, y_train, y_test, feature_names

    def transform_new_data(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using the fitted preprocessor."""
        return self.preprocessor.transform(df[FEATURE_COLUMNS])

if __name__ == "__main__":
    from src.data_loader import DataLoader

    loader = DataLoader(wine_type="combined")
    df = loader.load_data()

    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_pipeline(df)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    print("Features:", feature_names)
    print("\nSample X_train (first 3 rows):")
    print(X_train[:3])
    print("\nSample y_train (first 10):")
    print(y_train[:10].values)