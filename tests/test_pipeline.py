import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import TARGET_COLUMN, FEATURE_COLUMNS
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.predict import WineQualityPredictor


def test_data_loader():
    """Test that data loader works and returns proper DataFrame."""
    loader = DataLoader(wine_type="combined")
    df = loader.load_data()
    
    assert isinstance(df, pd.DataFrame), "Loaded data should be a pandas DataFrame"
    assert not df.empty, "DataFrame should not be empty"
    assert TARGET_COLUMN in df.columns, f"Target column '{TARGET_COLUMN}' must exist in dataset"
    assert "wine_type" in df.columns, "wine_type column must exist"
    assert set(FEATURE_COLUMNS).issubset(df.columns), "All feature columns must exist"
    assert df['wine_type'].isin(['red', 'white']).all(), "wine_type must be 'red' or 'white'"
    assert df[TARGET_COLUMN].between(3, 9).all(), "Quality must be between 3 and 9"


def test_data_loader_red_only():
    """Test loading red wine only."""
    loader = DataLoader(wine_type="red")
    df = loader.load_data()
    
    assert isinstance(df, pd.DataFrame), "Loaded data should be a pandas DataFrame"
    assert not df.empty, "DataFrame should not be empty"
    assert (df['wine_type'] == 'red').all(), "All wines should be red"


def test_data_loader_white_only():
    """Test loading white wine only."""
    loader = DataLoader(wine_type="white")
    df = loader.load_data()
    
    assert isinstance(df, pd.DataFrame), "Loaded data should be a pandas DataFrame"
    assert not df.empty, "DataFrame should not be empty"
    assert (df['wine_type'] == 'white').all(), "All wines should be white"


def test_preprocessor():
    """Test that preprocessor works correctly."""
    loader = DataLoader(wine_type="combined")
    df = loader.load_data()
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_pipeline(df)
    
    assert isinstance(X_train, np.ndarray), "X_train should be a numpy array"
    assert isinstance(X_test, np.ndarray), "X_test should be a numpy array"
    assert isinstance(y_train, pd.Series), "y_train should be a pandas Series"
    assert isinstance(y_test, pd.Series), "y_test should be a pandas Series"
    assert len(X_train) > 0, "X_train should not be empty"
    assert len(X_test) > 0, "X_test should not be empty"
    assert len(X_train) + len(X_test) == len(df), "Train + Test should equal total"
    assert len(feature_names) == 12, "Should have 11 numerical + 1 encoded categorical = 12 features"
    
    # Check no data leakage - scaler fitted only on train
    # Train mean should be ~0, test mean can be different
    assert abs(X_train.mean()) < 0.1, "Train features should be centered near 0"


def test_preprocessor_transform_new():
    """Test transforming new data with fitted preprocessor."""
    loader = DataLoader(wine_type="combined")
    df = loader.load_data()
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_pipeline(df)
    
    # Test transforming new data
    new_data = df[FEATURE_COLUMNS].iloc[:5]
    transformed = preprocessor.transform_new_data(new_data)
    
    assert transformed.shape[0] == 5, "Should transform 5 samples"
    assert transformed.shape[1] == len(feature_names), "Should have same number of features"


def test_predictor():
    """Test predictor if model exists."""
    model_path = os.path.join("models", "best_wine_model.joblib")
    scaler_path = os.path.join("models", "scaler.joblib")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        predictor = WineQualityPredictor()
        
        # Test with dict input
        sample = {
            "fixed acidity": 7.4, "volatile acidity": 0.7, "citric acid": 0.0,
            "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11.0,
            "total sulfur dioxide": 34.0, "density": 0.9978, "pH": 3.51,
            "sulphates": 0.56, "alcohol": 9.4, "wine_type": "red"
        }
        result = predictor.predict(sample)
        
        assert "predicted_quality" in result, "Result must contain 'predicted_quality'"
        assert "quality_category" in result, "Result must contain 'quality_category'"
        assert "quality_range" in result, "Result must contain 'quality_range'"
        assert 3 <= result["predicted_quality"] <= 9, "Prediction must be in valid range 3-9"
        assert result["quality_category"] in ["Excellent", "Good", "Average", "Below Average"]
        
        # Test with list input
        sample_list = [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, "red"]
        result2 = predictor.predict(sample_list)
        assert result2["predicted_quality"] == result["predicted_quality"], "Dict and list should give same result"
        
        # Test batch prediction
        batch_result = predictor.predict_batch([sample, sample])
        assert len(batch_result) == 2, "Batch should return 2 predictions"


def test_predictor_invalid_wine_type():
    """Test predictor rejects invalid wine_type."""
    model_path = os.path.join("models", "best_wine_model.joblib")
    scaler_path = os.path.join("models", "scaler.joblib")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        predictor = WineQualityPredictor()
        
        sample = {
            "fixed acidity": 7.4, "volatile acidity": 0.7, "citric acid": 0.0,
            "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11.0,
            "total sulfur dioxide": 34.0, "density": 0.9978, "pH": 3.51,
            "sulphates": 0.56, "alcohol": 9.4, "wine_type": "rose"  # Invalid
        }
        
        with pytest.raises(ValueError, match="wine_type must be 'red' or 'white'"):
            predictor.predict(sample)


def test_feature_count():
    """Test that feature count matches expectation."""
    assert len(FEATURE_COLUMNS) == 12, "Should have 12 features (11 chemical + wine_type)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])