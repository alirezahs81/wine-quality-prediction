import joblib
import numpy as np
import pandas as pd
import os
from src.config import MODEL_FILE, SCALER_FILE, FEATURE_COLUMNS

class WineQualityPredictor:
    """
    Handles inference/prediction for new wine samples using the trained model and preprocessor.
    """
    def __init__(self):
        if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
            raise FileNotFoundError(
                "Trained model or preprocessor not found. "
                "Please train the model first using main.py."
            )
        
        self.model = joblib.load(MODEL_FILE)
        self.preprocessor = joblib.load(SCALER_FILE)

    def predict(self, input_data: dict | list) -> dict:
        """
        Takes raw wine features, preprocesses them, and returns quality prediction.
        
        Args:
            input_data: Either a dict with feature names or a list in the correct order.
                       Expected features (11 chemical + wine_type):
                       fixed acidity, volatile acidity, citric acid, residual sugar,
                       chlorides, free sulfur dioxide, total sulfur dioxide, density,
                       pH, sulphates, alcohol, wine_type ('red' or 'white')
        
        Returns:
            dict with prediction, quality category, and confidence info
        """
        # Convert input to DataFrame with proper feature names
        if isinstance(input_data, dict):
            # Ensure all required features are present
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            if len(input_data) != len(FEATURE_COLUMNS):
                raise ValueError(
                    f"Expected {len(FEATURE_COLUMNS)} features, got {len(input_data)}. "
                    f"Order: {FEATURE_COLUMNS}"
                )
            df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
        else:
            raise TypeError("input_data must be a dict or list")
        
        # Validate wine_type
        if 'wine_type' in df.columns:
            wine_type = df['wine_type'].iloc[0].lower()
            if wine_type not in ['red', 'white']:
                raise ValueError("wine_type must be 'red' or 'white'")
            df['wine_type'] = wine_type
        
        # Preprocess input
        scaled_input = self.preprocessor.transform(df[FEATURE_COLUMNS])
        
        # Make prediction
        prediction = self.model.predict(scaled_input)[0]
        
        # Clip prediction to valid quality range
        prediction = np.clip(prediction, 3, 9)
        
        # Determine quality category
        if prediction >= 7:
            quality_category = "Excellent"
        elif prediction >= 6:
            quality_category = "Good"
        elif prediction >= 5:
            quality_category = "Average"
        else:
            quality_category = "Below Average"
        
        result = {
            "predicted_quality": round(float(prediction), 2),
            "quality_category": quality_category,
            "quality_range": "3-9 (higher is better)"
        }
        return result

    def predict_batch(self, input_list: list) -> list:
        """Predict quality for multiple wine samples at once."""
        return [self.predict(sample) for sample in input_list]


if __name__ == "__main__":
    # Example samples for testing prediction
    # Format: [fixed acidity, volatile acidity, citric acid, residual sugar,
    #          chlorides, free sulfur dioxide, total sulfur dioxide, density,
    #          pH, sulphates, alcohol, wine_type]
    
    sample_red = {
        "fixed acidity": 7.4,
        "volatile acidity": 0.7,
        "citric acid": 0.0,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
        "wine_type": "red"
    }
    
    sample_white = {
        "fixed acidity": 6.8,
        "volatile acidity": 0.27,
        "citric acid": 0.36,
        "residual sugar": 20.7,
        "chlorides": 0.045,
        "free sulfur dioxide": 45.0,
        "total sulfur dioxide": 170.0,
        "density": 1.001,
        "pH": 3.0,
        "sulphates": 0.45,
        "alcohol": 8.8,
        "wine_type": "white"
    }
    
    try:
        predictor = WineQualityPredictor()
        
        print("--- Red Wine Sample ---")
        res = predictor.predict(sample_red)
        print("Features:", sample_red)
        print("Prediction Result:", res)
        
        print("\n--- White Wine Sample ---")
        res = predictor.predict(sample_white)
        print("Features:", sample_white)
        print("Prediction Result:", res)
        
    except Exception as e:
        print("Error:", e)
        print("Train the model first with: python main.py")