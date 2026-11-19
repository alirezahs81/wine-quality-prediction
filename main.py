from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.predict import WineQualityPredictor
import joblib
import os
from src.config import MODEL_FILE

def main():
    print("=== Starting Wine Quality Prediction Pipeline ===")

    # Step 1: Load Data
    print("\n[Step 1/5] Loading Data...")
    loader = DataLoader(wine_type="combined")
    df = loader.load_data()
    print(f"Data loaded successfully with shape: {df.shape}")
    print(f"Quality range: {df['quality'].min()} - {df['quality'].max()}")
    print(f"Wine types: {df['wine_type'].value_counts().to_dict()}")

    # Step 2: Preprocess Data
    print("\n[Step 2/5] Preprocessing Data & Scaling...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_pipeline(df)
    print("Preprocessing completed. Training set shape:", X_train.shape)
    print("Features:", feature_names)

    # Step 3: Train Model
    print("\n[Step 3/5] Training Models & Selecting Best Estimator...")
    trainer = ModelTrainer()
    best_model = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)

    # Step 4: Evaluate Model
    print("\n[Step 4/5] Evaluating Best Model...")
    y_pred = best_model.predict(X_test)
    ModelEvaluator.print_evaluation_report(y_test, y_pred)

    # Step 5: Demo Prediction
    print("\n[Step 5/5] Running Sample Inference...")
    predictor = WineQualityPredictor()
    
    # Sample red wine
    sample_red = {
        "fixed acidity": 7.4, "volatile acidity": 0.7, "citric acid": 0.0,
        "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0, "density": 0.9978, "pH": 3.51,
        "sulphates": 0.56, "alcohol": 9.4, "wine_type": "red"
    }
    prediction_result = predictor.predict(sample_red)
    print("\nSample Red Wine Features:", sample_red)
    print("Prediction Result:", prediction_result)
    
    # Sample white wine
    sample_white = {
        "fixed acidity": 6.8, "volatile acidity": 0.27, "citric acid": 0.36,
        "residual sugar": 20.7, "chlorides": 0.045, "free sulfur dioxide": 45.0,
        "total sulfur dioxide": 170.0, "density": 1.001, "pH": 3.0,
        "sulphates": 0.45, "alcohol": 8.8, "wine_type": "white"
    }
    prediction_result = predictor.predict(sample_white)
    print("\nSample White Wine Features:", sample_white)
    print("Prediction Result:", prediction_result)

    print("\n=== Pipeline Completed Successfully! ===")
    print(f"\nModel saved to: {MODEL_FILE}")
    print("You can now use the predictor in your own code:")
    print("  from src.predict import WineQualityPredictor")
    print("  predictor = WineQualityPredictor()")
    print("  result = predictor.predict(your_wine_features)")

if __name__ == "__main__":
    main()