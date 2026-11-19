import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """
    Evaluates trained regression models using various metrics.
    """
    @staticmethod
    def evaluate_predictions(y_true, y_pred) -> dict:
        """
        Computes RMSE, MAE, R², MAPE.
        """
        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred) * 100
        }
        return metrics

    @staticmethod
    def print_evaluation_report(y_true, y_pred):
        """Prints a detailed evaluation report with metrics and prediction analysis."""
        print("\n--- Model Evaluation Report ---")
        metrics = ModelEvaluator.evaluate_predictions(y_true, y_pred)
        
        for metric_name, value in metrics.items():
            if metric_name == "mape":
                print(f"{metric_name.upper()}: {value:.2f}%")
            else:
                print(f"{metric_name.upper()}: {value:.4f}")

        # Additional analysis
        residuals = y_true - y_pred
        print(f"\nResiduals - Mean: {residuals.mean():.4f}, Std: {residuals.std():.4f}")
        
        # Accuracy within thresholds
        for threshold in [0.5, 1.0, 1.5]:
            within = np.abs(residuals) <= threshold
            print(f"Predictions within ±{threshold}: {within.mean()*100:.1f}%")

    @staticmethod
    def plot_predictions(y_true, y_pred, save_path=None):
        """Creates visualization plots for predictions vs actual."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Predicted vs Actual scatter
        ax = axes[0, 0]
        ax.scatter(y_true, y_pred, alpha=0.5, s=10)
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Quality')
        ax.set_ylabel('Predicted Quality')
        ax.set_title('Predicted vs Actual Quality')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Residuals plot
        ax = axes[0, 1]
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Quality')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Predicted')
        ax.grid(True, alpha=0.3)
        
        # 3. Residuals distribution
        ax = axes[1, 0]
        ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title('Residuals Distribution')
        ax.grid(True, alpha=0.3)
        
        # 4. Quality-wise accuracy
        ax = axes[1, 1]
        quality_levels = sorted(y_true.unique())
        accuracies = []
        for q in quality_levels:
            mask = y_true == q
            if mask.sum() > 0:
                acc = (np.abs(residuals[mask]) <= 0.5).mean() * 100
                accuracies.append(acc)
            else:
                accuracies.append(0)
        ax.bar([str(q) for q in quality_levels], accuracies, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Actual Quality')
        ax.set_ylabel('Accuracy within ±0.5 (%)')
        ax.set_title('Accuracy by Quality Level')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Evaluation plots saved to {save_path}")
        plt.show()

if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.preprocessing import DataPreprocessor
    import joblib
    import os
    from src.config import MODEL_FILE

    loader = DataLoader(wine_type="combined")
    df = loader.load_data()

    preprocessor = DataPreprocessor()
    _, X_test, _, y_test, _ = preprocessor.preprocess_pipeline(df)

    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
        y_pred = model.predict(X_test)
        ModelEvaluator.print_evaluation_report(y_test, y_pred)
        ModelEvaluator.plot_predictions(y_test, y_pred)
    else:
        print("Model file not found. Train the model first.")