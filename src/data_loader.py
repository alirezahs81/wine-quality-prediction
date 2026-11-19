import os
import pandas as pd
import numpy as np
from src.config import RAW_DATA_FILE, RED_DATA_FILE, WHITE_DATA_FILE, RAW_DATA_DIR

class DataLoader:
    """
    Handles data loading for Wine Quality dataset.
    Supports combined, red-only, or white-only datasets.
    """
    def __init__(self, data_path: str = RAW_DATA_FILE, wine_type: str = "combined"):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to the CSV file
            wine_type: 'combined', 'red', or 'white'
        """
        self.wine_type = wine_type
        if wine_type == "red":
            self.data_path = RED_DATA_FILE
        elif wine_type == "white":
            self.data_path = WHITE_DATA_FILE
        else:
            self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        """Loads data from CSV or generates sample data if not present."""
        if os.path.exists(self.data_path):
            print(f"Loading data from local file: {self.data_path}")
            df = pd.read_csv(self.data_path)
            if "wine_type" not in df.columns and self.wine_type != "combined":
                df["wine_type"] = self.wine_type
            return df
        else:
            print("Local dataset not found. Generating sample Wine Quality data...")
            return self._generate_and_save_sample()

    def _generate_and_save_sample(self) -> pd.DataFrame:
        """Generates sample Wine Quality data based on UCI dataset statistics."""
        np.random.seed(42)
        
        if self.wine_type == "red" or self.wine_type == "combined":
            n_red = 1599
            red = pd.DataFrame({
                'fixed acidity': np.random.normal(8.32, 1.74, n_red).clip(4, 16),
                'volatile acidity': np.random.normal(0.53, 0.18, n_red).clip(0.1, 1.6),
                'citric acid': np.random.normal(0.27, 0.19, n_red).clip(0, 1),
                'residual sugar': np.random.normal(2.54, 1.41, n_red).clip(0.5, 16),
                'chlorides': np.random.normal(0.087, 0.047, n_red).clip(0.01, 0.6),
                'free sulfur dioxide': np.random.normal(15.87, 10.46, n_red).clip(1, 72),
                'total sulfur dioxide': np.random.normal(46.47, 32.89, n_red).clip(6, 289),
                'density': np.random.normal(0.9968, 0.0019, n_red).clip(0.99, 1.01),
                'pH': np.random.normal(3.31, 0.15, n_red).clip(2.7, 4.1),
                'sulphates': np.random.normal(0.66, 0.17, n_red).clip(0.3, 2),
                'alcohol': np.random.normal(10.42, 1.07, n_red).clip(8, 15),
                'quality': np.random.choice([3, 4, 5, 6, 7, 8], n_red, p=[0.01, 0.07, 0.41, 0.37, 0.12, 0.02])
            })
            red['wine_type'] = 'red'
        else:
            red = pd.DataFrame()

        if self.wine_type == "white" or self.wine_type == "combined":
            n_white = 4898
            white = pd.DataFrame({
                'fixed acidity': np.random.normal(6.85, 0.84, n_white).clip(4, 12),
                'volatile acidity': np.random.normal(0.28, 0.10, n_white).clip(0.08, 1.1),
                'citric acid': np.random.normal(0.33, 0.12, n_white).clip(0, 1),
                'residual sugar': np.random.normal(6.39, 5.07, n_white).clip(0.6, 66),
                'chlorides': np.random.normal(0.045, 0.021, n_white).clip(0.01, 0.35),
                'free sulfur dioxide': np.random.normal(35.31, 16.94, n_white).clip(2, 289),
                'total sulfur dioxide': np.random.normal(138.36, 62.49, n_white).clip(9, 440),
                'density': np.random.normal(0.9940, 0.0017, n_white).clip(0.99, 1.01),
                'pH': np.random.normal(3.19, 0.10, n_white).clip(2.7, 3.8),
                'sulphates': np.random.normal(0.49, 0.11, n_white).clip(0.2, 1.1),
                'alcohol': np.random.normal(10.51, 1.18, n_white).clip(8, 14.5),
                'quality': np.random.choice([3, 4, 5, 6, 7, 8, 9], n_white, p=[0.005, 0.03, 0.15, 0.47, 0.28, 0.06, 0.005])
            })
            white['wine_type'] = 'white'
        else:
            white = pd.DataFrame()

        combined = pd.concat([red, white], ignore_index=True)
        
        # Save to appropriate file
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        combined.to_csv(self.data_path, index=False)
        print(f"Sample dataset generated and saved to {self.data_path}")
        
        return combined

if __name__ == "__main__":
    loader = DataLoader(wine_type="combined")
    df = loader.load_data()
    print("Dataset shape:", df.shape)
    print(df.head())
    print("\nQuality distribution:")
    print(df['quality'].value_counts().sort_index())
    print("\nWine type distribution:")
    print(df['wine_type'].value_counts())