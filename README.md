# 🍷 Wine Quality Prediction System

[![CI](https://github.com/alirezahs81/wine-quality-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/alirezahs81/wine-quality-prediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-EC6231?logo=xgboost&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end Machine Learning project for predicting wine quality based on physicochemical properties (UCI Wine Quality dataset). Built with a clean, modular, **leakage-free** and production-ready Python architecture.

---

## 📊 Dataset

The project uses the **UCI Wine Quality dataset** containing physicochemical properties of red and white variants of Portuguese "Vinho Verde" wine:

| Feature | Description |
|---------|-------------|
| **Fixed Acidity** | Tartaric acid concentration (g/dm³) |
| **Volatile Acidity** | Acetic acid concentration (g/dm³) |
| **Citric Acid** | Citric acid concentration (g/dm³) |
| **Residual Sugar** | Sugar remaining after fermentation (g/dm³) |
| **Chlorides** | Salt concentration (g/dm³) |
| **Free Sulfur Dioxide** | Free SO₂ (mg/dm³) |
| **Total Sulfur Dioxide** | Total SO₂ (mg/dm³) |
| **Density** | Wine density (g/cm³) |
| **pH** | Acidity level (3-4 range) |
| **Sulphates** | Potassium sulphate concentration (g/dm³) |
| **Alcohol** | Alcohol percentage (% vol) |
| **Wine Type** | Red or White (categorical) |

**Target:** Quality score (3-9, higher is better)

---

## 📂 Project Architecture

```text
wine-quality-prediction/
├── data/
│   └── raw/                 # Raw dataset (auto-downloaded from UCI on first run)
├── models/                  # Saved model + preprocessing pipeline (joblib)
├── src/
│   ├── config.py            # Global configurations & paths
│   ├── data_loader.py       # Data ingestion (supports red, white, combined)
│   ├── preprocessing.py     # Encoding, scaling & train-test split (leakage-free)
│   ├── train.py             # Model training & CV-based model selection
│   ├── evaluate.py          # RMSE, MAE, R², MAPE + visualizations
│   └── predict.py           # Inference pipeline for new wines
├── tests/                   # Automated unit tests with pytest
├── .github/workflows/ci.yml # GitHub Actions CI
├── main.py                  # Main execution script
├── requirements.txt
└── LICENSE
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/alirezahs81/wine-quality-prediction.git
cd wine-quality-prediction
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Run the Full Pipeline
Downloads the data (if needed), preprocesses it, trains and selects the best model, evaluates it, and runs sample inference:
```bash
python main.py
```

### Run Unit Tests
```bash
pytest tests/ -v
```

### Predict Quality for a New Wine
```python
from src.predict import WineQualityPredictor

predictor = WineQualityPredictor()

# Using a dictionary (recommended)
sample_wine = {
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
    "wine_type": "red"  # or "white"
}

result = predictor.predict(sample_wine)
print(result)
# {'predicted_quality': 5.67, 'quality_category': 'Average', 'quality_range': '3-9 (higher is better)'}
```

### Load Specific Wine Type Only
```python
from src.data_loader import DataLoader

# Load only red wines
loader = DataLoader(wine_type="red")
df_red = loader.load_data()

# Load only white wines
loader = DataLoader(wine_type="white")
df_white = loader.load_data()

# Load combined (default)
loader = DataLoader(wine_type="combined")
df_combined = loader.load_data()
```

---

## 🔬 Methodology Highlights

- **Leakage-free preprocessing:** Categorical encoding (OneHot) and scaling (StandardScaler) are fitted **on the training split only** via scikit-learn's `ColumnTransformer`. The entire pipeline is persisted and reused verbatim at inference time.
- **Honest model selection:** 8 regression models compared by **5-fold cross-validated RMSE** on the training split; the test set is never used for selection.
- **Multiple algorithms evaluated:** LinearRegression, Ridge, Lasso, ElasticNet, DecisionTree, RandomForest, GradientBoosting, XGBoost.
- **Regression metrics:** RMSE, MAE, R², MAPE + residual analysis and quality-level accuracy.
- **Visualizations:** Predicted vs Actual, Residuals, Residuals Distribution, Accuracy by Quality Level.
- **Deterministic:** All randomness seeded (`RANDOM_STATE = 42`) for reproducibility.
- **Flexible data loading:** Supports combined, red-only, or white-only datasets.

---

## 📈 Expected Performance

| Model | CV RMSE (train) | Test RMSE | Test MAE | Test R² |
|-------|----------------|-----------|----------|---------|
| Random Forest ⭐ | 0.625 | 0.607 | 0.435 | 0.50 |
| XGBoost | 0.651 | ~0.63 | ~0.45 | ~0.45 |
| Gradient Boosting | 0.678 | ~0.66 | ~0.48 | ~0.42 |
| Ridge/Linear | 0.734 | ~0.73 | ~0.55 | ~0.27 |

> The selected estimator is the one with the best cross-validated RMSE, refitted on the full training split and evaluated once on the held-out test split.

---

## 📝 Author

Developed with a clean, professional, and scalable Python Machine Learning architecture.

## 📄 License

This project is licensed under the [MIT License](LICENSE).
