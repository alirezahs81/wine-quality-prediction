# 🍷 Wine Quality Prediction System

[![CI](https://github.com/alirezahs81/wine-quality-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/alirezahs81/wine-quality-prediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-EC6231?logo=xgboost&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end Machine Learning project for predicting wine quality based on physicochemical properties (UCI Wine Quality dataset). Built with a clean, modular, **leakage-free** and production-ready Python architecture — plus a **FastAPI** inference service and **Docker** image.

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
├── data/raw/                # Raw dataset (auto-downloaded on first run)
├── models/                  # Saved model + preprocessor (joblib)
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── api.py               # FastAPI inference service
├── scripts/train_ci.py      # Lightweight train for CI / Docker
├── tests/
├── Dockerfile
├── main.py
└── requirements.txt
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/alirezahs81/wine-quality-prediction.git
cd wine-quality-prediction
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 💻 Usage

### Full training pipeline
```bash
python main.py
```

### Unit tests
```bash
# Optional: build model artifacts first (CI does this automatically)
python scripts/train_ci.py
pytest tests/ -v
```

### FastAPI server
```bash
python scripts/train_ci.py          # ensure models/ exists
uvicorn src.api:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

Example request:
```bash
curl -s http://localhost:8000/predict -H 'Content-Type: application/json' -d '{
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
}'
```

### Docker
```bash
docker build -t wine-quality-api .
docker run --rm -p 8000:8000 wine-quality-api
# GET  http://localhost:8000/health
# POST http://localhost:8000/predict
```

### Python predictor
```python
from src.predict import WineQualityPredictor

predictor = WineQualityPredictor()
sample_wine = {
    "fixed acidity": 7.4, "volatile acidity": 0.7, "citric acid": 0.0,
    "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0, "density": 0.9978, "pH": 3.51,
    "sulphates": 0.56, "alcohol": 9.4, "wine_type": "red",
}
print(predictor.predict(sample_wine))
```

---

## 🔬 Methodology Highlights

- **Leakage-free preprocessing:** encoding + scaling fitted **on the training split only**.
- **Honest model selection:** 8 regressors ranked by **5-fold CV RMSE** on train; test used once.
- **API + Docker** for serving the saved pipeline.
- **Deterministic:** `RANDOM_STATE = 42`.

---

## 📈 Expected Performance

| Model | CV RMSE (train) | Test RMSE | Test MAE | Test R² |
|-------|----------------|-----------|----------|---------|
| Random Forest ⭐ | 0.625 | 0.607 | 0.435 | 0.50 |
| XGBoost | 0.651 | ~0.63 | ~0.45 | ~0.45 |
| Gradient Boosting | 0.678 | ~0.66 | ~0.48 | ~0.42 |
| Ridge/Linear | 0.734 | ~0.73 | ~0.55 | ~0.27 |

---

## 📄 License

[MIT License](LICENSE).
