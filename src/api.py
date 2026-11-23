"""FastAPI service for wine quality prediction."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import WineQualityPredictor

app = FastAPI(
    title="Wine Quality Prediction API",
    description="Predict wine quality (3–9) from physicochemical features.",
    version="1.0.0",
)

_predictor: WineQualityPredictor | None = None


def get_predictor() -> WineQualityPredictor:
    global _predictor
    if _predictor is None:
        try:
            _predictor = WineQualityPredictor()
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=503,
                detail=str(e),
            ) from e
    return _predictor


class WineFeatures(BaseModel):
    fixed_acidity: float = Field(..., alias="fixed acidity", examples=[7.4])
    volatile_acidity: float = Field(..., alias="volatile acidity", examples=[0.7])
    citric_acid: float = Field(..., alias="citric acid", examples=[0.0])
    residual_sugar: float = Field(..., alias="residual sugar", examples=[1.9])
    chlorides: float = Field(..., examples=[0.076])
    free_sulfur_dioxide: float = Field(..., alias="free sulfur dioxide", examples=[11.0])
    total_sulfur_dioxide: float = Field(..., alias="total sulfur dioxide", examples=[34.0])
    density: float = Field(..., examples=[0.9978])
    pH: float = Field(..., examples=[3.51])
    sulphates: float = Field(..., examples=[0.56])
    alcohol: float = Field(..., examples=[9.4])
    wine_type: Literal["red", "white"] = Field(..., examples=["red"])

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    predicted_quality: float
    quality_category: str
    quality_range: str


@app.get("/health")
def health():
    model_ready = True
    try:
        get_predictor()
    except HTTPException:
        model_ready = False
    return {"status": "ok" if model_ready else "model_missing", "model_ready": model_ready}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: WineFeatures):
    predictor = get_predictor()
    payload = {
        "fixed acidity": features.fixed_acidity,
        "volatile acidity": features.volatile_acidity,
        "citric acid": features.citric_acid,
        "residual sugar": features.residual_sugar,
        "chlorides": features.chlorides,
        "free sulfur dioxide": features.free_sulfur_dioxide,
        "total sulfur dioxide": features.total_sulfur_dioxide,
        "density": features.density,
        "pH": features.pH,
        "sulphates": features.sulphates,
        "alcohol": features.alcohol,
        "wine_type": features.wine_type,
    }
    try:
        return predictor.predict(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/predict/batch")
def predict_batch(samples: list[WineFeatures]):
    predictor = get_predictor()
    results = []
    for features in samples:
        payload = {
            "fixed acidity": features.fixed_acidity,
            "volatile acidity": features.volatile_acidity,
            "citric acid": features.citric_acid,
            "residual sugar": features.residual_sugar,
            "chlorides": features.chlorides,
            "free sulfur dioxide": features.free_sulfur_dioxide,
            "total sulfur dioxide": features.total_sulfur_dioxide,
            "density": features.density,
            "pH": features.pH,
            "sulphates": features.sulphates,
            "alcohol": features.alcohol,
            "wine_type": features.wine_type,
        }
        results.append(predictor.predict(payload))
    return {"predictions": results}
