# api/main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np

# Pour l'instrumentation Prometheus
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
model = joblib.load("model.pkl")  # Charger le modèle entraîné

# Modèle de données d’entrée (adapté au dataset Iris)
class IrisFeatures(BaseModel):
    data: List[float]

# Initialiser l'instrumentateur Prometheus
Instrumentator().instrument(app).expose(app)

@app.get("/")
def home():
    return {"message": "API de prédiction Iris en ligne"}

@app.post("/predict")
def predict_iris(features: IrisFeatures):
    # Récupérer les features depuis la requête
    X = np.array(features.data).reshape(1, -1)
    # Faire la prédiction
    prediction = model.predict(X)
    return {"prediction": int(prediction[0])}
