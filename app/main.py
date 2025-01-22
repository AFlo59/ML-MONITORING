from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
import joblib
import numpy as np
from typing import List

# Initialize FastAPI
app = FastAPI(title="Prediction API", version="1.0.0")

# Load the trained model
try:
    model = joblib.load("model_training/model.pkl")
except FileNotFoundError:
    model = None

# Define the input data model
class PredictionInput(BaseModel):
    features: List[float]

# Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

@app.get("/")
def read_root():
    return {"message": "Prediction API is up and running!"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not available")

    try:
        input_array = np.array(input_data.features).reshape(1, -1)
        prediction = model.predict(input_array)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}
