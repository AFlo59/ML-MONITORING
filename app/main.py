from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
from prometheus_fastapi_instrumentator import Instrumentator
import logging

# Initialize FastAPI app
app = FastAPI()

# Initialize Prometheus Instrumentator
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load the trained model
MODEL_PATH = "models/random_forest_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Input data schema
class PredictionRequest(BaseModel):
    features: list

# Home endpoint
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>API Home</title>
        </head>
        <body>
            <h1>Welcome to the FastAPI ML Service</h1>
            <ul>
                <li><a href="/docs">API Documentation</a></li>
                <li><a href="/health">Health Check</a></li>
                <li><a href="/predict">Predict Endpoint (POST only)</a></li>
            </ul>
        </body>
    </html>
    """

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "healthy"}

# Prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features).tolist()
        logger.info(f"Prediction made: {prediction}")
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# Middleware for logging requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response