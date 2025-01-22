from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
import pickle
import pandas as pd
import logging

# Configurer le logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Charger le modèle ML
MODEL_PATH = "model_training/model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Modèle chargé avec succès depuis %s", MODEL_PATH)
except Exception as e:
    logger.error("Erreur lors du chargement du modèle : %s", str(e))
    model = None

# Définir l'application FastAPI
app = FastAPI()

# Ajouter Prometheus Instrumentator
Instrumentator().instrument(app).expose(app)

# Définir le modèle de données d'entrée
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

# Endpoint de prédiction
@app.post("/predict")
async def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Le modèle n'est pas disponible")

    # Convertir les données en DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Log des données reçues
    logger.info("Données reçues : %s", input_df.to_dict(orient="records"))

    try:
        # Faire la prédiction
        prediction = model.predict(input_df)
        logger.info("Prédiction réalisée avec succès")
        return {"prediction": prediction.tolist()}
    except Exception as e:
        logger.error("Erreur lors de la prédiction : %s", str(e))
        raise HTTPException(status_code=500, detail="Erreur lors de la prédiction")

@app.get("/health")
def health_check():
    return {"status": "ok"}