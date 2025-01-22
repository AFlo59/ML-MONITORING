# ML Monitoring Project

## Description
This project demonstrates how to monitor a machine learning API using FastAPI, Evidently AI, Prometheus, and Grafana. The system is designed to track model performance, detect data drifts, and visualize key metrics.

## Project Structure


## Instructions

### 1. Train the Model
Navigate to the `model_training/` directory and run:
```bash
python3 train_model.py
```

### 2. Start the Services
Run the following command from the project root:
```bash
docker-compose up --build
```

### 3. Access the Services

#### FastAPI: http://localhost:8000
#### Prometheus: http://localhost:9090
#### Grafana: http://localhost:3000

### 4. Testing
Run unit tests for the API:
```bash
pytest app/test_main.py
```

---

### Prochaines étapes :
Tous les fichiers nécessaires au projet ont été listés avec leur contenu. Si vous avez des questions, des ajustements ou des tests supplémentaires à effectuer, n'hésitez pas à demander !
