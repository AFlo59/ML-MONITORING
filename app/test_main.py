from fastapi.testclient import TestClient
from main import app

# Initialize TestClient
client = TestClient(app)

def test_health_check():
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_homepage():
    """Test the homepage / endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>API Home</title>" in response.text
    assert "<a href=\"/docs\">API Documentation</a>" in response.text

def test_predict_valid_input():
    """Test the /predict endpoint with valid input."""
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_invalid_input():
    """Test the /predict endpoint with invalid input."""
    payload = {"features": "invalid"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity for validation error

def test_predict_empty_input():
    """Test the /predict endpoint with empty input."""
    payload = {"features": []}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity for validation error