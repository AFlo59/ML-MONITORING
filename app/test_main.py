from fastapi.testclient import TestClient
import unittest
from main import app

# Créer un client de test
client = TestClient(app)

class TestAPI(unittest.TestCase):

    def test_health_endpoint(self):
        """Tester l'endpoint /health"""
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_predict_endpoint_success(self):
        """Tester l'endpoint /predict avec des données valides"""
        payload = {
            "feature1": 1.0,
            "feature2": 2.0,
            "feature3": 3.0,
            "feature4": 4.0
        }
        response = client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json())

    def test_predict_endpoint_missing_feature(self):
        """Tester l'endpoint /predict avec une feature manquante"""
        payload = {
            "feature1": 1.0,
            "feature2": 2.0,
            "feature3": 3.0
        }
        response = client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)  # Erreur de validation des données

    def test_predict_endpoint_invalid_data(self):
        """Tester l'endpoint /predict avec des données invalides"""
        payload = {
            "feature1": "invalid",
            "feature2": 2.0,
            "feature3": 3.0,
            "feature4": 4.0
        }
        response = client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)  # Erreur de validation des données

if __name__ == "__main__":
    unittest.main()
