# train_model.py
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_and_save_model():
    # Charger le dataset
    data = load_iris()
    X = data.data
    y = data.target
    
    # Séparer en entraînement et test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Entraîner un modèle simple
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Évaluer rapidement (optionnel, pour information)
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy on test: {accuracy:.2f}")
    
    # Sauvegarder le modèle
    joblib.dump(model, "model.pkl")

if __name__ == "__main__":
    train_and_save_model()
