import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

# Load dataset
def load_data(filepath):
    """Load the dataset from a given filepath."""
    data = pd.read_csv(filepath)
    return data

# Preprocess data
def preprocess_data(data):
    """Preprocess the dataset by handling missing values and encoding."""
    data = data.dropna()
    X = data.drop('target', axis=1)
    y = data['target']
    return X, y

# Train model
def train_model(X, y):
    """Train a Random Forest model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return model

# Save model
def save_model(model, output_path):
    """Save the trained model to a file."""
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    # Filepath to dataset
    DATA_PATH = "data/dataset.csv"
    OUTPUT_MODEL_PATH = "models/random_forest_model.pkl"

    # Execute pipeline
    data = load_data(DATA_PATH)
    X, y = preprocess_data(data)
    model = train_model(X, y)
    save_model(model, OUTPUT_MODEL_PATH)