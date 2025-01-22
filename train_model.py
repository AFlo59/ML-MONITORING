import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import pickle

# Charger le dataset Iris
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Diviser les données en jeux d'entraînement et de test
X = data.drop(columns=["target"])
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Entraîner le modèle
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Évaluer le modèle
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Précision du modèle : {accuracy:.2f}")

# Sauvegarder le modèle
with open("model_training/model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Modèle sauvegardé avec succès.")

# Générer le fichier reference.csv
X_train["target"] = y_train
X_train.to_csv("evidently/data/reference.csv", index=False)
print("reference.csv généré avec succès.")

# Générer le fichier current.csv à partir du modèle
X_test["target"] = model.predict(X_test)
X_test.to_csv("evidently/data/current.csv", index=False)
print("current.csv généré avec succès.")
