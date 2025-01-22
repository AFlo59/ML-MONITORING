FROM python:3.9-slim

# Installer les dépendances système
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /usr/src/app

# Copier les fichiers nécessaires
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Copier le fichier ou dossier contenant le modèle
COPY model_training /usr/src/app/model_training

# Exposer le port de l'application
EXPOSE 8000

# Commande pour démarrer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

