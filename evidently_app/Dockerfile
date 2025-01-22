# evidently_app/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Installer Evidently
RUN pip install evidently==0.3.2 pandas==1.5.3 scikit-learn==1.2.2

COPY . /app

# Exemple : générer un rapport directement au démarrage (pour la démo)
CMD ["python", "generate_report.py"]
