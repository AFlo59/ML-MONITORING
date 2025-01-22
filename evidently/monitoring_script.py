import pandas as pd
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, TargetDriftTab, RegressionPerformanceTab

# Configuration des chemins
REFERENCE_DATA_PATH = "reference.csv"
CURRENT_DATA_PATH = "current.csv"
OUTPUT_REPORT_PATH = "evidently_report.html"

# Charger les jeux de données
try:
    reference_data = pd.read_csv(REFERENCE_DATA_PATH)
    current_data = pd.read_csv(CURRENT_DATA_PATH)
    print("Données chargées avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement des données : {e}")
    exit(1)

# Créer un tableau de bord Evidently
dashboard = Dashboard([
    DataDriftTab(),
    TargetDriftTab(),
    RegressionPerformanceTab()
])

# Générer le rapport Evidently
def generate_report():
    try:
        dashboard.calculate(reference_data, current_data)
        dashboard.save(OUTPUT_REPORT_PATH)
        print(f"Rapport généré avec succès : {OUTPUT_REPORT_PATH}")
    except Exception as e:
        print(f"Erreur lors de la génération du rapport : {e}")

if __name__ == "__main__":
    generate_report()