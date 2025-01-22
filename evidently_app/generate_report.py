# evidently_app/generate_report.py
import pandas as pd
from sklearn.datasets import load_iris
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset, ClassificationPreset

def generate_evidently_report():
    # Chargez vos données de référence
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    # Pour la démo, on fait comme si X.head(50) = reference, X.tail(50) = current
    reference_data = X.head(50).copy()
    reference_data['target'] = y[:50]
    
    current_data = X.tail(50).copy()
    current_data['target'] = y[-50:]
    
    # Créer un rapport combinant différents "Presets"
    report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset(),
    ])

    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html("evidently_report.html")

if __name__ == "__main__":
    generate_evidently_report()
