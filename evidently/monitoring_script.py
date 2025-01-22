import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

# Paths from configuration
reference_dataset_path = "path/to/reference.csv"
current_dataset_path = "path/to/current.csv"
output_directory = "/reports"

# Load datasets
reference_data = pd.read_csv(reference_dataset_path)
current_data = pd.read_csv(current_dataset_path)

# Create report
report = Report(metrics=[
    DataDriftPreset(),
    TargetDriftPreset()
])

report.run(reference_data=reference_data, current_data=current_data)

# Save report
os.makedirs(output_directory, exist_ok=True)
report.save_html(os.path.join(output_directory, "data_drift_report.html"))
