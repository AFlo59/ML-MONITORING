
# Load reference and current data
def load_data(reference_path, current_path):
    reference_data = pd.read_csv(reference_path)
    current_data = pd.read_csv(current_path)
    return reference_data, current_data

# Generate Evidently report
def generate_report(reference_data, current_data, output_path):
    dashboard = Dashboard(tabs=[DataDriftTab(), RegressionPerformanceTab()])
    dashboard.calculate(reference_data, current_data)
    dashboard.save(output_path)
    print(f"Evidently report saved to {output_path}")

if __name__ == "__main__":
    # Filepaths
    REFERENCE_DATA_PATH = "data/reference.csv"
    CURRENT_DATA_PATH = "data/current.csv"
    OUTPUT_REPORT_PATH = "reports/evidently_report.html"

    # Generate report
    reference_data, current_data = load_data(REFERENCE_DATA_PATH, CURRENT_DATA_PATH)
    generate_report(reference_data, current_data, OUTPUT_REPORT_PATH)