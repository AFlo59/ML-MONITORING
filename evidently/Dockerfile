# Dockerfile for Evidently AI

FROM python:3.9-slim

# Define working directory
WORKDIR /evidently

# Copy requirements and install dependencies
COPY requirements.txt /evidently/requirements.txt
RUN pip install --no-cache-dir -r /evidently/requirements.txt

# Copy configuration and necessary scripts
COPY . /evidently

# Set entrypoint for running Evidently
CMD ["python", "monitoring_script.py"]
