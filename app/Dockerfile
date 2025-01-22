# Dockerfile for FastAPI app

FROM python:3.9-slim

# Define the working directory
WORKDIR /app

# Copy the requirements and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the app code
COPY . /app

# Expose the port used by the FastAPI application
EXPOSE 8000

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
