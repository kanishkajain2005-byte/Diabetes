

# Use a lightweight official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
# Use --no-cache-dir to keep the image size down
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts and the application code
COPY model/ ./model/
COPY prediction_api.py .
RUN chmod -R 755 /app/model
# Expose the port where the application will run
EXPOSE 8000

# Production command using Gunicorn and Uvicorn workers
# Gunicorn handles process management; Uvicorn handles async processing.
# Adjust --workers based on your deployment environment (e.g., 2 * CPU_cores + 1)
CMD ["gunicorn", "prediction_api:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]