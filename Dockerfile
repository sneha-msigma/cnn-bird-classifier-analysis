# BASE IMAGE: Python 3.9
FROM python:3.9-slim

# Working directory within the container
WORKDIR /app

# Install system dependencies for optimization artifact
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY requirements.txt .

# Install Python correlations
RUN pip install --no-cache-dir -r requirements.txt

# Copy logic artifacts
COPY . .

# Environment setup for external storage
# These paths must be mounted to the host system
ENV DATA_DIR=/app/external_data
ENV OUTPUT_DIR=/app/outputs

# Create necessary directories
RUN mkdir -p /app/external_data /app/outputs

# Execution entry point
CMD ["python", "main.py"]
