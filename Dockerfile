# Use Python latest stable slim image for a small footprint
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker build cache
COPY requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose port 5000 for the Flask application
EXPOSE 5000

<<<<<<< HEAD
# Set default Flask environment variable
ENV FLASK_APP=app.py

# Run the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
=======
# Environment variables (override at runtime)
ENV API_BASE_URL="https://router.huggingface.co/v1" \
    MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct" \
    HF_TOKEN=""

# Run via gunicorn (production-grade)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "server:app"]
>>>>>>> round1-submission
