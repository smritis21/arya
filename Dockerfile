FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.1"
ENV HF_TOKEN=""

CMD ["python", "inference.py"]
