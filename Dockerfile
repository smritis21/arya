FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENV API_BASE_URL="https://router.huggingface.co/v1" \
    MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct" \
    HF_TOKEN=""

CMD ["python", "server.py"]
