FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENV API_BASE_URL="https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta/v1" \
    MODEL_NAME="HuggingFaceH4/zephyr-7b-beta"

# cache bust: 2026-04-26
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "--log-level", "debug", "--capture-output", "server:app"]
