FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENV API_BASE_URL="https://router.huggingface.co/v1" \
    MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"

# cache bust: 2026-04-26
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "--log-level", "debug", "--capture-output", "server:app"]
