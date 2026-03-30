FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir flask flask-cors gunicorn openai pydantic requests pyyaml python-dotenv

COPY . .

EXPOSE 7860

ENV API_BASE_URL="https://router.huggingface.co/v1" \
    MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct" \
    HF_TOKEN=""

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "server:app"]
