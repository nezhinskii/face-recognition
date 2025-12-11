FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 3000

CMD ["bentoml", "serve", "service.py:FaceDetectionService", "--host", "0.0.0.0", "--port", "3000"]