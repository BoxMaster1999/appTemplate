FROM python:3.10-slim as base

WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
COPY . .

RUN mkdir -p /app/models

# Образ для разработки
FROM base as dev
CMD uvicorn app.main:app --host=0.0.0.0 --port=8080 --reload

# Образ для прода
FROM base as prod
CMD gunicorn app.main:app  --timeout 1000 --bind 0.0.0.0:8080
