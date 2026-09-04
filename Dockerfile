FROM python:3.12-slim AS base

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY crawl/ crawl/
COPY parse/ parse/
COPY indexer/ indexer/
COPY app/ app/
COPY evaluation/ evaluation/
COPY conftest.py .

COPY data/parsed/decisions.jsonl data/parsed/decisions.jsonl
COPY data/index/index.pkl data/index/index.pkl

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
