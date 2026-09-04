FROM python:3.12-slim AS lfs-fetch

RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*
RUN git lfs install

WORKDIR /src
COPY . .
RUN git lfs pull

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=lfs-fetch /src/crawl/ crawl/
COPY --from=lfs-fetch /src/parse/ parse/
COPY --from=lfs-fetch /src/indexer/ indexer/
COPY --from=lfs-fetch /src/app/ app/
COPY --from=lfs-fetch /src/evaluation/ evaluation/
COPY --from=lfs-fetch /src/conftest.py .
COPY --from=lfs-fetch /src/data/parsed/decisions.jsonl data/parsed/decisions.jsonl
COPY --from=lfs-fetch /src/data/index/index.pkl data/index/index.pkl

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
