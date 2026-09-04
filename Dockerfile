FROM python:3.12-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY crawl/ crawl/
COPY parse/ parse/
COPY indexer/ indexer/
COPY app/ app/
COPY evaluation/ evaluation/
COPY conftest.py .

RUN mkdir -p data/parsed data/index

RUN curl -sL -o data/index/index.pkl \
    "$(curl -s -X POST https://github.com/ninoninonino19/SC_Search_Engine.git/info/lfs/objects/batch \
       -H 'Content-Type: application/json' \
       -H 'Accept: application/vnd.git-lfs+json' \
       -d '{"operation":"download","transfers":["basic"],"objects":[{"oid":"f2ed7b1fe6e02e3bb32c31e2dccc8c364a559d3a99417193037739a330140411","size":78374213}]}' \
       | python3 -c 'import sys,json; print(json.load(sys.stdin)["objects"][0]["actions"]["download"]["href"])')"

RUN curl -sL -o data/parsed/decisions.jsonl \
    "$(curl -s -X POST https://github.com/ninoninonino19/SC_Search_Engine.git/info/lfs/objects/batch \
       -H 'Content-Type: application/json' \
       -H 'Accept: application/vnd.git-lfs+json' \
       -d '{"operation":"download","transfers":["basic"],"objects":[{"oid":"57b4a93f2f311096569eee821342295ea5f30ac426893a0825cb3f779f2c95d8","size":467812760}]}' \
       | python3 -c 'import sys,json; print(json.load(sys.stdin)["objects"][0]["actions"]["download"]["href"])')"

RUN python3 -c "import pickle; pickle.load(open('data/index/index.pkl','rb')); print('index.pkl verified OK')"

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
