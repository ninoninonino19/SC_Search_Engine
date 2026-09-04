# Philippine Supreme Court Decision Search

A full-text search engine built from scratch over 14,116 Philippine Supreme Court
decisions (2010-2026). No Elasticsearch, no Lucene, no Postgres full-text search,
no vector database, no embedding API anywhere in the retrieval path. Every
component - crawler, parser, tokenizer, inverted index, and BM25 ranking - is
hand-written in Python.

## Performance

| Metric | Score |
|--------|-------|
| Precision@10 | 0.879 |
| MRR | 0.983 |
| Median query latency | 238 ms |
| TF-IDF baseline P@10 | 0.434 |

Evaluated over 29 judged queries with a BM25 parameter sweep (k1=1.2, b=0.75).

## Running locally

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Build the corpus (takes hours, resumable)

```bash
# Set your contact email for polite crawling
set SC_SEARCH_CONTACT=you@example.com   # Windows
# export SC_SEARCH_CONTACT=you@example.com  # macOS/Linux

python -m crawl --start-year 2010 --end-year 2026
python -m parse
python -m indexer
```

### Start the server

```bash
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000.

## Endpoints

- `GET /` - search page with example queries
- `GET /search?q=...&page=1` - HTML results with highlighted snippets
- `GET /api/search?q=...&limit=10&offset=0` - JSON API
- `GET /about` - corpus stats and query syntax reference
- `GET /health` - engine status

## Query syntax

| Syntax | Example |
|--------|---------|
| Terms (AND) | `grave abuse of discretion` |
| OR | `mandamus OR certiorari` |
| Phrase | `"beyond reasonable doubt"` |
| Citation | `G.R. No. 192393` or `gr:192393` |
| Ponente | `ponente:Leonen` |
| Year | `year:2019` |
| Division | `division:"En Banc"` |
| Combined | `ponente:Caguioa year:2019 certiorari` |

## Project layout

```
app/          FastAPI web layer
  main.py         Routes
  engine.py       SearchEngine protocol (the seam between web and retrieval)
  models.py       CaseDocument, SearchResult, SearchResponse
  templates/      Jinja2 templates (search, about)
  static/         CSS (light/dark mode)
crawl/        Stage 1 - fetch and cache raw HTML from LawPhil
parse/        Stage 2 - HTML to normalised CaseDocument JSONL
indexer/      Stages 3-5
  tokenizer.py    Domain-aware tokenizer (citations, sections, Latin terms)
  porter.py       Porter stemmer
  codec.py        Delta + varint compression for postings
  index.py        Positional inverted index (build, save, load, query)
  ranking.py      BM25 and TF-IDF rankers
  engine.py       Two-stage retrieval (boolean + ranking) with snippet highlighting
  query.py        Query parser (terms, phrases, OR, field filters)
evaluation/   Stage 6 - 30 judged queries, P@10, MRR, parameter sweep
tests/        Unit tests (68 passing)
```

## Deployment

Dockerfile and fly.toml are included for Fly.io deployment.

```bash
fly auth login
fly launch
fly deploy
```

## The constraint

No Elasticsearch, no Lucene, no Postgres full-text search (`to_tsvector`,
`tsquery`), no vector database, no embedding API in the core retrieval path.
Storing the index in SQLite or Postgres is fine - delegating the search to them
is not. The point of the project is building the part those tools would do for you.

## Corpus

14,116 decisions from 2010-2026, crawled from LawPhil. 156,151 unique terms,
32.5M tokens indexed. 74.7 MiB compressed index, 446.1 MiB corpus. See
`data/parsed/report.txt` for the full corpus report.
