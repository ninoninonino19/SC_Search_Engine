# Philippine Supreme Court decision search

A search engine built from scratch over Philippine Supreme Court decisions:
crawler, parser, tokenizer, inverted index, and BM25 ranking, with no external
search service in the retrieval path.

## The constraint that defines the project

No Elasticsearch, no Lucene, no Postgres full-text search (`to_tsvector`,
`tsquery`), no vector database, no embedding API in the core retrieval path.
Storing the index *in* SQLite or Postgres is fine — delegating the search to
them is not. The point of the project is the part those tools would do for you.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000. It serves fixture data until the crawler runs.

- `GET /` — search page
- `GET /search?q=...&page=1` — HTML results
- `GET /api/search?q=...&limit=10&offset=0` — JSON, for testing retrieval without HTML
- `GET /health` — active engine and document count

## Layout

```
app/          FastAPI layer. Thin by design.
  models.py       CaseDocument, SearchResult, SearchResponse
  engine.py       SearchEngine protocol — the seam
  fixture_engine.py  Linear scan over fixture data. The baseline to beat.
  main.py         Routes
crawl/        Stage 1 — fetch and cache raw HTML
parse/        Stage 2 — HTML to normalised CaseDocument records
indexer/      Stages 3-5 — tokenizer, inverted index, BM25
evaluation/   Stage 6 — judged queries, precision@10, MRR
data/         Crawled and derived artifacts (gitignored except fixtures)
```

`app/` depends on `app/engine.SearchEngine` and nothing else. When the real
engine is ready, change `build_engine()` in `app/main.py`; routes and templates
stay as they are.

## Build order

1. **Crawler.** Target is LawPhil (`lawphil.net/judjuris/`) — static
   server-rendered HTML, predictable year/month/case URL structure, no
   JavaScript. Cache raw HTML to `data/raw/` before parsing anything; you will
   re-parse many times and should never re-fetch. Rate-limit to ~1 req/sec, set
   a real User-Agent with a contact address, respect `robots.txt`, and make the
   run resumable.
2. **Parser.** HTML to JSONL: G.R. number, title, promulgation date, division,
   ponente, body, separate opinions. The messiest stage — markup drifts across
   decades, consolidated cases share a G.R. number, and some pages are
   malformed.
3. **Tokenizer.** Lowercase, strip punctuation, remove stopwords, Porter stem.
   Domain rules matter here: keep `G.R. No. 123456` as one token, keep the `v.`
   in case titles, handle section references like `Art. 315(2)(a)`.
4. **Inverted index.** Term to postings list of `(doc_id, term_freq, positions)`.
   Store positions from the start — retrofitting them for phrase search is
   painful. Start with a dict pickled to disk, move to SQLite when it stops
   fitting in memory comfortably. Delta-encoding doc IDs is a good stretch goal.
5. **Ranking.** TF-IDF first as a baseline, then BM25, so the improvement is
   measured rather than asserted.
6. **Evaluation.** ~30 hand-judged queries in `evaluation/queries.json`, scored
   on precision@10 and MRR. Build this *before* tuning, not after. It is what
   turns "I built a search engine" into "precision@10 went from 0.42 to 0.68."

## Scope

Start with decisions from 2010–2025 (roughly 10,000–15,000). Large enough that
linear scan is visibly too slow, small enough to rebuild the index in minutes
while iterating. Pre-1960 decisions carry Spanish-language passages and OCR
noise — problems worth having later, not first. Widening the range is a config
change.

## Open decisions

- **Separate opinions.** One document per case, or one per opinion? Affects
  document length normalisation in BM25 and what a "hit" means to a reader.
- **Index hosting.** A serverless function is a poor host for a few hundred MB
  of postings. A small VPS or a container on Fly/Render fits better.

## Data note

`data/fixture_cases.json` holds placeholder records so the app runs before the
crawl. Only the Angara entry has real citation metadata, and its body text is a
stub. Delete the whole file once the real index is wired in.
