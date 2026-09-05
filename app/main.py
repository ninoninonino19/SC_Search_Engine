"""FastAPI entry point.

Deliberately thin. The route handlers parse HTTP arguments, call `engine.search()`,
and render. Anything resembling retrieval logic that shows up in this file belongs
behind the `SearchEngine` protocol instead.
"""

from __future__ import annotations

import csv
import io
import os
import re
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.engine import SearchEngine
from indexer.engine import IndexEngine

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR.parent / "data" / "index" / "index.pkl"
PARSED_PATH = BASE_DIR.parent / "data" / "parsed" / "decisions.jsonl"
MAX_LIMIT = 50
CACHE_MAX = 256
CACHE_TTL = 300  # seconds

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class SearchCache:
    """Thread-safe LRU cache with TTL for search results."""

    def __init__(self, maxsize: int = CACHE_MAX, ttl: float = CACHE_TTL) -> None:
        self._store: OrderedDict[str, tuple[float, object]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            ts, value = entry
            if time.monotonic() - ts > self._ttl:
                del self._store[key]
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: str, value: object) -> None:
        with self._lock:
            self._store[key] = (time.monotonic(), value)
            self._store.move_to_end(key)
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)


def build_engine() -> SearchEngine:
    """Single place where the active implementation is chosen.

    Ranking parameters come from the environment so a deployment can be
    re-tuned, and so `evaluation/` can start the same app under a different
    configuration without editing code:

        SC_SEARCH_RANKER=tfidf  SC_SEARCH_K1=1.2  SC_SEARCH_B=0.75
    """
    if not INDEX_PATH.exists():
        raise RuntimeError(
            f"no index at {INDEX_PATH}. Build one first:\n"
            "  python -m crawl   # fetch decisions (hours; resumable)\n"
            "  python -m parse   # cached HTML -> data/parsed/decisions.jsonl\n"
            "  python -m indexer # -> data/index/index.pkl"
        )
    engine = IndexEngine.load(
        INDEX_PATH,
        ranker=os.environ.get("SC_SEARCH_RANKER", "bm25"),
        k1=float(os.environ.get("SC_SEARCH_K1", "1.2")),
        b=float(os.environ.get("SC_SEARCH_B", "0.75")),
    )
    if PARSED_PATH.exists():
        engine.index.source_path = PARSED_PATH
    return engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = build_engine()
    app.state.cache = SearchCache()
    app.state.request_count = 0
    app.state.started = time.monotonic()
    _build_metadata_lists(app)
    yield
    app.state.engine = None


def _build_metadata_lists(app: FastAPI) -> None:
    """Extract unique ponentes, divisions, and year range from the index."""
    engine = app.state.engine
    index = getattr(engine, "index", None)
    if index is None:
        app.state.ponentes = []
        app.state.divisions = []
        app.state.year_range = (2010, 2026)
        return
    ponentes: set[str] = set()
    divisions: set[str] = set()
    years: list[int] = []
    for doc in index.docs:
        if doc.ponente:
            ponentes.add(doc.ponente)
        if doc.division:
            divisions.add(doc.division)
        if doc.year:
            years.append(doc.year)
    app.state.ponentes = sorted(ponentes)
    app.state.divisions = sorted(divisions)
    app.state.year_range = (min(years), max(years)) if years else (2010, 2026)


app = FastAPI(
    title="Philippine Supreme Court decision search",
    description="A search engine built from scratch over Supreme Court decisions.",
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=500)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.middleware("http")
async def count_requests(request: Request, call_next):
    request.app.state.request_count += 1
    response = await call_next(request)
    return response


@app.get("/health")
def health(request: Request) -> JSONResponse:
    engine: SearchEngine = request.app.state.engine
    cache: SearchCache = request.app.state.cache
    uptime = time.monotonic() - request.app.state.started
    return JSONResponse({
        "status": "ok",
        "engine": engine.name,
        "documents": engine.doc_count,
        "requests": request.app.state.request_count,
        "uptime_seconds": round(uptime),
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
    })


@app.get("/about", response_class=HTMLResponse)
def about(request: Request):
    engine: SearchEngine = request.app.state.engine
    index = getattr(engine, "index", None)
    stats = {
        "doc_count": f"{engine.doc_count:,}",
        "engine_name": engine.name,
    }
    if index is not None:
        index_bytes = INDEX_PATH.stat().st_size if INDEX_PATH.exists() else 0
        jsonl_bytes = PARSED_PATH.stat().st_size if PARSED_PATH.exists() else 0
        stats.update({
            "vocabulary": f"{index.vocabulary_size:,}",
            "total_tokens": f"{index.total_tokens:,}",
            "avg_doc_length": f"{index.avgdl:,.0f}",
            "index_size_mb": f"{index_bytes / 1_048_576:.1f}",
            "corpus_size_mb": f"{jsonl_bytes / 1_048_576:.1f}",
            "build_seconds": f"{index.build_seconds:.1f}",
        })
    return templates.TemplateResponse(
        request=request,
        name="about.html",
        context={"stats": stats, "engine": engine},
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    engine: SearchEngine = request.app.state.engine
    return templates.TemplateResponse(
        request=request,
        name="search.html",
        context={
            "response": None,
            "engine": engine,
            "query": "",
            "ponentes": request.app.state.ponentes,
            "divisions": request.app.state.divisions,
            "year_range": request.app.state.year_range,
        },
    )


@app.get("/search", response_class=HTMLResponse)
def search(
    request: Request,
    q: str = Query("", description="Search query"),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=MAX_LIMIT),
):
    started = time.perf_counter()
    engine: SearchEngine = request.app.state.engine
    cache: SearchCache = request.app.state.cache
    cache_key = f"{q}|{limit}|{page}"
    cached = cache.get(cache_key)
    if cached is not None:
        result, suggestion, related, total_ms = cached
    else:
        result = engine.search(q, limit=limit, offset=(page - 1) * limit)
        suggestion = None
        related = []
        if result.total_hits == 0 and q.strip():
            suggestion = _spell_suggest(engine, q)
        elif result.total_hits > 0:
            related = _related_searches(engine, result, q)
        total_ms = (time.perf_counter() - started) * 1000
        cache.put(cache_key, (result, suggestion, related, total_ms))

    return templates.TemplateResponse(
        request=request,
        name="search.html",
        context={
            "response": result,
            "engine": engine,
            "query": q,
            "page": page,
            "suggestion": suggestion,
            "related": related,
            "total_ms": total_ms,
            "ponentes": request.app.state.ponentes,
            "divisions": request.app.state.divisions,
            "year_range": request.app.state.year_range,
        },
    )


@app.get("/api/search")
def api_search(
    request: Request,
    q: str = Query("", description="Search query"),
    limit: int = Query(10, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
) -> JSONResponse:
    """JSON form of the same query, so the retrieval layer is testable without HTML."""
    engine: SearchEngine = request.app.state.engine
    cache: SearchCache = request.app.state.cache
    page = (offset // limit) + 1 if limit else 1
    cache_key = f"api|{q}|{limit}|{page}"
    result = cache.get(cache_key)
    if result is None:
        result = engine.search(q, limit=limit, offset=offset)
        cache.put(cache_key, result)
    payload = asdict(result)
    payload["has_more"] = result.has_more
    for item, original in zip(payload["results"], result.results):
        item["promulgated"] = original.promulgated.isoformat() if original.promulgated else None
    return JSONResponse(payload)


@app.get("/api/suggest")
def api_suggest(
    request: Request,
    q: str = Query("", description="Prefix to complete"),
    limit: int = Query(8, ge=1, le=20),
) -> JSONResponse:
    """Autocomplete suggestions from the vocabulary."""
    engine: SearchEngine = request.app.state.engine
    index = getattr(engine, "index", None)
    if not index or len(q.strip()) < 2:
        return JSONResponse([])

    from indexer.tokenizer import normalize
    prefix = normalize(q.strip())
    suggestions = _prefix_suggest(index, prefix, limit)
    return JSONResponse(suggestions)


@app.get("/decision/{doc_id}", response_class=HTMLResponse)
def decision_detail(request: Request, doc_id: int):
    engine: SearchEngine = request.app.state.engine
    index = getattr(engine, "index", None)
    if index is None or doc_id < 0 or doc_id >= index.doc_count:
        return HTMLResponse("<h1>Decision not found</h1>", status_code=404)

    import json
    meta = index.docs[doc_id]
    with index.source_path.open("rb") as handle:
        handle.seek(meta.offset)
        record = json.loads(handle.readline())

    return templates.TemplateResponse(
        request=request,
        name="decision.html",
        context={"record": record, "meta": meta, "engine": engine},
    )


@app.get("/export")
def export_csv(
    request: Request,
    q: str = Query("", description="Search query"),
    limit: int = Query(100, ge=1, le=500),
):
    engine: SearchEngine = request.app.state.engine
    result = engine.search(q, limit=limit, offset=0)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Rank", "G.R. Number", "Title", "Date", "Division", "Ponente", "Score", "Source URL"])
    for i, hit in enumerate(result.results, 1):
        writer.writerow([
            i,
            hit.gr_number,
            hit.title,
            hit.promulgated.isoformat() if hit.promulgated else "",
            hit.division or "",
            hit.ponente or "",
            f"{hit.score:.4f}",
            hit.source_url,
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="sc-search-results.csv"'},
    )


# -- helpers ----------------------------------------------------------------

def _prefix_suggest(index, prefix: str, limit: int) -> list[str]:
    """Find vocabulary terms starting with prefix, ranked by document frequency."""
    from indexer.tokenizer import STOPWORDS
    matches: list[tuple[str, int]] = []
    max_len = max(len(prefix) + 12, 20)
    for term in index.postings:
        if not term.startswith(prefix):
            continue
        if term in STOPWORDS or ":" in term:
            continue
        if len(term) > max_len:
            continue
        if len(term) < 3:
            continue
        matches.append((term, len(index.postings[term])))
    matches.sort(key=lambda x: (-x[1], len(x[0])))
    return [term for term, _ in matches[:limit]]


def _spell_suggest(engine, query: str) -> str | None:
    """Suggest a corrected query when 0 results by trying edit-distance-1 variants."""
    from indexer.tokenizer import normalize, tokenize
    index = getattr(engine, "index", None)
    if index is None:
        return None

    tokens = tokenize(normalize(query))
    if not tokens:
        return None

    corrected_parts = []
    changed = False
    raw_words = query.strip().split()

    for word in raw_words:
        if ":" in word or word.upper() == "OR" or word.startswith('"'):
            corrected_parts.append(word)
            continue

        word_tokens = tokenize(normalize(word))
        if not word_tokens:
            corrected_parts.append(word)
            continue

        stemmed = word_tokens[0].text
        if stemmed in index.postings:
            corrected_parts.append(word)
            continue

        best, best_df = None, 0
        edits = _edits1(normalize(word))
        for candidate in edits:
            candidate_tokens = tokenize(candidate)
            if not candidate_tokens:
                continue
            ct = candidate_tokens[0].text
            if ct in index.postings:
                df = len(index.postings[ct])
                if df > best_df:
                    best, best_df = candidate, df

        if not best and len(word) >= 4:
            checked = 0
            for e1 in edits:
                if checked > 5000:
                    break
                for candidate in _edits1(e1):
                    checked += 1
                    if checked > 5000:
                        break
                    candidate_tokens = tokenize(candidate)
                    if not candidate_tokens:
                        continue
                    ct = candidate_tokens[0].text
                    if ct in index.postings:
                        df = len(index.postings[ct])
                        if df > best_df:
                            best, best_df = candidate, df
                if best_df > 100:
                    break

        if best and best_df > 5:
            corrected_parts.append(best)
            changed = True
        else:
            corrected_parts.append(word)

    return " ".join(corrected_parts) if changed else None


def _edits1(word: str) -> set[str]:
    """All strings that are one edit away from the input."""
    letters = "abcdefghijklmnopqrstuvwxyzñ"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def _related_searches(engine, result, query: str) -> list[str]:
    """Extract distinctive terms from top results to suggest as related searches."""
    import math
    from indexer.tokenizer import normalize, tokenize, scan, STOPWORDS, PROTECTED
    from indexer.porter import stem as porter_stem
    index = getattr(engine, "index", None)
    if index is None or not result.results:
        return []

    n_docs = index.doc_count
    upper = int(n_docs * 0.15)
    max_stems_per_doc = 150

    query_terms = {t.text for t in tokenize(normalize(query))}
    term_scores: dict[str, float] = {}
    surface_forms: dict[str, str] = {}
    _HAS_DIGIT = re.compile(r"\d")

    for hit in result.results[:5]:
        body = index.body(hit.doc_id)
        seen_stems: set[str] = set()
        for tok in scan(body):
            if len(seen_stems) >= max_stems_per_doc:
                break
            word = tok.text
            if ":" in word or word == "vs" or word in STOPWORDS or len(word) < 4:
                continue
            if _HAS_DIGIT.search(word):
                continue
            stemmed = word if word in PROTECTED else porter_stem(word)
            if stemmed in query_terms or stemmed in seen_stems:
                continue
            seen_stems.add(stemmed)
            df = index.doc_frequency(stemmed)
            if 10 < df < upper:
                idf = math.log(n_docs / (df + 1))
                term_scores[stemmed] = term_scores.get(stemmed, 0) + idf
                if stemmed not in surface_forms or len(word) > len(surface_forms[stemmed]):
                    surface_forms[stemmed] = word

    ranked = sorted(term_scores.items(), key=lambda x: -x[1])
    return [surface_forms[stem] for stem, _ in ranked[:6] if stem in surface_forms]
