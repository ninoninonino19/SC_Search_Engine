"""FastAPI entry point.

Deliberately thin. The route handlers parse HTTP arguments, call `engine.search()`,
and render. Anything resembling retrieval logic that shows up in this file belongs
behind the `SearchEngine` protocol instead.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.engine import SearchEngine
from indexer.engine import IndexEngine

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR.parent / "data" / "index" / "index.pkl"
PARSED_PATH = BASE_DIR.parent / "data" / "parsed" / "decisions.jsonl"
MAX_LIMIT = 50

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


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
    # Loading at startup rather than per-request matters more than it looks:
    # the real index will be hundreds of megabytes and must be read once.
    app.state.engine = build_engine()
    yield
    app.state.engine = None


app = FastAPI(
    title="Philippine Supreme Court decision search",
    description="A search engine built from scratch over Supreme Court decisions.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/health")
def health(request: Request) -> JSONResponse:
    engine: SearchEngine = request.app.state.engine
    return JSONResponse({"status": "ok", "engine": engine.name, "documents": engine.doc_count})


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
        context={"response": None, "engine": engine, "query": ""},
    )


@app.get("/search", response_class=HTMLResponse)
def search(
    request: Request,
    q: str = Query("", description="Search query"),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=MAX_LIMIT),
):
    engine: SearchEngine = request.app.state.engine
    result = engine.search(q, limit=limit, offset=(page - 1) * limit)
    return templates.TemplateResponse(
        request=request,
        name="search.html",
        context={"response": result, "engine": engine, "query": q, "page": page},
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
    result = engine.search(q, limit=limit, offset=offset)
    payload = asdict(result)
    payload["has_more"] = result.has_more
    for item, original in zip(payload["results"], result.results):
        item["promulgated"] = original.promulgated.isoformat() if original.promulgated else None
    return JSONResponse(payload)
