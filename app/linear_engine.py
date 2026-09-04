"""Linear scan over the corpus. The engine the index exists to beat.

This started life as a fixture engine over a dozen hand-entered placeholder
cases, so the app would run before the crawler did. The placeholders are gone —
invented case law has no business in a legal tool, even as scaffolding — but the
class is worth keeping, pointed at the real corpus, for two reasons:

1. **It is the honest "before" number.** It reads every document, counts terms,
   and sorts. Timing it against the same queries the index answers turns "an
   inverted index is faster" from an assertion into a measurement, which is what
   `python -m evaluation.run --compare` reports.
2. **It is a correctness oracle.** For any conjunctive query, BM25's result
   *set* should match this engine's result set exactly; only the *ordering*
   should differ. When they diverge, the postings lists have a bug.

Deliberately naive: no index, no stemming, no BM25. Scoring is raw term
frequency. Do not optimise it — its slowness is the point.
"""

from __future__ import annotations

import json
import re
import time
from datetime import date
from pathlib import Path

from app.models import SearchResponse, SearchResult

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "parsed" / "decisions.jsonl"

_WORD = re.compile(r"\w+", re.UNICODE)


def _terms(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class LinearScanEngine:
    """Scan every document, every query. O(corpus) per search, by design."""

    name = "linear scan, term-frequency scoring (no index)"

    def __init__(self, path: Path = CORPUS_PATH) -> None:
        self._docs: list[dict] = []
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    self._docs.append(json.loads(line))

    @property
    def doc_count(self) -> int:
        return len(self._docs)

    def search(self, query: str, *, limit: int = 10, offset: int = 0) -> SearchResponse:
        started = time.perf_counter()
        query_terms = _terms(query)

        if not query_terms:
            return SearchResponse(
                query=query,
                results=[],
                total_hits=0,
                took_ms=(time.perf_counter() - started) * 1000,
                limit=limit,
                offset=offset,
            )

        scored: list[tuple[float, dict]] = []
        for doc in self._docs:
            haystack = _terms(f"{doc['gr_number']} {doc['title']} {doc['body']}")
            # Require every query term to appear (implicit AND), matching the
            # default a real boolean retriever would use.
            if not all(term in haystack for term in query_terms):
                continue
            score = float(sum(haystack.count(term) for term in query_terms))
            scored.append((score, doc))

        scored.sort(key=lambda pair: (-pair[0], pair[1]["doc_id"]))
        window = scored[offset : offset + limit]

        results = [
            SearchResult(
                doc_id=doc["doc_id"],
                gr_number=doc["gr_number"],
                title=doc["title"],
                promulgated=_parse_date(doc.get("promulgated")),
                score=score,
                snippet=_snippet(doc["body"], query_terms),
                division=doc.get("division"),
                ponente=doc.get("ponente"),
                source_url=doc.get("source_url", ""),
            )
            for score, doc in window
        ]

        return SearchResponse(
            query=query,
            results=results,
            total_hits=len(scored),
            took_ms=(time.perf_counter() - started) * 1000,
            limit=limit,
            offset=offset,
        )


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _snippet(body: str, query_terms: list[str], width: int = 260) -> str:
    """A window of `body` around the first query-term match, found by scanning.

    The indexed engine does this from stored positions instead, which is both
    faster and more accurate — a substring search lands on the first textual
    occurrence, which after stemming is often not the one that matched.
    """
    lowered = body.lower()
    hit = -1
    for term in query_terms:
        found = lowered.find(term)
        if found != -1 and (hit == -1 or found < hit):
            hit = found

    if hit == -1:
        return body[:width].strip() + ("..." if len(body) > width else "")

    start = max(0, hit - width // 3)
    end = min(len(body), start + width)
    fragment = body[start:end].strip()
    return ("..." if start > 0 else "") + fragment + ("..." if end < len(body) else "")
