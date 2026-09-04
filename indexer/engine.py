"""The `SearchEngine` implementation the API talks to.

Three members — `search()`, `doc_count`, `name` — matching `app.engine.
SearchEngine`, so wiring it in is a one-line change in `app/main.py` and nothing
in `app/` moves.

Retrieval runs in two stages, which is the shape almost every real engine has:

1. **Boolean.** Intersect postings lists (and check phrase positions, and apply
   metadata filters) to get the eligible set. This is exact, and it is where the
   index earns its keep — the work is proportional to the postings lists
   touched, not to the corpus.
2. **Ranking.** Score only the eligible set with BM25 (or TF-IDF, for the
   comparison), sort, and take the window the caller asked for.

Snippets come out of the stored positions rather than a substring search: the
index already knows the match is at token 412, so the snippet is a slice around
token 412's character offset. A substring search would find the first textual
occurrence, which after stemming is frequently not the one that matched.
"""

from __future__ import annotations

import html
import re
import time
from pathlib import Path

from app.models import SearchResponse, SearchResult
from indexer.index import InvertedIndex
from indexer.query import Query, matches_filters, parse
from indexer.ranking import Ranker, build_ranker
from indexer.tokenizer import PROTECTED, STOPWORDS, normalize, scan
from indexer.porter import stem

SNIPPET_WIDTH = 260


class IndexEngine:
    """Positional inverted index plus a ranker."""

    def __init__(self, index: InvertedIndex, ranker: Ranker | None = None) -> None:
        self.index = index
        self.ranker = ranker or build_ranker("bm25")

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        ranker: str = "bm25",
        k1: float = 1.2,
        b: float = 0.75,
    ) -> "IndexEngine":
        return cls(InvertedIndex.load(path), build_ranker(ranker, k1=k1, b=b))

    @property
    def name(self) -> str:
        return f"positional inverted index + {self.ranker.name}"

    @property
    def doc_count(self) -> int:
        return self.index.doc_count

    # -- retrieval ----------------------------------------------------------

    def candidates(self, query: Query) -> list[int] | None:
        """Eligible documents, or None when nothing constrains the set."""
        sets: list[set[int]] = []

        if query.terms:
            sets.append(set(self.index.intersect(query.terms)))
        if query.should:
            sets.append(set(self.index.union(query.should)))
        for phrase in query.phrases:
            positions = [token.position for token in phrase]
            sets.append(set(self.index.phrase([t.text for t in phrase], gaps=positions)))

        if not sets:
            # Filters alone ("everything Leonen wrote in 2019") are a legitimate
            # query; the whole corpus is the starting set.
            return None if not query.filters else list(range(self.index.doc_count))

        eligible = sets[0]
        for other in sets[1:]:
            eligible &= other
        return sorted(eligible)

    def search(self, query: str, *, limit: int = 10, offset: int = 0) -> SearchResponse:
        started = time.perf_counter()
        parsed = parse(query)

        if parsed.is_empty:
            return SearchResponse(
                query=query,
                results=[],
                total_hits=0,
                took_ms=(time.perf_counter() - started) * 1000,
                limit=limit,
                offset=offset,
            )

        eligible = self.candidates(parsed)
        if eligible is not None and parsed.filters:
            eligible = [
                doc_id
                for doc_id in eligible
                if matches_filters(parsed, self.index.docs[doc_id])
            ]

        scored = self.ranker.score(
            self.index,
            parsed.all_terms,
            candidates=set(eligible) if eligible is not None else None,
        )

        # A filters-only query has no terms to score. Ordering by recency is the
        # honest fallback: there is no relevance signal to sort on.
        if not scored and eligible:
            ranked = [(doc_id, 0.0) for doc_id in sorted(eligible, reverse=True)]
        else:
            ranked = sorted(scored.items(), key=lambda pair: (-pair[1], pair[0]))

        window = ranked[offset : offset + limit]
        first_match = self._first_match_positions(
            [doc_id for doc_id, _ in window], parsed
        )
        results = [
            self._result(doc_id, score, first_match.get(doc_id), parsed.all_terms)
            for doc_id, score in window
        ]

        return SearchResponse(
            query=query,
            results=results,
            total_hits=len(ranked),
            took_ms=(time.perf_counter() - started) * 1000,
            limit=limit,
            offset=offset,
        )

    # -- presentation -------------------------------------------------------

    def _result(
        self, doc_id: int, score: float, position: int | None, query_terms: list[str] | None = None,
    ) -> SearchResult:
        meta = self.index.docs[doc_id]
        raw_snippet = self.snippet(doc_id, position)
        highlighted = highlight(raw_snippet, query_terms or [])
        return SearchResult(
            doc_id=doc_id,
            gr_number=meta.gr_number,
            title=meta.title,
            promulgated=meta.promulgated_date,
            score=score,
            snippet=highlighted,
            division=meta.division,
            ponente=meta.ponente,
            source_url=meta.source_url,
        )

    def snippet(self, doc_id: int, position: int | None, width: int = SNIPPET_WIDTH) -> str:
        body = self.index.body(doc_id)
        if position is None:
            return _ellipsis(body, 0, width, len(body))

        tokens = scan(body)
        if position >= len(tokens):
            return _ellipsis(body, 0, width, len(body))

        start = max(0, tokens[position].start - width // 3)
        return _ellipsis(body, start, width, len(body))

    def _first_match_positions(
        self, doc_ids: list[int], query: Query
    ) -> dict[int, int]:
        """Earliest position at which any query term occurs, per document.

        Read straight out of the postings lists — the payoff for storing
        positions. Each term's list is decoded once for the whole result window
        rather than once per result, because a common term's postings list is
        long and decoding it ten times to answer ten rows is the kind of thing
        that turns a fast engine into a slow one.
        """
        wanted = set(doc_ids)
        earliest: dict[int, int] = {}

        # A phrase match is the better anchor when there is one: the reader
        # asked for the phrase, so that is what the snippet should open on.
        for phrase in query.phrases:
            positions = [token.position for token in phrase]
            for doc_id, anchor in self.index.phrase_matches(
                [token.text for token in phrase], gaps=positions
            ).items():
                if doc_id in wanted:
                    earliest[doc_id] = anchor
        phrase_anchored = set(earliest)
        if len(phrase_anchored) == len(wanted):
            return earliest

        for term in set(query.all_terms):
            for doc_id, _, positions in self.index.postings_for(term):
                if doc_id not in wanted or doc_id in phrase_anchored or not positions:
                    continue
                current = earliest.get(doc_id)
                if current is None or positions[0] < current:
                    earliest[doc_id] = positions[0]

        return earliest


def _ellipsis(text: str, start: int, width: int, length: int) -> str:
    end = min(length, start + width)
    fragment = text[start:end].strip().replace("\n", " ")
    return ("…" if start > 0 else "") + fragment + ("…" if end < length else "")


def highlight(snippet: str, query_terms: list[str]) -> str:
    """Wrap tokens in *snippet* that match *query_terms* with <mark> tags.

    Operates on the raw snippet text: re-scans it, stems each word the same way
    the indexer does, and wraps surface forms whose stems match a query term.
    Uses the original text for output to preserve casing.
    """
    if not query_terms:
        return html.escape(snippet)

    term_set = set(query_terms)
    normed = normalize(snippet)
    _HAS_DIGIT = re.compile(r"\d")
    spans: list[tuple[int, int]] = []

    for tok in scan(snippet):
        value = tok.text
        if ":" in value or value == "vs":
            if value in term_set:
                spans.append((tok.start, tok.end))
            continue
        if _HAS_DIGIT.search(value):
            continue
        if value in STOPWORDS:
            continue
        stemmed = value if value in PROTECTED else stem(value)
        if stemmed in term_set:
            spans.append((tok.start, tok.end))

    if not spans:
        return html.escape(snippet)

    parts: list[str] = []
    prev = 0
    for s, e in spans:
        parts.append(html.escape(snippet[prev:s]))
        parts.append(f"<mark>{html.escape(snippet[s:e])}</mark>")
        prev = e
    parts.append(html.escape(snippet[prev:]))
    return "".join(parts)
