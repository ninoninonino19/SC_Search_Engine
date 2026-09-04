"""Domain models shared across the crawl -> parse -> index -> search pipeline.

These are deliberately plain dataclasses with no framework imports. The crawler,
the indexer and the API all speak this vocabulary, so none of them has to depend
on any of the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class CaseDocument:
    """One Supreme Court decision, as produced by the parse stage.

    `doc_id` is the internal integer-backed identifier used in postings lists;
    `gr_number` is the human citation. They are separate on purpose: postings
    lists get much smaller when doc IDs are dense integers you can delta-encode,
    and G.R. numbers are neither dense nor reliably unique (consolidated cases
    share one, and a few historical entries reuse them).
    """

    doc_id: int
    gr_number: str
    title: str
    promulgated: date | None
    body: str
    division: str | None = None
    ponente: str | None = None
    source_url: str = ""
    # Separate opinions are kept alongside the majority rather than concatenated
    # into `body`, so the indexing decision (one document or several) stays open.
    separate_opinions: tuple[str, ...] = field(default_factory=tuple)
    # Footnotes are parsed out of `body` — they are dense with `Id.`, `supra` and
    # `Rollo`, which inflate term frequencies without carrying meaning — but they
    # are kept rather than dropped, because they also hold the citations. Storing
    # them separately means "index the footnotes too" stays a one-line experiment
    # instead of a re-parse of the whole corpus.
    footnotes: str = ""

    @property
    def year(self) -> int | None:
        return self.promulgated.year if self.promulgated else None


@dataclass(frozen=True)
class SearchResult:
    """A single ranked hit."""

    doc_id: int
    gr_number: str
    title: str
    promulgated: date | None
    score: float
    snippet: str
    division: str | None = None
    ponente: str | None = None
    source_url: str = ""

    @property
    def year(self) -> int | None:
        return self.promulgated.year if self.promulgated else None


@dataclass(frozen=True)
class SearchResponse:
    """Everything the API layer needs to render a result page.

    `took_ms` is measured here rather than in the route handler so that the
    number reported is retrieval time, not retrieval plus template rendering.
    That distinction matters once you start quoting latency on a resume.
    """

    query: str
    results: list[SearchResult]
    total_hits: int
    took_ms: float
    limit: int = 10
    offset: int = 0

    @property
    def has_more(self) -> bool:
        return self.offset + len(self.results) < self.total_hits
