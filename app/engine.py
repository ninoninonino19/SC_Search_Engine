"""The seam between the web layer and the retrieval layer.

The API depends on this Protocol and nothing else. Today the only implementation
is `FixtureEngine` (a linear scan over a handful of hand-entered cases). When the
inverted index lands, it implements the same three members and gets swapped in at
`app/main.py` — the routes and templates do not change.

Keeping this boundary honest is what stops the interesting work from leaking into
route handlers. Query parsing, stemming, scoring and pagination all live behind
`search()`; the route only turns a `SearchResponse` into HTML.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.models import SearchResponse


@runtime_checkable
class SearchEngine(Protocol):
    """Anything that can answer a query over the decision corpus."""

    @property
    def doc_count(self) -> int:
        """Number of documents currently searchable."""
        ...

    @property
    def name(self) -> str:
        """Short label for the active implementation, shown in the footer.

        Useful in screenshots and demos: it makes it obvious at a glance whether
        you are looking at fixture data or the real index.
        """
        ...

    def search(self, query: str, *, limit: int = 10, offset: int = 0) -> SearchResponse:
        """Return ranked results for `query`.

        Implementations should return an empty result set for a blank or
        all-stopword query rather than raising.
        """
        ...
