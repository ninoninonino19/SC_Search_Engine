"""Ranking: TF-IDF first, then BM25, so the improvement is measured.

TF-IDF is here to be beaten. Building it costs an hour and it is the only way to
say "precision@10 went from A to B" instead of "BM25 is better, as everyone
knows".

**Why length normalisation matters unusually much in this corpus.** Supreme
Court decisions run from two-page minute resolutions to eighty-page
constitutional rulings — in the sample corpus the longest body is 200 times the
shortest. Raw term frequency ranks by length: an eighty-page decision mentions
*grave abuse of discretion* more times than a focused ten-page one simply
because it mentions everything more times. BM25's `b` term divides term
frequency by document length relative to the corpus average, which is precisely
the correction this corpus needs; that is why `b` is a parameter worth sweeping
here rather than a constant to copy.
"""

from __future__ import annotations

import math
from typing import Iterable, Protocol

from indexer.index import InvertedIndex


class Ranker(Protocol):
    """Anything that turns query terms plus candidate documents into scores."""

    @property
    def name(self) -> str: ...

    def score(
        self,
        index: InvertedIndex,
        terms: Iterable[str],
        candidates: set[int] | None = None,
    ) -> dict[int, float]: ...


class TfIdf:
    """The baseline: `(1 + log tf) * log(N / df)`, cosine-normalised.

    Document norms are computed once on first use — the whole index has to be
    walked to get them, which is a second of work at startup and nothing per
    query. The query vector's norm is deliberately left out: it is a positive
    constant within a single query and so cannot change the ordering.
    """

    def __init__(self) -> None:
        self._norms: list[float] | None = None

    @property
    def name(self) -> str:
        return "tf-idf (cosine-normalised)"

    def _document_norms(self, index: InvertedIndex) -> list[float]:
        if self._norms is not None:
            return self._norms

        totals = [0.0] * index.doc_count
        n_docs = index.doc_count
        for term in index.postings:
            pairs = index.doc_freq_pairs(term)
            idf = math.log(n_docs / len(pairs)) if pairs else 0.0
            if idf == 0.0:
                continue
            for doc_id, term_freq in pairs:
                weight = (1.0 + math.log(term_freq)) * idf
                totals[doc_id] += weight * weight

        self._norms = [math.sqrt(total) if total > 0 else 1.0 for total in totals]
        return self._norms

    def score(
        self,
        index: InvertedIndex,
        terms: Iterable[str],
        candidates: set[int] | None = None,
    ) -> dict[int, float]:
        norms = self._document_norms(index)
        n_docs = index.doc_count
        scores: dict[int, float] = {}

        for term in terms:
            pairs = index.doc_freq_pairs(term)
            if not pairs:
                continue
            idf = math.log(n_docs / len(pairs))
            if idf <= 0.0:
                continue
            for doc_id, term_freq in pairs:
                if candidates is not None and doc_id not in candidates:
                    continue
                weight = (1.0 + math.log(term_freq)) * idf
                scores[doc_id] = scores.get(doc_id, 0.0) + weight * idf

        return {doc_id: value / norms[doc_id] for doc_id, value in scores.items()}


class BM25:
    """Okapi BM25.

        score(d, q) = Σ IDF(t) · tf(t,d)·(k₁+1) / (tf(t,d) + k₁·(1 − b + b·|d|/avgdl))
        IDF(t)      = ln( (N − df(t) + 0.5) / (df(t) + 0.5) + 1 )

    `k₁` controls how fast term frequency saturates: the tenth mention of
    *certiorari* should count for far less than the second. `b` controls length
    normalisation, from 0 (ignore length entirely) to 1 (divide it out fully).
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    @property
    def name(self) -> str:
        return f"bm25 (k1={self.k1:g}, b={self.b:g})"

    def score(
        self,
        index: InvertedIndex,
        terms: Iterable[str],
        candidates: set[int] | None = None,
    ) -> dict[int, float]:
        n_docs = index.doc_count
        avgdl = index.avgdl or 1.0
        lengths = index.doc_lengths
        k1, b = self.k1, self.b
        scores: dict[int, float] = {}

        for term in terms:
            pairs = index.doc_freq_pairs(term)
            if not pairs:
                continue
            doc_freq = len(pairs)
            idf = math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

            for doc_id, term_freq in pairs:
                if candidates is not None and doc_id not in candidates:
                    continue
                norm = k1 * (1.0 - b + b * lengths[doc_id] / avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * term_freq * (k1 + 1.0) / (
                    term_freq + norm
                )

        return scores


def build_ranker(name: str, *, k1: float = 1.2, b: float = 0.75) -> Ranker:
    if name == "tfidf":
        return TfIdf()
    if name == "bm25":
        return BM25(k1=k1, b=b)
    raise ValueError(f"unknown ranker {name!r} (expected 'bm25' or 'tfidf')")
