"""Retrieval metrics, and the loading of judged queries.

Nothing here invents a judgment. A query with an empty `relevant_gr` list is
reported as unjudged and excluded from the precision and MRR averages, rather
than being scored as zero — an unjudged query scored as a miss would drag every
average down and make tuning look harmful.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

QUERIES_PATH = Path(__file__).resolve().parent / "queries.json"


@dataclass
class JudgedQuery:
    id: str
    query: str
    intent: str = ""
    category: str = ""
    relevant_gr: list[str] = field(default_factory=list)

    @property
    def is_judged(self) -> bool:
        return bool(self.relevant_gr)

    def is_relevant(self, gr_number: str) -> bool:
        """Match on docket number, tolerantly.

        A consolidated case is stored as "G.R. No. 205972; G.R. No. 164352" and a
        judgment naming either of them should count. Whitespace and case vary
        between what LawPhil prints and what a person types, so both sides are
        squashed before comparison.
        """
        found = _squash(gr_number)
        return any(_squash(relevant) in found for relevant in self.relevant_gr)


def _squash(value: str) -> str:
    return "".join(value.split()).lower()


def load_queries(path: Path = QUERIES_PATH) -> list[JudgedQuery]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        JudgedQuery(
            id=entry["id"],
            query=entry["query"],
            intent=entry.get("intent", ""),
            category=entry.get("category", ""),
            relevant_gr=list(entry.get("relevant_gr", [])),
        )
        for entry in raw
    ]


def save_queries(queries: list[JudgedQuery], path: Path = QUERIES_PATH) -> None:
    payload = [
        {
            "id": q.id,
            "category": q.category,
            "query": q.query,
            "intent": q.intent,
            "relevant_gr": q.relevant_gr,
        }
        for q in queries
    ]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def precision_at_k(judged: JudgedQuery, results, k: int = 10) -> float:
    """Of the top k, how many were relevant. The headline number."""
    top = results[:k]
    if not top:
        return 0.0
    hits = sum(1 for result in top if judged.is_relevant(result.gr_number))
    return hits / min(k, len(top))


def reciprocal_rank(judged: JudgedQuery, results) -> float:
    """1/rank of the first relevant result — "did the right case come up first"."""
    for rank, result in enumerate(results, start=1):
        if judged.is_relevant(result.gr_number):
            return 1.0 / rank
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. Used for p95 latency, where the tail is the point."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * len(ordered) + 0.5)) - 1)
    return ordered[index]
