"""Stage 6: score ranking configurations against the judged query set.

    python -m evaluation.run                 TF-IDF vs BM25, P@10 / MRR / latency
    python -m evaluation.run --sweep         sweep k1 and b, ranked by P@10
    python -m evaluation.run --compare       index vs linear scan, latency only

Built before tuning, deliberately. Tune first and evaluate after, and every
improvement is a story rather than a measurement — there is no baseline left to
compare against.

Latency is reported separately from precision, and always: it needs no
judgments, so the `--compare` numbers are meaningful from the first run.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from crawl.config import PROJECT_ROOT
from evaluation.metrics import (
    JudgedQuery,
    load_queries,
    mean,
    percentile,
    precision_at_k,
    reciprocal_rank,
)
from indexer.engine import IndexEngine
from indexer.index import InvertedIndex
from indexer.ranking import build_ranker

INDEX_PATH = PROJECT_ROOT / "data" / "index" / "index.pkl"
REPORT_PATH = PROJECT_ROOT / "evaluation" / "results.txt"


class Outcome:
    """One configuration's numbers over the whole query set."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.precisions: list[float] = []
        self.reciprocal_ranks: list[float] = []
        self.latencies: list[float] = []
        self.hits: list[int] = []

    @property
    def judged_count(self) -> int:
        return len(self.precisions)

    def row(self) -> str:
        return (
            f"{self.label:<34}"
            f"{mean(self.precisions):>8.3f}"
            f"{mean(self.reciprocal_ranks):>8.3f}"
            f"{percentile(self.latencies, 0.5):>11.1f}"
            f"{percentile(self.latencies, 0.95):>10.1f}"
            f"{self.judged_count:>9}"
        )


HEADER = (
    f"{'configuration':<34}{'P@10':>8}{'MRR':>8}"
    f"{'median ms':>11}{'p95 ms':>10}{'judged':>9}"
)


def evaluate(engine, queries: list[JudgedQuery], label: str, *, repeats: int = 3) -> Outcome:
    outcome = Outcome(label)

    for judged in queries:
        # Repeat and take the fastest run: the first call to a query pulls its
        # postings and its snippet source off disk, and that page-cache miss is
        # not what the ranking costs.
        best = float("inf")
        response = None
        for _ in range(repeats):
            started = time.perf_counter()
            response = engine.search(judged.query, limit=10)
            best = min(best, (time.perf_counter() - started) * 1000)

        outcome.latencies.append(best)
        outcome.hits.append(response.total_hits)

        if judged.is_judged:
            outcome.precisions.append(precision_at_k(judged, response.results, k=10))
            outcome.reciprocal_ranks.append(reciprocal_rank(judged, response.results))

    return outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evaluation.run", description=__doc__)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--sweep", action="store_true", help="sweep k1 and b")
    parser.add_argument(
        "--compare", action="store_true", help="also time the linear-scan baseline"
    )
    args = parser.parse_args(argv)

    if not args.index.exists():
        parser.error(f"no index at {args.index} — run `python -m indexer` first")

    queries = load_queries()
    judged = [q for q in queries if q.is_judged]
    index = InvertedIndex.load(args.index)

    lines: list[str] = []
    lines.append(f"corpus     {index.doc_count:,} documents, {index.vocabulary_size:,} terms")
    lines.append(f"queries    {len(queries)} total, {len(judged)} judged")
    lines.append("")
    lines.append(HEADER)
    lines.append("-" * len(HEADER))

    configurations: list[tuple[str, dict]] = [
        ("tf-idf (baseline)", {"name": "tfidf"}),
        ("bm25 k1=1.2 b=0.75 (default)", {"name": "bm25", "k1": 1.2, "b": 0.75}),
    ]
    if args.sweep:
        configurations = [("tf-idf (baseline)", {"name": "tfidf"})] + [
            (f"bm25 k1={k1:g} b={b:g}", {"name": "bm25", "k1": k1, "b": b})
            for k1 in (0.9, 1.2, 1.5, 2.0)
            for b in (0.0, 0.3, 0.5, 0.75, 1.0)
        ]

    for label, config in configurations:
        engine = IndexEngine(index, build_ranker(**config))
        lines.append(evaluate(engine, queries, label).row())

    if args.compare:
        from app.linear_engine import LinearScanEngine

        lines.append("")
        lines.append("baseline without an index")
        lines.append("-" * len(HEADER))
        lines.append(evaluate(LinearScanEngine(), queries, "linear scan", repeats=1).row())

    if not judged:
        lines.append("")
        lines.append(
            "No queries are judged yet, so P@10 and MRR are 0.000 by definition —\n"
            "they are averages over an empty set, not a measurement of bad ranking.\n"
            "Judge the query set first:  python -m evaluation.judge"
        )

    text = "\n".join(lines)
    REPORT_PATH.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nsaved to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
