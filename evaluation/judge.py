"""The few hours of work that turn assertions into measurements.

    python -m evaluation.judge              judge every unjudged query
    python -m evaluation.judge --id q03     judge or re-judge one
    python -m evaluation.judge --depth 20   how many results to show

For each query it prints the top results and asks which are relevant. Answer
with the numbers of the relevant results (`1 3 4`), `n` for none, `s` to skip,
or `q` to save and stop. Judgments are written back into `queries.json` as
docket numbers, so they survive re-crawling, re-parsing and re-indexing — a
judgment recorded against a doc_id would be invalidated by the next rebuild.

Nothing here judges anything on your behalf. Relevance to a legal query is a
question about law, and an automated guess at it would produce numbers that look
like measurements and are not.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from crawl.config import PROJECT_ROOT
from evaluation.metrics import JudgedQuery, load_queries, save_queries
from indexer.engine import IndexEngine

INDEX_PATH = PROJECT_ROOT / "data" / "index" / "index.pkl"


def judge_one(engine: IndexEngine, judged: JudgedQuery, depth: int) -> str:
    response = engine.search(judged.query, limit=depth)

    print("=" * 78)
    print(f"[{judged.id}] {judged.query}")
    if judged.intent:
        print(f"      {judged.intent}")
    print(f"      {response.total_hits} hits in {response.took_ms:.1f} ms")
    if judged.relevant_gr:
        print(f"      already judged: {', '.join(judged.relevant_gr)}")
    print("-" * 78)

    if not response.results:
        print("  (no results)")
        return "next"

    for number, result in enumerate(response.results, start=1):
        marker = "*" if judged.is_relevant(result.gr_number) else " "
        year = result.year or "—"
        print(f"{marker}{number:>3}. {result.gr_number}  ({year})  {result.title[:70]}")
        print(f"      {result.snippet[:150]}")

    print()
    answer = input("relevant numbers / n=none / s=skip / q=save and quit: ").strip().lower()

    if answer in {"q", "quit"}:
        return "quit"
    if answer in {"s", "skip", ""}:
        return "next"
    if answer in {"n", "none"}:
        judged.relevant_gr = []
        return "next"

    chosen: list[str] = []
    for piece in answer.replace(",", " ").split():
        if not piece.isdigit():
            continue
        position = int(piece)
        if 1 <= position <= len(response.results):
            chosen.append(response.results[position - 1].gr_number)

    judged.relevant_gr = chosen
    print(f"  recorded {len(chosen)} relevant")
    return "next"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evaluation.judge", description=__doc__)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--depth", type=int, default=20, help="results to show per query")
    parser.add_argument("--id", dest="query_id", help="judge only this query id")
    parser.add_argument(
        "--all", action="store_true", help="revisit queries that are already judged"
    )
    args = parser.parse_args(argv)

    if not args.index.exists():
        parser.error(f"no index at {args.index} — run `python -m indexer` first")

    engine = IndexEngine.load(args.index)
    queries = load_queries()

    pending = [
        q
        for q in queries
        if (args.query_id is None or q.id == args.query_id)
        and (args.all or args.query_id is not None or not q.is_judged)
    ]

    if not pending:
        print("nothing to judge — every query already has judgments (use --all to revisit)")
        return 0

    print(f"{len(pending)} queries to judge, {args.depth} results each.\n")

    for judged in pending:
        if judge_one(engine, judged, args.depth) == "quit":
            break

    save_queries(queries)
    total = sum(1 for q in queries if q.is_judged)
    print(f"\nsaved. {total} of {len(queries)} queries now judged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
