"""Stage 4 entry point: build the index and print the numbers.

    python -m indexer                 build from data/parsed/decisions.jsonl
    python -m indexer --measure       also report what the compression bought

Every figure the stage-4 milestone asks you to be able to quote comes out of
here: build time, on-disk size, vocabulary size, and total postings count.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from crawl.config import PROJECT_ROOT
from indexer.codec import decode_postings, posting_count
from indexer.index import InvertedIndex
from indexer.tokenizer import scan, terms

PARSED_PATH = PROJECT_ROOT / "data" / "parsed" / "decisions.jsonl"
INDEX_PATH = PROJECT_ROOT / "data" / "index" / "index.pkl"


def uncompressed_size(index: InvertedIndex) -> int:
    """What the postings would cost as plain pickled Python lists.

    Measured term by term rather than by building the whole uncompressed index
    at once — the point of the compression is that the uncompressed form is the
    thing you cannot comfortably hold.
    """
    total = 0
    for blob in index.postings.values():
        total += len(pickle.dumps(decode_postings(blob), protocol=pickle.HIGHEST_PROTOCOL))
    return total


def vocabulary_reduction(jsonl_path: Path, sample: int = 500) -> tuple[int, int]:
    """Distinct terms before and after stemming, over the same sample of text."""
    import json

    unstemmed: set[str] = set()
    stemmed: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for count, line in enumerate(handle):
            if count >= sample:
                break
            record = json.loads(line)
            text = f"{record.get('title', '')}\n{record.get('body', '')}"
            unstemmed.update(terms(text, stemming=False))
            stemmed.update(terms(text, stemming=True))
    return len(unstemmed), len(stemmed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="indexer", description=__doc__)
    parser.add_argument("--parsed", type=Path, default=PARSED_PATH)
    parser.add_argument("--out", type=Path, default=INDEX_PATH)
    parser.add_argument(
        "--measure",
        action="store_true",
        help="report uncompressed postings size and stemming reduction (slower)",
    )
    args = parser.parse_args(argv)

    if not args.parsed.exists():
        parser.error(f"{args.parsed} does not exist — run `python -m parse` first")

    index = InvertedIndex.build(args.parsed)
    index.save(args.out)

    postings_total = sum(posting_count(blob) for blob in index.postings.values())
    on_disk = args.out.stat().st_size
    compressed = sum(len(blob) for blob in index.postings.values())

    print(f"documents          {index.doc_count:,}")
    print(f"vocabulary         {index.vocabulary_size:,} terms")
    print(f"postings           {postings_total:,} (term, document) pairs")
    print(f"tokens indexed     {index.total_tokens:,}")
    print(f"average length     {index.avgdl:,.0f} tokens")
    print(f"build time         {index.build_seconds:.1f}s")
    print(f"index on disk      {on_disk / 1_048_576:.1f} MiB  ({args.out})")
    print(f"  postings         {compressed / 1_048_576:.1f} MiB delta+varint encoded")

    if args.measure:
        raw = uncompressed_size(index)
        saved = (1 - compressed / raw) * 100 if raw else 0.0
        print(f"  uncompressed     {raw / 1_048_576:.1f} MiB as pickled lists")
        print(f"  saved            {saved:.1f}%")

        before, after = vocabulary_reduction(args.parsed)
        reduction = (1 - after / before) * 100 if before else 0.0
        print(f"vocabulary before stemming {before:,}")
        print(f"vocabulary after stemming  {after:,}  ({reduction:.1f}% smaller)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
