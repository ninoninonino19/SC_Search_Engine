"""Stage 2 entry point: cached HTML to `data/parsed/decisions.jsonl`.

    python -m parse

Reads only from the cache, never the network, so it is cheap to run again every
time a new edge case turns up — which is the whole reason stage 1 writes raw
HTML to disk. Prints a field-completeness report at the end; that report is the
number this milestone asks you to be able to quote.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crawl.config import PROJECT_ROOT, RAW_DIR
from parse.records import build_records
from parse.validate import report

PARSED_PATH = PROJECT_ROOT / "data" / "parsed" / "decisions.jsonl"
REPORT_PATH = PROJECT_ROOT / "data" / "parsed" / "report.txt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="parse", description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=PARSED_PATH)
    args = parser.parse_args(argv)

    records, stats = build_records(args.raw_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    text = report(records, stats)
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwrote {len(records):,} records to {args.out}")
    print(f"report saved to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
