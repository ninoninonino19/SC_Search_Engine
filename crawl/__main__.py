"""Stage 1 entry point: walk LawPhil's year -> month -> decision tree.

    python -m crawl --start-year 2010 --end-year 2025

Resumable by construction: every page already on disk is skipped without a
request, so a crash costs you one page rather than a week. Run it twice and the
second run makes almost no network requests at all.

Smoke test before committing to the full corpus:

    python -m crawl --start-year 2019 --end-year 2019 --months mar --max-decisions 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from crawl.config import (
    DEFAULT_DELAY,
    DEFAULT_END_YEAR,
    DEFAULT_START_YEAR,
    MANIFEST_PATH,
    RAW_DIR,
    USER_AGENT,
)
from crawl.decode import decode_html
from crawl.fetch import Fetcher
from crawl.lawphil import MonthPage, decision_urls, month_pages, year_index_url


def _index_dir(root: Path, year: int) -> Path:
    return root / str(year) / "_index"


def _decision_path(root: Path, month: MonthPage, url: str) -> Path:
    # Mirroring the site's own year/month directories keeps filenames unique
    # without inventing an ID scheme, and makes a cached page trivially
    # traceable back to the URL it came from.
    return root / str(month.year) / month.month / url.rsplit("/", 1)[-1]


def crawl_year(fetcher: Fetcher, root: Path, year: int, args) -> None:
    index_url = year_index_url(year)
    index_path = _index_dir(root, year) / f"juri{year}.html"

    if fetcher.fetch(index_url, index_path) == "failed":
        print(f"  {year}: year index unavailable, skipping", file=sys.stderr)
        return

    months = month_pages(decode_html(index_path.read_bytes()), year, index_url)
    if args.months:
        wanted = {m.lower() for m in args.months}
        months = [m for m in months if m.month in wanted]

    for month in months:
        month_path = _index_dir(root, year) / f"{month.slug}.html"
        if fetcher.fetch(month.url, month_path) == "failed":
            print(f"  {month.slug}: month index unavailable, skipping", file=sys.stderr)
            continue

        urls = decision_urls(decode_html(month_path.read_bytes()), month.url, month.slug)
        if args.max_decisions:
            urls = urls[: args.max_decisions]

        for url in urls:
            fetcher.fetch(url, _decision_path(root, month, url))

        print(f"  {month.slug}: {len(urls)} pages linked  |  {fetcher.stats.summary()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crawl", description=__doc__)
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                        help="seconds between requests (default: 1.0)")
    parser.add_argument("--months", nargs="*", default=None,
                        help="restrict to these months, e.g. --months jan feb")
    parser.add_argument("--max-decisions", type=int, default=0,
                        help="cap decisions per month, for smoke tests")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args(argv)

    print(f"User-Agent: {USER_AGENT}")
    if "example.invalid" in USER_AGENT:
        print("  warning: set SC_SEARCH_CONTACT to a real address before a full run",
              file=sys.stderr)

    fetcher = Fetcher(manifest_path=args.manifest, delay=args.delay)

    for year in range(args.start_year, args.end_year + 1):
        print(f"{year}")
        crawl_year(fetcher, args.raw_dir, year, args)

    stats = fetcher.stats
    print("\n" + stats.summary())
    if stats.failures:
        print(f"\nfirst {min(20, len(stats.failures))} failures:")
        for url, reason in stats.failures[:20]:
            print(f"  {reason}  {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
