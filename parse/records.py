"""Cached HTML plus month indexes, assembled into `CaseDocument` records.

The month index is the spine: it decides what a case *is*, which file holds the
majority opinion, and which files are separate opinions filed with it. The
decision pages supply the text, the division and the ponente. Where they
disagree the index wins on grouping and the page wins on content, because the
index is generated from the same database for every row while the pages were
laid out by hand over fifteen years.

**One record per case, not per opinion.** A concurrence is stored in
`separate_opinions` rather than concatenated into `body`, which leaves the real
question open for the indexer: index a case as one document, or index each
opinion as its own. That choice changes BM25's length normalisation and what a
result row means, and it should not be pre-empted here.

`doc_id`s are assigned in chronological order. Dense integers compress well as
postings-list deltas, and a chronological assignment means a year filter is a
contiguous range of IDs rather than a scattered set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from crawl.decode import decode_html
from crawl.lawphil import MONTHS
from parse.decision import ParsedPage, parse_decision
from parse.listing import ListingRow, parse_month_index

BASE_URL = "https://lawphil.net/judjuris"


@dataclass
class ParseStats:
    """Counts for the parser's own report. Silent failures corrupt every
    downstream metric, so everything that goes wrong is counted here."""

    months: int = 0
    rows: int = 0
    records: int = 0
    missing_html: int = 0
    empty_body: int = 0
    opinions_attached: int = 0
    problems: list[tuple[str, str]] = field(default_factory=list)

    def note(self, kind: str, detail: str) -> None:
        self.problems.append((kind, detail))


def source_url(year: int, month: str, filename: str) -> str:
    return f"{BASE_URL}/juri{year}/{month}{year}/{filename}"


def _month_dirs(raw_dir: Path) -> list[tuple[int, str, Path]]:
    """(year, month, index-path) for every month page in the cache, in order."""
    found: list[tuple[int, str, Path]] = []
    for index_path in sorted(raw_dir.glob("*/_index/*.html")):
        name = index_path.stem  # "mar2019"
        if len(name) != 7 or not name[3:].isdigit():
            continue  # the year index itself, "juri2019"
        found.append((int(name[3:]), name[:3], index_path))
    # Calendar order, not alphabetical — "apr" before "aug" before "dec" would
    # scatter doc_ids through the year and undo the point of assigning them
    # chronologically.
    return sorted(found, key=lambda item: (item[0], MONTHS.index(item[1])))


def _opinion_block(page: ParsedPage) -> str:
    """A separate opinion, with its kind and author kept in the text.

    `CaseDocument.separate_opinions` is a tuple of strings, so the only place the
    authorship can survive is the string itself. Keeping it there means a query
    for a dissenting justice still reaches the dissent.
    """
    header = page.kind.title() if page.kind else "Separate Opinion"
    if page.ponente:
        header = f"{header} — {page.ponente}"
    return f"{header}\n\n{page.body}".strip()


def build_record(
    row: ListingRow,
    year: int,
    month: str,
    month_dir: Path,
    doc_id: int,
    stats: ParseStats,
) -> dict | None:
    html_path = month_dir / row.decision_file
    if not html_path.exists():
        stats.missing_html += 1
        stats.note("missing-html", str(html_path))
        return None

    page = parse_decision(decode_html(html_path.read_bytes()), filename=row.decision_file)

    opinions: list[str] = []
    for opinion_file in row.opinion_files:
        opinion_path = month_dir / opinion_file
        if not opinion_path.exists():
            stats.missing_html += 1
            stats.note("missing-opinion", str(opinion_path))
            continue
        opinion_page = parse_decision(
            decode_html(opinion_path.read_bytes()), filename=opinion_file
        )
        if opinion_page.body:
            opinions.append(_opinion_block(opinion_page))
            stats.opinions_attached += 1

    if not page.body:
        stats.empty_body += 1
        stats.note("empty-body", str(html_path))

    promulgated: date | None = row.promulgated
    citations = row.citations or page.citations

    return {
        "doc_id": doc_id,
        "gr_number": "; ".join(citations),
        "title": row.caption or page.caption,
        "promulgated": promulgated.isoformat() if promulgated else None,
        "body": page.body,
        "division": page.division,
        "ponente": page.ponente,
        "source_url": source_url(year, month, row.decision_file),
        "separate_opinions": opinions,
        "footnotes": page.footnotes,
    }


def build_records(raw_dir: Path) -> tuple[list[dict], ParseStats]:
    stats = ParseStats()
    records: list[dict] = []

    for year, month, index_path in _month_dirs(raw_dir):
        stats.months += 1
        month_dir = raw_dir / str(year) / month
        rows = parse_month_index(decode_html(index_path.read_bytes()))
        stats.rows += len(rows)

        for row in rows:
            record = build_record(row, year, month, month_dir, len(records), stats)
            if record is not None:
                records.append(record)

    stats.records = len(records)
    return records, stats
