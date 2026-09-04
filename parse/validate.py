"""Field-completeness reporting.

A parser that drops 8% of decisions without saying so corrupts every number
downstream — precision, latency, corpus size, all of it — and does so quietly.
This module is the parser's test suite until there is a real one: it does not
assert, it counts, and it prints what is missing so the number is quotable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

REQUIRED_FIELDS = ("gr_number", "title", "promulgated", "body")
OPTIONAL_FIELDS = ("division", "ponente", "separate_opinions", "footnotes")


@dataclass
class Completeness:
    total: int
    present: dict[str, int]

    def rate(self, field: str) -> float:
        return (self.present[field] / self.total * 100) if self.total else 0.0


def _is_present(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return bool(value)
    return True


def completeness(records: list[dict]) -> Completeness:
    fields = REQUIRED_FIELDS + OPTIONAL_FIELDS
    present = {field: 0 for field in fields}
    for record in records:
        for field in fields:
            if _is_present(record.get(field)):
                present[field] += 1
    return Completeness(total=len(records), present=present)


def report(records: list[dict], stats) -> str:
    lines: list[str] = []
    check = completeness(records)

    years = sorted({r["promulgated"][:4] for r in records if r.get("promulgated")})
    body_lengths = sorted(len(r["body"]) for r in records if r.get("body"))

    lines.append(f"months read        {stats.months}")
    lines.append(f"index rows         {stats.rows}")
    lines.append(f"records written    {stats.records}")
    lines.append(f"separate opinions  {stats.opinions_attached}")
    lines.append(f"missing html       {stats.missing_html}")
    lines.append(f"empty body         {stats.empty_body}")
    if years:
        lines.append(f"date range         {years[0]}-{years[-1]}")
    if body_lengths:
        median = body_lengths[len(body_lengths) // 2]
        lines.append(
            f"body chars         median {median:,}  "
            f"min {body_lengths[0]:,}  max {body_lengths[-1]:,}"
        )

    lines.append("")
    lines.append("field completeness")
    for field in REQUIRED_FIELDS + OPTIONAL_FIELDS:
        marker = "*" if field in REQUIRED_FIELDS else " "
        lines.append(
            f" {marker} {field:<18} {check.present[field]:>6,} / {check.total:,}"
            f"  {check.rate(field):5.1f}%"
        )

    if stats.problems:
        lines.append("")
        lines.append("problems by kind")
        for kind, count in Counter(kind for kind, _ in stats.problems).most_common():
            lines.append(f"   {kind:<18} {count:>6,}")
        lines.append("")
        lines.append("first 10 problems")
        for kind, detail in stats.problems[:10]:
            lines.append(f"   {kind}: {detail}")

    return "\n".join(lines)
