"""Month index pages: the spine of the parse stage.

Each row of a month page carries the citation, the promulgation date, the case
caption, and — critically — links to any separate opinions filed with the
decision. Reading grouping off the index is far more reliable than inferring it
from the decision pages, and it is free: the crawler already cached these.

A row looks like this (2019 markup, reformatted):

    <tr class="xy">
      <td><a href="gr_192393_2019.html">G.R. No. 192393</a><br />March 27, 2019</td>
      <td>Fil-Estate Management, Inc. vs. Republic ...
          Concurring Opinion
          <a href="gr_192393_perlas-bernabe.html">Justice Estela M. Perlas-Bernabe</a></td>
      <td><a href="pdf/gr_192393_2019.pdf">...</a></td>
    </tr>
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from parse.text import clean, to_soup

# "March 27, 2019", "March 27,2019", "March 27 2019"
_DATE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2})\s*,?\s*(\d{4})\b",
    re.IGNORECASE,
)

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

# "G.R. No. 192393", "G.R. Nos. 192393-95", "A.C. No. 9218", "A.M. No. RTJ-19-2552",
# "UDK-16666", "OCA IPI No. 17-4663-P", "B.M. No. 3288".
_CITATION = re.compile(
    r"\b(?:G\.?\s?R\.?|A\.?\s?C\.?|A\.?\s?M\.?|B\.?\s?M\.?|OCA\s+IPI|UDK)"
    r"(?:\s*Nos?\.?)?\s*"
    r"[A-Z]?[\d][\w\-–.]*",
    re.IGNORECASE,
)
# The number stops at `/`: LawPhil writes consolidated cases as
# "G.R. No. 205972/G.R. No. 164352", and letting the number run through the
# slash swallows the next docket's prefix and produces "205972/G.R".

_HTML_LINK = re.compile(r"^[a-z0-9][a-z0-9._\-]*\.html$", re.IGNORECASE)


@dataclass
class ListingRow:
    """One case as the month index describes it."""

    decision_file: str
    citations: list[str]
    caption: str
    promulgated: date | None
    opinion_files: list[str] = field(default_factory=list)

    @property
    def gr_number(self) -> str:
        return "; ".join(self.citations)


def parse_date(text: str) -> date | None:
    match = _DATE.search(text)
    if not match:
        return None
    month = _MONTHS[match.group(1).lower()]
    try:
        return date(int(match.group(3)), month, int(match.group(2)))
    except ValueError:
        return None


def citations(text: str) -> list[str]:
    """Every docket number in `text`, de-duplicated, order preserved.

    Consolidated cases carry several. They are kept as a list rather than
    collapsed, because a search for the second G.R. number of a consolidated
    case should still find it.
    """
    found: dict[str, None] = {}
    for raw in _CITATION.findall(text):
        found.setdefault(clean(raw).rstrip(".,;"), None)
    return list(found)


def parse_month_index(html: str) -> list[ListingRow]:
    soup = to_soup(html)
    rows: list[ListingRow] = []

    for tr in soup.find_all("tr"):
        cells = tr.find_all("td", recursive=False)
        if len(cells) < 2:
            continue

        links = [
            a["href"].strip()
            for a in tr.find_all("a", href=True)
            if _HTML_LINK.match(a["href"].strip())
        ]
        if not links:
            continue

        citation_text = clean(cells[0].get_text(" "))
        caption = clean(cells[1].get_text(" "))

        # The first link is the majority opinion; anything else in the row is a
        # separate opinion filed with it.
        rows.append(
            ListingRow(
                decision_file=links[0],
                citations=citations(citation_text) or citations(caption),
                caption=_strip_opinion_tail(caption),
                promulgated=parse_date(citation_text) or parse_date(caption),
                opinion_files=links[1:],
            )
        )

    return rows


def _strip_opinion_tail(caption: str) -> str:
    """Drop the "Concurring Opinion / Justice So-and-so" tail from a caption.

    It lives in the same cell as the caption but belongs to the separate
    opinion, and left in place it pollutes every caption search with the names
    of justices who merely concurred.
    """
    cut = re.split(
        r"\b(?:Concurring|Dissenting|Separate)\b[\w\s]*\bOpinion\b",
        caption,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return clean(cut).rstrip(" ,;")
