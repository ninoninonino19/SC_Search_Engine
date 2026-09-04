"""What LawPhil's URL structure looks like, and how to walk it.

The site is three levels deep:

    judjuris/judjuris.html                     index of years
    judjuris/juri2019/juri2019.html            one year, links to months
    judjuris/juri2019/mar2019/mar2019.html     one month, links to decisions
    judjuris/juri2019/mar2019/gr_192393_2019.html   one decision

Month links are discovered from the year page rather than generated, because a
year with no decisions in a month simply omits the link, and because the naming
has drifted in places. Decision links are discovered the same way.

Pure functions over HTML strings — no network, no disk, so this is the part that
is cheap to test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from crawl.config import BASE_URL

MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")

# "mar2019/mar2019.html" — the month directory and the page inside it agree.
_MONTH_HREF = re.compile(r"^(" + "|".join(MONTHS) + r")(\d{4})/\1\2\.html$", re.IGNORECASE)

# "gr_192393_2019.html", "am_rtj-19-2552_2019.html", "ac_9218_2019.html", and
# separate opinions like "gr_192393_perlas-bernabe.html". Anything relative,
# ending in .html, that is not the month page itself.
_DECISION_HREF = re.compile(r"^[a-z0-9][a-z0-9._\-]*\.html$", re.IGNORECASE)


@dataclass(frozen=True)
class MonthPage:
    year: int
    month: str  # three-letter lowercase, e.g. "mar"
    url: str

    @property
    def slug(self) -> str:
        return f"{self.month}{self.year}"


def year_index_url(year: int) -> str:
    return urljoin(BASE_URL, f"juri{year}/juri{year}.html")


def _links(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    return [a["href"].strip() for a in soup.find_all("a", href=True)]


def month_pages(html: str, year: int, page_url: str) -> list[MonthPage]:
    """Month pages linked from a year index, in calendar order.

    The year page lists months in a 3x4 grid that runs bottom-up (January is in
    the last row), so the order on the page is not calendar order. Sorting here
    means the crawl and the manifest read chronologically.
    """
    found: dict[str, MonthPage] = {}
    for href in _links(html):
        match = _MONTH_HREF.match(href)
        if not match:
            continue
        month = match.group(1).lower()
        # A year page occasionally links a neighbouring year's month; keep only
        # the year we asked for so a page never lands in the wrong directory.
        if int(match.group(2)) != year:
            continue
        found[month] = MonthPage(year=year, month=month, url=urljoin(page_url, href))
    return [found[m] for m in MONTHS if m in found]


def decision_urls(html: str, page_url: str, month_slug: str) -> list[str]:
    """Decision and separate-opinion URLs linked from a month index.

    Both are returned. Which of them is a majority opinion and which is a
    concurrence is a parse-stage question — the month page carries that in its
    row markup, and the cached HTML keeps it available without re-fetching.
    """
    seen: dict[str, None] = {}
    for href in _links(html):
        if not _DECISION_HREF.match(href):
            continue
        if href.lower() == f"{month_slug}.html":
            continue
        seen.setdefault(urljoin(page_url, href), None)
    return list(seen)
