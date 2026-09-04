"""One decision (or one separate opinion) page, turned into fields.

There is no single "2019 template" and "1995 template" to dispatch on — the same
month serves at least two layouts. `ac_9218_2019.html` and `gr_192393_2019.html`
were promulgated on the same day and share no paragraph classes. So the strategy
here is feature detection rather than era detection: find the opinion heading
(`DECISION`, `R E S O L U T I O N`, `CONCURRING OPINION`), and let it split the
page into header, body, and footnotes. That one landmark is present in every
layout seen so far, and the extractors below key off it rather than off markup.

Footnotes are kept out of `body` but stored, not discarded. They are dense with
`Id.`, `Rollo` and `supra`, which inflate term frequencies without carrying
meaning, but they also hold the citations — so the decision to index them stays
reversible instead of being burned into the parse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, NavigableString, Tag

from parse.listing import citations, parse_date
from parse.text import clean, squash, tag_text, to_soup

# "DECISION", "R E S O L U T I O N", "CONCURRING AND DISSENTING OPINION".
_HEADING = re.compile(r"^(DECISION|RESOLUTION|ORDER|[A-Z]*OPINION)$")

# "CAGUIOA, J.:", "PERLAS-BERNABE, S.A.J.:", "PER CURIAM:", "LEONEN, J:"
_PONENTE = re.compile(
    r"^(PER\s+CURIAM|[A-ZÑÁÉÍÓÚ][A-ZÑÁÉÍÓÚ'\-. ]{1,45}?)"
    r"(?:\s*,\s*((?:[A-Z]\.\s*){0,3}(?:C\.?\s*)?J\.?))?\s*[:.]"
)

_DIVISION = re.compile(
    r"\b(EN\s+BANC"
    r"|(?:FIRST|SECOND|THIRD|FOURTH)\s+DIVISION"
    r"|SPECIAL\s+(?:FIRST|SECOND|THIRD|FOURTH)?\s*DIVISION)\b",
    re.IGNORECASE,
)

_FOOTNOTE_CLASSES = {"jn", "fn"}
_JUSTICE_SUFFIX = re.compile(r",?\s*\[?(C\.?J|S?A?J|JJ?)\.?\]?\s*$", re.IGNORECASE)


@dataclass
class SiblingOpinion:
    """An entry from the page's own opinion group, e.g. "Concurring Opinion, Perlas-Bernabe"."""

    kind: str
    justice: str
    href: str


@dataclass
class ParsedPage:
    kind: str = ""             # DECISION, RESOLUTION, CONCURRING OPINION, ...
    caption: str = ""
    citations: list[str] = field(default_factory=list)
    promulgated_text: str = ""
    division: str | None = None
    ponente: str | None = None
    body: str = ""
    footnotes: str = ""
    siblings: list[SiblingOpinion] = field(default_factory=list)

    @property
    def is_separate_opinion(self) -> bool:
        return "OPINION" in self.kind.upper()


def parse_decision(html: str, *, filename: str = "") -> ParsedPage:
    soup = to_soup(html)
    page = ParsedPage()

    page.siblings = _opinion_group(soup)
    if filename:
        for sibling in page.siblings:
            if sibling.href.lower() == filename.lower():
                page.kind = sibling.kind.upper()
                page.ponente = _format_justice(sibling.justice)

    paragraphs = [(_class_of(p), tag_text(p)) for p in soup.find_all("p")]
    paragraphs = [(cls, text) for cls, text in paragraphs if text]

    heading = _heading_index(paragraphs)
    header = paragraphs[: heading + 1] if heading is not None else paragraphs[:6]
    rest = paragraphs[heading + 1 :] if heading is not None else paragraphs

    if heading is not None and not page.kind:
        page.kind = squash(paragraphs[heading][1])

    header_text = "\n".join(text for _, text in header)
    page.division = _division(header_text) or _division(_head_text(soup))
    page.citations = citations(header_text)
    page.promulgated_text = header_text
    page.caption = _caption(header)

    body_paragraphs, footnote_paragraphs = _split_footnotes(rest)

    if body_paragraphs:
        ponente = _ponente(body_paragraphs[0][1])
        if ponente and len(body_paragraphs[0][1]) < 80:
            # The ponente line is a byline, not an argument. Dropping it keeps a
            # justice's name out of every one of their decisions' term counts.
            page.ponente = page.ponente or ponente
            body_paragraphs = body_paragraphs[1:]

    page.body = clean("\n\n".join(text for _, text in body_paragraphs))
    page.footnotes = clean("\n".join(text for _, text in footnote_paragraphs))
    return page


# -- pieces -----------------------------------------------------------------


def _class_of(tag: Tag) -> str:
    value = tag.get("class")
    return (value[0] if value else "").lower()


def _head_text(soup: BeautifulSoup) -> str:
    title = soup.find("title")
    return clean(title.get_text(" ")) if title else ""


def _heading_index(paragraphs: list[tuple[str, str]]) -> int | None:
    for index, (_, text) in enumerate(paragraphs):
        if len(text) <= 60 and _HEADING.match(squash(text)):
            return index
    return None


def _split_footnotes(
    paragraphs: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Everything from the first footnote marker onward is footnotes.

    Two markers, because the templates disagree: a `jn`-classed paragraph, or a
    standalone `Footnotes` heading with plain paragraphs after it.
    """
    for index, (cls, text) in enumerate(paragraphs):
        if cls in _FOOTNOTE_CLASSES or squash(text) == "FOOTNOTES":
            start = index + 1 if squash(text) == "FOOTNOTES" else index
            return paragraphs[:index], paragraphs[start:]
    return paragraphs, []


def _division(text: str) -> str | None:
    match = _DIVISION.search(text)
    if not match:
        return None
    return " ".join(match.group(1).split()).title()


def _ponente(text: str) -> str | None:
    candidate = text.strip()[:80]
    match = _PONENTE.match(candidate)
    if not match:
        return None
    name = clean(match.group(1)).rstrip(",")
    # Guard against a body paragraph that merely opens in capitals, e.g.
    # "WHEREFORE, the petition is DENIED." — those are not names.
    if name.upper() in {"WHEREFORE", "SO ORDERED", "ACCORDINGLY", "FOOTNOTES"}:
        return None
    if not match.group(2) and "CURIAM" not in name.upper():
        return None
    suffix = clean(match.group(2) or "").rstrip(".")
    return _canonical(name, suffix or "")


def _format_justice(name: str) -> str:
    """`Perlas-Bernabe, [J]` from the opinion group becomes `Perlas-Bernabe, J.`"""
    raw = clean(name)
    suffix = _JUSTICE_SUFFIX.search(raw)
    label = suffix.group(1).upper().rstrip(".") if suffix else "J"
    return _canonical(_JUSTICE_SUFFIX.sub("", raw).rstrip(","), label)


def _canonical(name: str, suffix: str) -> str:
    """One spelling per justice.

    The byline gives `CAGUIOA, J.` and the opinion group gives `Caguioa, [J]`
    for the same person. A field query like `ponente:Caguioa` has to match both,
    so they are collapsed here rather than in the query parser. `.title()`
    already does the right thing with `Perlas-Bernabe` and `O'Brien`.
    """
    titled = name.title().strip(" ,")
    suffix = suffix.upper().rstrip(".")
    return f"{titled}, {suffix}." if suffix else titled


def _caption(header: list[tuple[str, str]]) -> str:
    """The longest header paragraph that is not a heading, date or docket line.

    Captions run long and everything else in the header is short, so length is a
    better signal here than any class name that only one template uses.
    """
    best = ""
    for _, text in header:
        squashed = squash(text)
        if _HEADING.match(squashed) or _DIVISION.search(text):
            continue
        if len(text) < 25:
            continue
        if parse_date(text) and len(text) < 60:
            continue
        if len(text) > len(best):
            best = text
    return " ".join(best.split()).strip(" .[]")


def _opinion_group(soup: BeautifulSoup) -> list[SiblingOpinion]:
    """Read the `div#so` block that lists every opinion filed in the case.

    Present on modern pages and worth reading precisely because it states the
    kind of each opinion and its author — the two fields that are otherwise the
    hardest to get right.
    """
    block = soup.find(id="so")
    if not isinstance(block, Tag):
        return []

    opinions: list[SiblingOpinion] = []
    buffer = ""
    for node in block.descendants:
        if isinstance(node, NavigableString):
            buffer += str(node)
        elif isinstance(node, Tag) and node.name == "a" and node.get("href"):
            # The bullet is U+2666 (the suit symbol), not U+25C6; strip both.
            kind = clean(buffer).split("\n")[-1].strip("♦◆ ,;")
            opinions.append(
                SiblingOpinion(
                    kind=kind or "Opinion",
                    justice=clean(node.get_text(" ")),
                    href=node["href"].strip(),
                )
            )
            buffer = ""
    return opinions
