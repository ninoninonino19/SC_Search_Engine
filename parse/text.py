"""HTML to clean text.

Two jobs, both boring and both load-bearing:

* **Strip what is not the decision.** Navigation scripts, the site chrome, and
  LawPhil's `<cite>` watermarks — short runs of private-use characters injected
  mid-sentence to catch copy-paste — all look like body text to a tokenizer and
  none of them are.
* **Normalise.** Pages are windows-1252 with UTF-8 declared on some and
  `windows-1252!` on others; non-breaking spaces are used as ordinary spaces
  throughout. Everything comes out NFC-normalised with plain spaces, so the
  tokenizer never has to think about it.
"""

from __future__ import annotations

import re
import unicodedata

from bs4 import BeautifulSoup, Tag

# Site chrome, and the watermark spans that sit inside sentences.
_DROP_TAGS = ("script", "style", "cite", "noscript")

_WS = re.compile(r"[ \t\u00a0\u2007\u202f]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def to_soup(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(_DROP_TAGS):
        tag.decompose()
    return soup


def clean(text: str) -> str:
    """NFC, real spaces, no runs of blank lines. Safe to call twice."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", text).strip()


def tag_text(tag: Tag) -> str:
    """Text of one element, with `<br>` treated as a line break."""
    return clean(tag.get_text("\n" if tag.find("br") else " "))


def squash(text: str) -> str:
    """Uppercase with all whitespace removed.

    Headings are letter-spaced on some templates (`D E C I S I O N`) and not on
    others (`DECISION`). Squashing makes one comparison work for both.
    """
    return re.sub(r"\s+", "", text).upper()
