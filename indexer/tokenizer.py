"""Text to tokens. Pure: string in, tokens out, no I/O.

This is where the domain shows up. A generic tokenizer does the wrong thing to
legal text in four specific ways, and each of them is handled here:

* **Citations shatter.** `G.R. No. 192393` becomes `g`, `r`, `no`, `192393` —
  four useless tokens instead of one precise one. Docket numbers are matched
  before generic word splitting and emitted whole, as `gr:192393`.
* **Section references never collide.** `Sec. 5`, `Section 5` and `Secs. 5` are
  the same reference written three ways. They are normalised to `sec:5`, so a
  search for one finds all of them, and `Art. 315(2)(a)` keeps its subsections.
* **Latin gets mangled.** Porter turns `mandamus` into `mandamu` and would
  happily do worse. A protected-terms list skips stemming for the vocabulary the
  algorithm was never designed for.
* **Stopword lists eat evidence.** Every off-the-shelf English list contains
  `no` and `party`. In case law `G.R. No.`, `no fault`, `third party` and
  `real party in interest` all matter. The list below is built from scratch and
  deliberately short — see `STOPWORDS`.

**Positions count every token the scanner sees**, including the stopwords and
bare numerals that are then dropped. Keeping the gaps means a phrase query for
`grave abuse of discretion` still lines up against an index that never stored
`of`: the query and the index agree on the gap because they went through the
same function.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from indexer.porter import stem

# Deliberately short. Every word here is a pure function word that carries no
# retrieval signal in this corpus. Words a generic English list would include
# but this one does not, and why:
#   no       - "G.R. No.", "no fault", "no less than"
#   not      - negation flips a holding
#   party    - "third party", "real party in interest", "indispensable party"
#   may/shall/must - the difference between permissive and mandatory language,
#                    which is frequently the entire dispute
#   against/under/before/without - "under Rule 65", "before the trial court"
STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those
    of to in on at by for with from as into onto upon
    is are was were be been being am
    it its they them their there here he she his her him we us our you your
    which who whom whose what when where while
    such so also thus hence however moreover furthermore
    have has had having do does did done
    said same other another any each both all some more most
    """.split()
)

# Terms the stemmer must not touch. Latin of art, mostly: Porter was built for
# English morphology and these have none of it. `mandamus` -> `mandamu` and
# `res` -> `re` are both silent precision losses.
#
# Only the content words are listed. `a quo` and `mala in se` lose their `a` and
# `in` to the stopword list, which is fine — positions are gap-preserving, so a
# phrase query for either still matches.
PROTECTED = frozenset(
    """
    certiorari mandamus prohibition quo warranto habeas corpus
    res judicata litis pendentia forum shopping
    prima facie mala prohibita se
    dolo culpa delicto flagrante
    amicus curiae stare decisis obiter dictum dicta ratio decidendi
    ultra vires nunc pro tunc pro hac vice
    ex parte ex post facto ipso facto de facto de jure
    quo alibi animus mens rea actus reus
    ponente ponencia en banc
    """.split()
)

# "G.R. No. 192393", "G.R. Nos. 192393-95", "A.M. No. RTJ-19-2552",
# "A.C. No. 9218", "G.R. No. L-45081", "UDK-16666", "OCA IPI No. 17-4663-P".
_DOCKET = r"""
    (?P<docket_kind>g\.?\s?r\.?|a\.?\s?m\.?|a\.?\s?c\.?|b\.?\s?m\.?|oca\s+ipi|udk)
    \s*(?:nos?\.?)?\s*
    (?P<docket_number>[a-z]{0,4}-?\d[\w\-]*)
"""

# "Sec. 5", "Section 5", "Art. 315(2)(a)", "Rule 65", "par. 2".
_SECTION = r"""
    \b(?P<section_kind>sections?|secs?\.?|articles?|arts?\.?|rules?|pars?\.?|paragraphs?)
    \s*(?P<section_number>\d+[a-z]?)
    (?P<section_subs>(?:\s*\(\s*[\da-z]{1,4}\s*\))*)
"""

# The `v.` in `People v. Santos`. Dropping it costs the ability to search a
# caption as a caption.
_VERSUS = r"\bvs?\.(?=\s)|\bvs\b"

_WORD = r"[a-z0-9ñáéíóúü]+(?:['’][a-z]+)?"

_SCANNER = re.compile(
    "|".join(
        (
            f"(?P<docket>{_DOCKET})",
            f"(?P<section>{_SECTION})",
            f"(?P<versus>{_VERSUS})",
            f"(?P<word>{_WORD})",
        )
    ),
    re.VERBOSE,
)

_HAS_DIGIT = re.compile(r"\d")

_SECTION_CANON = {
    "sec": "sec", "secs": "sec", "section": "sec", "sections": "sec",
    "art": "art", "arts": "art", "article": "art", "articles": "art",
    "rule": "rule", "rules": "rule",
    "par": "par", "pars": "par", "paragraph": "par", "paragraphs": "par",
}


@dataclass(frozen=True)
class Token:
    """One indexable term, where it sat in the token stream, and where in the text.

    `start`/`end` are character offsets into the *normalised* text. They are what
    turns a stored position back into a snippet: the index knows a term occurred
    at token 412, and re-scanning the body maps that to a character offset
    without a substring search that could land on the wrong occurrence.
    """

    text: str
    position: int
    start: int = 0
    end: int = 0


def normalize(text: str) -> str:
    """NFC, lowercase, curly quotes flattened. Idempotent."""
    text = unicodedata.normalize("NFC", text).lower()
    return text.replace("’", "'").replace("‘", "'")


def _docket_token(kind: str, number: str) -> str:
    prefix = re.sub(r"[^a-z]", "", kind)
    return f"{prefix}:{number.strip('-')}"


def _section_token(kind: str, number: str, subs: str) -> str:
    prefix = _SECTION_CANON.get(re.sub(r"[^a-z]", "", kind), "sec")
    return f"{prefix}:{number}{re.sub(r'\s+', '', subs)}"


def scan(text: str) -> list[Token]:
    """Every token the scanner sees, before stopwords or stemming.

    Exposed separately because it is the honest denominator: vocabulary "before
    stemming" means the output of this, not of `tokenize`.
    """
    tokens: list[Token] = []
    for position, match in enumerate(_SCANNER.finditer(normalize(text))):
        if match.lastgroup is None:
            continue
        if match.group("docket"):
            value = _docket_token(match.group("docket_kind"), match.group("docket_number"))
        elif match.group("section"):
            value = _section_token(
                match.group("section_kind"),
                match.group("section_number"),
                match.group("section_subs") or "",
            )
        elif match.group("versus"):
            value = "vs"
        else:
            value = match.group("word")
        tokens.append(
            Token(text=value, position=position, start=match.start(), end=match.end())
        )
    return tokens


def tokenize(text: str, *, stemming: bool = True) -> list[Token]:
    """The indexing pipeline: scan, drop noise, stem what should be stemmed.

    `stemming=False` is not a convenience — it is how the "vocabulary before and
    after stemming" number gets measured on identical inputs.
    """
    output: list[Token] = []
    for token in scan(text):
        value = token.text

        # Structured tokens are already canonical; stemming them would only
        # break the collision they exist to create.
        if ":" in value or value == "vs":
            output.append(token)
            continue

        # Anything with a digit left in it at this point is a page number, a
        # peso amount, a date, or the debris of one (`P1,500,000.00` splits into
        # `p1`, `500`, `000`, `00`). All of it bloats the vocabulary and ranks
        # nothing. Digits inside a docket or section token survive, because
        # there they carry the entire meaning.
        if _HAS_DIGIT.search(value):
            continue

        if value in STOPWORDS:
            continue

        if stemming and value not in PROTECTED:
            value = stem(value)

        if value:
            output.append(
                Token(
                    text=value,
                    position=token.position,
                    start=token.start,
                    end=token.end,
                )
            )
    return output


def terms(text: str, *, stemming: bool = True) -> list[str]:
    """`tokenize` without the positions, for callers that do not need them."""
    return [token.text for token in tokenize(text, stemming=stemming)]
