"""Query parsing: the little language in the search box.

Four things a legal researcher actually types, and what each becomes:

    grave abuse of discretion       AND over the stemmed terms
    "doctrine of condonation"       phrase, checked against stored positions
    ponente:Leonen year:2019        filters over document metadata
    gr:192393  /  G.R. No. 192393   the same docket token, either way

Everything unquoted is AND by default, because a researcher narrowing a search
expects each added word to narrow it. `OR` in capitals switches the remaining
terms to a union — capitalised so the word `or` in a phrase stays a word.

Field values are matched against metadata rather than the index: there are five
of them and they are cheap, and keeping them out of the postings lists avoids
polluting term statistics with names that appear in every decision a justice
wrote.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from indexer.tokenizer import Token, tokenize

METADATA_FIELDS = frozenset({"ponente", "year", "division", "docket"})

# `gr:192393` is not a filter — it is the exact token the tokenizer emits for
# "G.R. No. 192393", so typing either form finds the same thing.
DOCKET_PREFIXES = frozenset({"gr", "am", "ac", "bm", "udk", "ocaipi"})

_PHRASE = re.compile(r'"([^"]*)"')
_FIELD = re.compile(r'\b(\w+):("[^"]*"|\S+)')


@dataclass
class Query:
    raw: str
    terms: list[str] = field(default_factory=list)
    should: list[str] = field(default_factory=list)
    phrases: list[list[Token]] = field(default_factory=list)
    filters: dict[str, str] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (self.terms or self.should or self.phrases or self.filters)

    @property
    def all_terms(self) -> list[str]:
        """Every term worth scoring, phrases included.

        Ranking runs over this: a phrase query should still rank by how often
        its words occur, with the phrase check acting as the filter that decided
        which documents are eligible.
        """
        found = list(self.terms) + list(self.should)
        for phrase in self.phrases:
            found.extend(token.text for token in phrase)
        return found


def parse(raw: str) -> Query:
    query = Query(raw=raw)
    text = raw

    for match in _PHRASE.finditer(text):
        tokens = tokenize(match.group(1))
        if tokens:
            query.phrases.append(tokens)
    text = _PHRASE.sub(" ", text)

    def take_field(match: re.Match) -> str:
        name = match.group(1).lower()
        value = match.group(2).strip('"')
        if name in METADATA_FIELDS:
            query.filters[name] = value
            return " "
        if name in DOCKET_PREFIXES:
            query.terms.append(f"{name}:{value.lower()}")
            return " "
        return match.group(0)

    text = _FIELD.sub(take_field, text)

    # `OR` only counts in capitals, so that the ordinary word `or` — which the
    # stopword list drops anyway — cannot silently widen a query.
    if re.search(r"\bOR\b", text):
        for part in re.split(r"\bOR\b", text):
            query.should.extend(token.text for token in tokenize(part))
    else:
        query.terms.extend(token.text for token in tokenize(text))

    return query


def matches_filters(query: Query, meta) -> bool:
    """Metadata filters, evaluated against one `DocMeta`."""
    for name, value in query.filters.items():
        if name == "year":
            if not value.isdigit() or meta.year != int(value):
                return False
        elif name == "ponente":
            if not meta.ponente or value.lower() not in meta.ponente.lower():
                return False
        elif name == "division":
            if not meta.division or value.lower() not in meta.division.lower():
                return False
        elif name == "docket":
            if value.lower() not in meta.gr_number.lower():
                return False
    return True
