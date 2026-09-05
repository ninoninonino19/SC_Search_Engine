"""The positional inverted index.

    term -> [ (doc_id, term_freq, [positions...]), ... ]   sorted by doc_id

plus a document store mapping `doc_id -> (length_in_tokens, metadata)`.

Three decisions worth stating, because each of them is load-bearing:

**Positions are stored from the start.** They roughly triple the index and they
are not optional: retrofitting them later means rebuilding everything, and they
pay for themselves twice — phrase search needs them, and snippet generation gets
to jump straight to the match instead of re-scanning the body for a substring.

**Postings are sorted by doc_id.** That is what makes intersecting two lists a
linear merge rather than a set operation, and it is what makes the delta
encoding in `indexer.codec` possible at all.

**One indexed document per case, not per opinion.** A concurrence is indexed
with the case that produced it. The alternative — a document per opinion — makes
`dissenting` behave more sensibly as a query and keeps BM25's length
normalisation over comparable units, but it lets one case occupy three result
slots, and a reader searching case law is looking for cases. Only 6% of the
decisions in this corpus carry a separate opinion, so the distortion to `avgdl`
that the split would have fixed is small and measurable rather than serious.
`_unit_text` is the one function that changes if that trade is ever revisited.

**Bodies are not stored here.** The index keeps a byte offset into
`decisions.jsonl` per document and seeks for the text when a snippet is needed.
An index that also carried the full text of 15,000 decisions would be dominated
by data that ranking never reads.
"""

from __future__ import annotations

import json
import pickle
import time
from array import array
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from indexer.codec import decode_postings, iter_doc_freqs, read_varint, write_varint
from indexer.tokenizer import Token, scan, tokenize

INDEX_VERSION = 1


@dataclass(frozen=True)
class DocMeta:
    """Everything a result row needs that is not the body text."""

    doc_id: int
    gr_number: str
    title: str
    promulgated: str | None
    division: str | None
    ponente: str | None
    source_url: str
    length: int
    offset: int  # byte offset of this record's line in decisions.jsonl

    @property
    def year(self) -> int | None:
        return int(self.promulgated[:4]) if self.promulgated else None

    @property
    def promulgated_date(self) -> date | None:
        return date.fromisoformat(self.promulgated) if self.promulgated else None


def _unit_text(record: dict) -> str:
    """The text of one indexed document.

    The docket number comes first, because the parser strips the header — where
    `G.R. No. 192393` is printed — out of the body. Without this line a citation
    lookup, which is one of the four things a researcher actually types, finds
    nothing at all. Then the caption, so a caption match is a match; then the
    majority opinion; then any separate opinions, each of which carries its own
    `Concurring Opinion — Justice` header from the parse stage, so a query for a
    dissenting justice still reaches the dissent.
    """
    parts = [record.get("gr_number", ""), record.get("title", ""), record.get("body", "")]
    parts.extend(record.get("separate_opinions") or ())
    return "\n\n".join(part for part in parts if part)


class InvertedIndex:
    """Built once, loaded once, queried many times."""

    def __init__(
        self,
        postings: dict[str, bytes],
        docs: list[DocMeta],
        source_path: Path,
        build_seconds: float = 0.0,
    ) -> None:
        self.postings = postings
        self.docs = docs
        self.source_path = source_path
        self.build_seconds = build_seconds
        self.doc_lengths = array("i", (doc.length for doc in docs))
        total = sum(self.doc_lengths)
        self.avgdl = total / len(docs) if docs else 0.0
        self.total_tokens = total
        self._df_cache: dict[str, int] = {}

    # -- build --------------------------------------------------------------

    @classmethod
    def build(cls, jsonl_path: Path) -> "InvertedIndex":
        started = time.perf_counter()

        # Postings accumulate straight into their encoded form. Holding
        # 15,000 documents' worth of Python lists of positions would cost well
        # over a gigabyte of object overhead for data that compresses to tens of
        # megabytes; encoding as we go keeps the build flat in memory.
        buffers: dict[str, bytearray] = defaultdict(bytearray)
        last_doc: dict[str, int] = {}
        docs: list[DocMeta] = []

        with jsonl_path.open("rb") as handle:
            offset = 0
            for line in handle:
                record = json.loads(line)
                doc_id = len(docs)

                occurrences: dict[str, list[int]] = defaultdict(list)
                length = 0
                for token in tokenize(_unit_text(record)):
                    occurrences[token.text].append(token.position)
                    length += 1

                for term, positions in occurrences.items():
                    buffer = buffers[term]
                    previous = last_doc.get(term, 0)
                    _append_posting(buffer, doc_id - previous, positions)
                    last_doc[term] = doc_id

                docs.append(
                    DocMeta(
                        doc_id=doc_id,
                        gr_number=record.get("gr_number", ""),
                        title=record.get("title", ""),
                        promulgated=record.get("promulgated"),
                        division=record.get("division"),
                        ponente=record.get("ponente"),
                        source_url=record.get("source_url", ""),
                        length=length,
                        offset=offset,
                    )
                )
                offset += len(line)

        postings = {term: bytes(buffer) for term, buffer in buffers.items()}
        return cls(
            postings=postings,
            docs=docs,
            source_path=jsonl_path,
            build_seconds=time.perf_counter() - started,
        )

    # -- persistence --------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": INDEX_VERSION,
            "postings": self.postings,
            "docs": self.docs,
            "source_path": str(self.source_path),
            "build_seconds": self.build_seconds,
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "InvertedIndex":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("version") != INDEX_VERSION:
            raise ValueError(
                f"index at {path} is version {payload.get('version')}, "
                f"expected {INDEX_VERSION} — rebuild it with `python -m indexer`"
            )
        return cls(
            postings=payload["postings"],
            docs=payload["docs"],
            source_path=Path(payload["source_path"]),
            build_seconds=payload.get("build_seconds", 0.0),
        )

    # -- lookups ------------------------------------------------------------

    @property
    def doc_count(self) -> int:
        return len(self.docs)

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)

    def doc_frequency(self, term: str) -> int:
        cached = self._df_cache.get(term)
        if cached is not None:
            return cached
        blob = self.postings.get(term)
        if not blob:
            return 0
        df = sum(1 for _ in iter_doc_freqs(blob))
        self._df_cache[term] = df
        return df

    def postings_for(self, term: str) -> list[tuple[int, int, list[int]]]:
        blob = self.postings.get(term)
        return decode_postings(blob) if blob else []

    def doc_freq_pairs(self, term: str) -> list[tuple[int, int]]:
        """`(doc_id, term_freq)` pairs — the form ranking wants."""
        blob = self.postings.get(term)
        return list(iter_doc_freqs(blob)) if blob else []

    def body(self, doc_id: int) -> str:
        """Read one document's text back from the parsed JSONL by byte offset."""
        meta = self.docs[doc_id]
        with self.source_path.open("rb") as handle:
            handle.seek(meta.offset)
            record = json.loads(handle.readline())
        return _unit_text(record)

    def tokens_of(self, doc_id: int) -> list[Token]:
        return scan(self.body(doc_id))

    # -- boolean primitives -------------------------------------------------

    def intersect(self, terms: list[str]) -> list[int]:
        """Documents containing every term. Shortest postings list first.

        Starting from the rarest term means the working set shrinks fastest, so
        the merge does the least total work.
        """
        if not terms:
            return []
        lists = [self.doc_freq_pairs(term) for term in terms]
        if any(not postings for postings in lists):
            return []
        lists.sort(key=len)

        result = [doc_id for doc_id, _ in lists[0]]
        for postings in lists[1:]:
            result = _merge_and(result, [doc_id for doc_id, _ in postings])
            if not result:
                break
        return result

    def union(self, terms: list[str]) -> list[int]:
        found: set[int] = set()
        for term in terms:
            found.update(doc_id for doc_id, _ in self.doc_freq_pairs(term))
        return sorted(found)

    def phrase(self, terms: list[str], *, gaps: list[int] | None = None) -> list[int]:
        return sorted(self.phrase_matches(terms, gaps=gaps))

    def phrase_matches(
        self, terms: list[str], *, gaps: list[int] | None = None
    ) -> dict[int, int]:
        """Documents where `terms` occur adjacently, mapped to the first match.

        `gaps` carries the position offsets the tokenizer recorded for the query,
        so a phrase containing a stopword still lines up: "grave abuse of
        discretion" indexes as positions 0, 1, 3, and the check below looks for
        exactly those offsets rather than 0, 1, 2.

        The match position is returned, not just the doc ID, because it is what
        the snippet should be centred on. Anchoring on the earliest occurrence of
        any single query term instead would routinely open the snippet on a
        stray `abuse` several pages before the phrase itself.
        """
        if not terms:
            return {}
        if len(terms) == 1:
            return {
                doc_id: positions[0]
                for doc_id, _, positions in self.postings_for(terms[0])
                if positions
            }

        offsets = gaps or list(range(len(terms)))
        base = offsets[0]
        offsets = [offset - base for offset in offsets]

        candidates = set(self.intersect(terms))
        if not candidates:
            return {}

        positions_by_term = [
            {
                doc_id: set(positions)
                for doc_id, _, positions in self.postings_for(term)
                if doc_id in candidates
            }
            for term in terms
        ]

        matches: dict[int, int] = {}
        for doc_id in sorted(candidates):
            anchors = positions_by_term[0].get(doc_id)
            if not anchors:
                continue
            for anchor in sorted(anchors):
                if all(
                    (anchor + offsets[i]) in positions_by_term[i].get(doc_id, ())
                    for i in range(1, len(terms))
                ):
                    matches[doc_id] = anchor
                    break
        return matches


def _append_posting(buffer: bytearray, doc_gap: int, positions: list[int]) -> None:
    """Encode one posting onto the end of a term's buffer.

    Written out rather than calling `encode_postings` for a one-element list:
    this runs once per (term, document) pair — hundreds of thousands of times
    per build — and the temporary list and bytes object per call were a
    measurable slice of build time.
    """
    write_varint(buffer, doc_gap)
    write_varint(buffer, len(positions))
    previous = 0
    for position in positions:
        write_varint(buffer, position - previous)
        previous = position


def _merge_and(left: list[int], right: list[int]) -> list[int]:
    """Linear merge of two sorted doc-ID lists — the reason postings are sorted."""
    result: list[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            result.append(left[i])
            i += 1
            j += 1
        elif left[i] < right[j]:
            i += 1
        else:
            j += 1
    return result
