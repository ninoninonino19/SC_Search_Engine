"""Postings-list compression: delta encoding plus varints.

A postings list is a sorted run of increasing doc IDs, each with a sorted run of
increasing positions. Two facts follow from "sorted and increasing":

* The **gaps** are far smaller than the values. Doc ID 14,203 needs 14 bits;
  the gap from 14,198 needs 3. Storing gaps instead of absolute values is free
  compression that costs one addition when reading.
* Small numbers should cost few bytes. A varint spends 7 bits of payload per
  byte and uses the top bit as a continuation flag, so a gap of 5 costs one byte
  instead of the four a fixed-width integer would spend.

Positions are what make this matter. They are the bulk of the index — roughly
one entry per token in the corpus — and they are exactly the values that delta
encoding shrinks the most, because consecutive occurrences of a term inside one
document are often a handful of tokens apart.

Layout, per posting, all varints:

    doc_gap, term_freq, position_gap * term_freq

where `doc_gap` is relative to the previous posting in the list and the first
`position_gap` is the absolute first position.
"""

from __future__ import annotations

Posting = tuple[int, int, list[int]]  # (doc_id, term_freq, positions)


def write_varint(buffer: bytearray, value: int) -> None:
    """Append `value` as an unsigned LEB128 varint."""
    while value >= 0x80:
        buffer.append((value & 0x7F) | 0x80)
        value >>= 7
    buffer.append(value)


def read_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Return `(value, next_offset)`."""
    result = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, offset
        shift += 7


def encode_postings(postings: list[tuple[int, list[int]]]) -> bytes:
    """Encode `(doc_id, positions)` pairs. Input must be sorted by doc_id."""
    buffer = bytearray()
    previous_doc = 0
    for doc_id, positions in postings:
        write_varint(buffer, doc_id - previous_doc)
        previous_doc = doc_id
        write_varint(buffer, len(positions))
        previous_position = 0
        for position in positions:
            write_varint(buffer, position - previous_position)
            previous_position = position
    return bytes(buffer)


def decode_postings(blob: bytes) -> list[Posting]:
    """Full decode, positions included. Use for phrase queries and snippets."""
    postings: list[Posting] = []
    offset = 0
    doc_id = 0
    length = len(blob)

    while offset < length:
        gap, offset = read_varint(blob, offset)
        doc_id += gap
        term_freq, offset = read_varint(blob, offset)

        positions: list[int] = []
        position = 0
        for _ in range(term_freq):
            position_gap, offset = read_varint(blob, offset)
            position += position_gap
            positions.append(position)

        postings.append((doc_id, term_freq, positions))

    return postings


def iter_doc_freqs(blob: bytes):
    """Yield `(doc_id, term_freq)` without materialising positions.

    Ranking never looks at positions — BM25 needs only `tf` and `|d|` — and this
    is the hot loop, run once per query term per matching document. Skipping the
    position lists here rather than building lists to throw away is most of the
    difference between a fast scorer and a slow one.
    """
    offset = 0
    doc_id = 0
    length = len(blob)

    while offset < length:
        gap, offset = read_varint(blob, offset)
        doc_id += gap
        term_freq, offset = read_varint(blob, offset)
        yield doc_id, term_freq
        for _ in range(term_freq):
            while blob[offset] >= 0x80:
                offset += 1
            offset += 1


def posting_count(blob: bytes) -> int:
    return sum(1 for _ in iter_doc_freqs(blob))
