"""Turning cached bytes back into text.

The cache stores what the server sent. LawPhil sends windows-1252 and declares
it as `windows-1252` on some pages, `windows-1252!` on others, and occasionally
prefixes a UTF-8 BOM. Rather than trusting the declaration, walk a short ladder:
UTF-8 first (strict, so it fails loudly on 1252 bytes), then cp1252, then
latin-1, which cannot fail and so terminates the ladder.
"""

from __future__ import annotations

from crawl.config import ENCODING_FALLBACKS


def decode_html(raw: bytes) -> str:
    for encoding in ENCODING_FALLBACKS:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        return text.lstrip("\ufeff")
    return raw.decode("latin-1").lstrip("\ufeff")
