"""Crawl-stage configuration.

Everything here is a knob you might reasonably turn from the command line or an
environment variable. Nothing here does I/O.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_URL = "https://lawphil.net/judjuris/"
ROBOTS_URL = "https://lawphil.net/robots.txt"

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "manifest.jsonl"

# The corpus window from the build plan. Widening it is a flag, not a code change.
DEFAULT_START_YEAR = 2010
DEFAULT_END_YEAR = 2026

# One request per second, no concurrency. The site does not owe us bandwidth.
DEFAULT_DELAY = 1.0
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 4

# A real contact address is the difference between a polite crawler and a
# blocked one. Set SC_SEARCH_CONTACT before a full run.
CONTACT = os.environ.get("SC_SEARCH_CONTACT", "unset-contact@example.invalid")
USER_AGENT = f"sc-search/0.1 (personal research crawler; contact: {CONTACT})"

# LawPhil serves windows-1252 and declares it inconsistently (one page in the
# sample declares `windows-1252!`). Raw bytes are cached as-is; this is the
# decode ladder the parse stage walks.
ENCODING_FALLBACKS = ("utf-8", "cp1252", "latin-1")
