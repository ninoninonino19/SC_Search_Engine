"""Polite, resumable fetching with a cache and a manifest.

Three rules the rest of the crawler depends on:

1. **Never re-fetch.** If the file is already on disk we do not touch the
   network. You will re-parse the corpus dozens of times as you find edge cases;
   re-crawling for each pass is slow and rude.
2. **Never lose a failure.** Every network attempt appends a line to a JSONL
   manifest, so "what did we miss and why" is answerable later without a second
   crawl to find out.
3. **Raw bytes, not decoded text.** LawPhil declares windows-1252 and gets the
   declaration wrong on some pages. Decoding is a parser decision; the cache
   stores exactly what the server sent.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.robotparser import RobotFileParser

import requests

from crawl.config import (
    DEFAULT_DELAY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    ROBOTS_URL,
    USER_AGENT,
)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


@dataclass
class CrawlStats:
    """Counters for the numbers the milestone asks you to be able to quote."""

    attempted: int = 0
    fetched: int = 0
    cached: int = 0
    failed: int = 0
    disallowed: int = 0
    bytes_fetched: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        total = self.fetched + self.cached
        rate = (self.failed / self.attempted * 100) if self.attempted else 0.0
        return (
            f"attempted {self.attempted}, fetched {self.fetched}, "
            f"already cached {self.cached}, failed {self.failed} ({rate:.2f}%), "
            f"robots-disallowed {self.disallowed}, "
            f"{self.bytes_fetched / 1_048_576:.1f} MiB downloaded"
        )


class Fetcher:
    """One request at a time, one second apart, with backoff and a cache."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        delay: float = DEFAULT_DELAY,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        user_agent: str = USER_AGENT,
        obey_robots: bool = True,
    ) -> None:
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent
        self.manifest_path = manifest_path
        self.stats = CrawlStats()

        self._session = requests.Session()
        self._session.headers["User-Agent"] = user_agent
        self._last_request = 0.0
        self._robots = self._load_robots() if obey_robots else None

    # -- politeness ---------------------------------------------------------

    def _load_robots(self) -> RobotFileParser | None:
        """Read robots.txt once. A 4xx means no rules, which means allow all.

        Fetched with our own session so the request carries the same User-Agent
        as everything else, rather than urllib's default.
        """
        parser = RobotFileParser()
        parser.set_url(ROBOTS_URL)
        try:
            response = self._session.get(ROBOTS_URL, timeout=self.timeout)
        except requests.RequestException:
            # Cannot tell either way. Assume rules exist and stay conservative
            # only about the delay, not about refusing to crawl at all.
            parser.parse([])
            return parser
        if response.status_code == 200:
            parser.parse(response.text.splitlines())
        elif response.status_code in (401, 403):
            parser.disallow_all = True
        else:
            parser.allow_all = True
        return parser

    def allowed(self, url: str) -> bool:
        if self._robots is None:
            return True
        return self._robots.can_fetch(self.user_agent, url)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.monotonic()

    # -- fetching -----------------------------------------------------------

    def _record(self, url: str, path: Path | None, status: str, detail: str = "") -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "url": url,
            "path": str(path) if path else None,
            "status": status,
            "detail": detail,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def fetch(self, url: str, dest: Path) -> str:
        """Fetch `url` into `dest` unless it is already there.

        Returns one of "cached", "fetched", "failed", "disallowed". Never
        raises for an unreachable page: a crawl of 20,000 URLs that dies on the
        first bad one is not a crawl.
        """
        if dest.exists() and dest.stat().st_size > 0:
            self.stats.cached += 1
            return "cached"

        if not self.allowed(url):
            self.stats.disallowed += 1
            self._record(url, None, "disallowed")
            return "disallowed"

        self.stats.attempted += 1
        last_error = ""

        for attempt in range(self.max_retries):
            self._throttle()
            try:
                response = self._session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(response.content)
                    self.stats.fetched += 1
                    self.stats.bytes_fetched += len(response.content)
                    self._record(url, dest, "ok", str(len(response.content)))
                    return "fetched"

                last_error = f"HTTP {response.status_code}"
                # A 404 is an answer, not a hiccup. Retrying it wastes a second
                # of the site's time and gets the same reply.
                if response.status_code not in RETRYABLE_STATUS:
                    break

            if attempt < self.max_retries - 1:
                time.sleep(self.delay * (2**attempt))

        self.stats.failed += 1
        self.stats.failures.append((url, last_error))
        self._record(url, None, "failed", last_error)
        return "failed"
