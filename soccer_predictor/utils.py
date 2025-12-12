from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class HttpCache:
    cache_dir: Path
    min_delay_s: float = 3.0
    timeout_s: float = 30.0
    user_agent: str = DEFAULT_UA
    use_playwright: bool = False

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_fetch_ts: float = 0.0
        self._session = requests.Session()

    def _sleep_if_needed(self) -> None:
        elapsed = time.time() - self._last_fetch_ts
        if elapsed < self.min_delay_s:
            time.sleep(self.min_delay_s - elapsed)

    def get(self, url: str, force: bool = False) -> str:
        """Fetch URL with on-disk cache and polite rate limiting."""
        key = _sha256(url)
        path = self.cache_dir / f"{key}.html"

        if path.exists() and not force:
            return path.read_text(encoding="utf-8", errors="ignore")

        self._sleep_if_needed()

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://fbref.com/",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }

        resp = self._session.get(url, headers=headers, timeout=self.timeout_s)

        # FBref is often protected by Cloudflare; in some environments you may get a challenge (403).
        if resp.status_code == 403 and self.use_playwright:
            from .browser import fetch_html_with_playwright

            html = fetch_html_with_playwright(url, user_agent=self.user_agent, timeout_s=self.timeout_s)
        else:
            resp.raise_for_status()
            html = resp.text

        path.write_text(html, encoding="utf-8")
        self._last_fetch_ts = time.time()
        return html


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
