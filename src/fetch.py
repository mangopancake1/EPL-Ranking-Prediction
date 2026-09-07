"""
Shared HTTP layer: on-disk cache, per-host rate limiting, 429 backoff.

Every scraper goes through get(). Re-running the pipeline costs nothing
after the first pass, which matters because FBref and Transfermarkt both
throttle hard and a full crawl is hundreds of pages.
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

import config

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_last_request: dict[str, float] = {}

# FBref buries most of its tables inside HTML comments to keep them out of
# naive scrapers. Uncommenting the document is the whole trick.
_COMMENT_RE = re.compile(r"<!--(.*?)-->", re.DOTALL)


def _host_key(url: str) -> str:
    host = urlparse(url).netloc.lower()
    for key in config.REQUEST_DELAY:
        if key.replace("-", "") in host.replace("-", "").replace(".", ""):
            return key
    return "default"


def _cache_path(url: str) -> Path:
    digest = hashlib.sha256(url.encode()).hexdigest()[:20]
    host = urlparse(url).netloc.replace(".", "_")
    return config.CACHE / f"{host}_{digest}.html"


def _throttle(url: str) -> None:
    key = _host_key(url)
    delay = config.REQUEST_DELAY.get(key, 2.0)
    elapsed = time.monotonic() - _last_request.get(key, 0.0)
    if elapsed < delay:
        time.sleep(delay - elapsed)
    _last_request[key] = time.monotonic()


def get(
    url: str,
    *,
    headers: dict | None = None,
    use_cache: bool = True,
    max_retries: int = 4,
) -> str:
    """Fetch a URL as text. Cached hits skip the network entirely."""
    path = _cache_path(url)

    if use_cache and path.exists():
        age_days = (time.time() - path.stat().st_mtime) / 86400
        if age_days < config.CACHE_TTL_DAYS:
            return path.read_text(encoding="utf-8", errors="replace")

    req_headers = {"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"}
    if headers:
        req_headers.update(headers)

    backoff = 10.0
    for attempt in range(max_retries):
        _throttle(url)
        try:
            resp = requests.get(url, headers=req_headers, timeout=30)
        except requests.RequestException as exc:
            if attempt == max_retries - 1:
                raise
            print(f"  ! {type(exc).__name__} on {url}, retry in {backoff:.0f}s")
            time.sleep(backoff)
            backoff *= 2
            continue

        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", backoff))
            print(f"  ! 429 rate limited on {url}, waiting {wait:.0f}s")
            time.sleep(wait)
            backoff *= 2
            continue

        if resp.status_code == 404:
            raise FileNotFoundError(f"404: {url}")

        resp.raise_for_status()
        text = resp.text
        if use_cache:
            path.write_text(text, encoding="utf-8")
        return text

    raise RuntimeError(f"Gave up on {url} after {max_retries} attempts")


def get_json(url: str, *, headers: dict | None = None, use_cache: bool = True):
    """Fetch and parse JSON, sharing the same cache and throttle."""
    import json

    return json.loads(get(url, headers=headers, use_cache=use_cache))


def uncomment(html: str) -> str:
    """Expose FBref tables hidden inside HTML comments."""
    return _COMMENT_RE.sub(lambda m: m.group(1), html)


def clear_cache(pattern: str = "*") -> int:
    """Delete cached pages. Returns how many were removed."""
    files = list(config.CACHE.glob(f"{pattern}.html" if pattern != "*" else "*.html"))
    for f in files:
        f.unlink()
    return len(files)


if __name__ == "__main__":
    # Smoke test: the cache must actually prevent a second network call.
    import sys

    url = "https://fbref.com/en/comps/9/Premier-League-Stats"
    t0 = time.monotonic()
    html = get(url)
    first = time.monotonic() - t0

    t0 = time.monotonic()
    html2 = get(url)
    second = time.monotonic() - t0

    assert html == html2, "cache returned different content"
    assert second < first, f"cache not faster: {second:.2f}s vs {first:.2f}s"
    assert "Premier League" in html, "unexpected page content"
    assert len(uncomment(html)) >= len(html), "uncomment shrank the document"
    print(f"OK  fetch {len(html):,} bytes in {first:.2f}s, cached in {second:.3f}s")
    sys.exit(0)
