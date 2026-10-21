"""Fetch and cache a small number of public RSS headlines."""

from __future__ import annotations

import threading
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import feedparser

from config import ARTICLES_PER_FEED, RSS_CACHE_SECONDS, RSS_FEEDS

_cache_lock = threading.Lock()
_cached_at = 0.0
_cached_articles: list[dict] = []
_cached_errors: list[str] = []


def clear_news_cache() -> None:
    """Force the next call to fetch RSS feeds again."""
    global _cached_at
    with _cache_lock:
        _cached_at = 0.0


def fetch_latest_news(force_refresh: bool = False) -> tuple[list[dict], list[str]]:
    """Fetch feeds with short timeouts and normalize optional RSS fields."""
    global _cached_at, _cached_articles, _cached_errors
    now = time.monotonic()
    with _cache_lock:
        if not force_refresh and now - _cached_at < RSS_CACHE_SECONDS:
            return list(_cached_articles), list(_cached_errors)

    articles: list[dict] = []
    errors: list[str] = []
    seen_titles: set[str] = set()
    for source, url in RSS_FEEDS.items():
        try:
            request = Request(url, headers={"User-Agent": "TruthLens/1.0 (student demo)"})
            with urlopen(request, timeout=8) as response:
                parsed = feedparser.parse(response.read())
            entries = parsed.entries[:ARTICLES_PER_FEED]
            for entry in entries:
                title = str(entry.get("title", "")).strip()
                link = str(entry.get("link", "")).strip()
                title_key = " ".join(title.casefold().split())
                if not title or title_key in seen_titles:
                    continue
                seen_titles.add(title_key)
                if not link.lower().startswith("https://"):
                    link = ""
                published = str(
                    entry.get("published", entry.get("updated", "Time not provided"))
                ).strip()
                articles.append({
                    "source": source,
                    "title": title,
                    "published": published or "Time not provided",
                    "link": link,
                })
            if getattr(parsed, "bozo", False) and not entries:
                errors.append(f"{source}: the feed returned unreadable data.")
        except (URLError, TimeoutError, OSError, ValueError) as error:
            errors.append(f"{source}: could not be reached ({error.__class__.__name__}).")
        except Exception:
            errors.append(f"{source}: the feed could not be read.")

    with _cache_lock:
        _cached_articles = articles
        _cached_errors = errors
        _cached_at = time.monotonic()
        return list(_cached_articles), list(_cached_errors)
