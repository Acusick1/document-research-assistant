from __future__ import annotations

from pathlib import Path

from diskcache import Cache


def create_cache(cache_dir: Path) -> Cache:
    return Cache(str(cache_dir))


def filing_key(ticker: str, year: int) -> str:
    return f"filing:{ticker.upper()}:10K:{year}"


def facts_key(ticker: str) -> str:
    return f"facts:{ticker.upper()}"
