from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TypedDict, runtime_checkable

from diskcache import Cache


class FilingCacheEntry(TypedDict):
    sections: dict[str, str]
    filing_date: str
    company_name: str


class FactsCacheEntry(TypedDict):
    name: str
    facts_columns: dict[str, list[Any]]


@runtime_checkable
class EdgarCache(Protocol):
    def get(self, key: str, default: Any = ...) -> Any: ...
    def set(self, key: str, value: Any) -> Any: ...


def create_cache(cache_dir: Path) -> EdgarCache:
    return Cache(str(cache_dir))


def filing_key(ticker: str, year: int) -> str:
    return f"filing:{ticker.upper()}:10K:{year}"


def facts_key(ticker: str) -> str:
    return f"facts:{ticker.upper()}"
