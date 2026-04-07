from __future__ import annotations

import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd

logger = logging.getLogger(__name__)


class FilingCacheEntry(TypedDict):
    sections: dict[str, str]
    filing_date: str


class FactsCacheEntry(TypedDict):
    name: str
    facts_df: pd.DataFrame


class EdgarCache:
    def __init__(self, cache_dir: Path) -> None:
        self._filings_dir = cache_dir / "filings"
        self._facts_dir = cache_dir / "facts"

    def get_filing(self, ticker: str, year: int) -> FilingCacheEntry | None:
        path = self._filing_path(ticker, year)
        return self._load(path)

    def put_filing(self, ticker: str, year: int, data: FilingCacheEntry) -> None:
        path = self._filing_path(ticker, year)
        self._save(path, data)

    def get_facts(self, ticker: str) -> FactsCacheEntry | None:
        path = self._facts_path(ticker)
        return self._load(path)

    def put_facts(self, ticker: str, data: FactsCacheEntry) -> None:
        path = self._facts_path(ticker)
        self._save(path, data)

    def _filing_path(self, ticker: str, year: int) -> Path:
        return self._filings_dir / f"{ticker.upper()}_10K_{year}.pkl"

    def _facts_path(self, ticker: str) -> Path:
        return self._facts_dir / f"{ticker.upper()}.pkl"

    def _load(self, path: Path) -> Any | None:
        if not path.exists():
            return None
        try:
            with path.open("rb") as f:
                return pickle.load(f)  # noqa: S301
        except Exception:
            logger.warning("Corrupt cache file %s, ignoring", path)
            return None

    def _save(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(data, f)
            os.replace(tmp, path)
        except BaseException:
            os.unlink(tmp)
            raise
        logger.debug("Cached %s", path)
