from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research_assistant.corpus.edgar.cache import EdgarCache, FactsCacheEntry, FilingCacheEntry


class TestEdgarCache:
    @pytest.fixture()
    def cache(self, tmp_path: Path) -> EdgarCache:
        return EdgarCache(tmp_path / "cache")

    def test_filing_cache_miss(self, cache: EdgarCache) -> None:
        assert cache.get_filing("AAPL", 2024) is None

    def test_filing_cache_hit(self, cache: EdgarCache) -> None:
        data: FilingCacheEntry = {"sections": {"Item 1": "text"}, "filing_date": "2024-11-01"}
        cache.put_filing("AAPL", 2024, data)
        result = cache.get_filing("AAPL", 2024)
        assert result == data

    def test_facts_cache_miss(self, cache: EdgarCache) -> None:
        assert cache.get_facts("AAPL") is None

    def test_facts_cache_hit(self, cache: EdgarCache) -> None:
        df = pd.DataFrame({"col": [1, 2, 3]})
        entry: FactsCacheEntry = {"name": "Apple Inc.", "facts_df": df}
        cache.put_facts("AAPL", entry)
        result = cache.get_facts("AAPL")
        assert result is not None
        assert result["name"] == "Apple Inc."
        pd.testing.assert_frame_equal(result["facts_df"], df)

    def test_corrupt_file_returns_none(self, cache: EdgarCache) -> None:
        path = cache._filing_path("AAPL", 2024)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not valid pickle data")
        assert cache.get_filing("AAPL", 2024) is None

    def test_creates_directories(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        cache = EdgarCache(nested)
        data: FilingCacheEntry = {"sections": {}, "filing_date": "2024-01-01"}
        cache.put_filing("MSFT", 2024, data)
        assert cache.get_filing("MSFT", 2024) == data

    def test_ticker_normalized_to_uppercase(self, cache: EdgarCache) -> None:
        data: FilingCacheEntry = {"sections": {}, "filing_date": "2024-01-01"}
        cache.put_filing("aapl", 2024, data)
        assert cache.get_filing("AAPL", 2024) == data
