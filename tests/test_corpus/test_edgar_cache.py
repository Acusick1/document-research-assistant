from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from diskcache import Cache

from research_assistant.corpus.edgar.cache import create_cache, facts_key, filing_key


class TestCacheKeys:
    def test_filing_key_normalizes_ticker(self) -> None:
        assert filing_key("aapl", 2024) == "filing:AAPL:10K:2024"

    def test_facts_key_normalizes_ticker(self) -> None:
        assert facts_key("aapl") == "facts:AAPL"


class TestEdgarCache:
    @pytest.fixture()
    def cache(self, tmp_path: Path) -> Cache:
        return create_cache(tmp_path / "cache")

    def test_filing_miss(self, cache: Cache) -> None:
        key = filing_key("AAPL", 2024)
        assert cache.get(key) is None

    def test_filing_roundtrip(self, cache: Cache) -> None:
        key = filing_key("AAPL", 2024)
        data = {"sections": {"Item 1": "text"}, "filing_date": "2024-11-01"}
        cache.set(key, data)
        assert cache.get(key) == data

    def test_facts_roundtrip(self, cache: Cache) -> None:
        key = facts_key("AAPL")
        df = pd.DataFrame({"col": [1, 2, 3]})
        entry = {"name": "Apple Inc.", "facts_df": df}
        cache.set(key, entry)
        result = cache.get(key)
        assert result is not None
        assert result["name"] == "Apple Inc."
        pd.testing.assert_frame_equal(result["facts_df"], df)

    def test_ticker_normalization(self, cache: Cache) -> None:
        cache.set(filing_key("aapl", 2024), {"sections": {}, "filing_date": "2024-01-01"})
        assert cache.get(filing_key("AAPL", 2024)) is not None
