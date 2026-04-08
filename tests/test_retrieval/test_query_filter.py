from __future__ import annotations

import pytest

from research_assistant.retrieval.query_filter import (
    ExtractedEntities,
    QueryFilterExtractor,
    QueryFilters,
)


class TestResolve:
    @pytest.fixture
    def extractor(self) -> QueryFilterExtractor:
        extractor = object.__new__(QueryFilterExtractor)
        extractor._name_to_ticker = {
            "aapl": "AAPL",
            "apple inc.": "AAPL",
            "msft": "MSFT",
            "microsoft corp": "MSFT",
            "googl": "GOOGL",
            "alphabet inc.": "GOOGL",
            "meta": "META",
            "meta platforms, inc.": "META",
            "amzn": "AMZN",
            "amazon com inc": "AMZN",
            "nvda": "NVDA",
            "nvidia corp": "NVDA",
        }
        extractor._valid_periods = {"FY2022", "FY2023", "FY2024", "FY2025"}
        return extractor

    def test_single_ticker_by_name(self, extractor: QueryFilterExtractor) -> None:
        entities = ExtractedEntities(companies=["Apple"], years=[2024])
        result = extractor._resolve(entities)
        assert result == QueryFilters(tickers=["AAPL"], periods=["FY2024"])

    def test_single_ticker_by_symbol(self, extractor: QueryFilterExtractor) -> None:
        entities = ExtractedEntities(companies=["MSFT"], years=[2023])
        result = extractor._resolve(entities)
        assert result == QueryFilters(tickers=["MSFT"], periods=["FY2023"])

    def test_multiple_tickers(self, extractor: QueryFilterExtractor) -> None:
        entities = ExtractedEntities(companies=["Apple", "Microsoft"], years=[2024])
        result = extractor._resolve(entities)
        assert result == QueryFilters(tickers=["AAPL", "MSFT"], periods=["FY2024"])

    def test_company_name_with_suffix(self, extractor: QueryFilterExtractor) -> None:
        entities = ExtractedEntities(companies=["Apple Inc."], years=[])
        result = extractor._resolve(entities)
        assert result.tickers == ["AAPL"]

    def test_unknown_company_skipped(self, extractor: QueryFilterExtractor) -> None:
        entities = ExtractedEntities(companies=["Tesla"], years=[2024])
        result = extractor._resolve(entities)
        assert result.tickers == []
        assert result.periods == ["FY2024"]

    def test_invalid_period_skipped(self, extractor: QueryFilterExtractor) -> None:
        entities = ExtractedEntities(companies=["AAPL"], years=[2020])
        result = extractor._resolve(entities)
        assert result.tickers == ["AAPL"]
        assert result.periods == []

    def test_empty_entities(self, extractor: QueryFilterExtractor) -> None:
        entities = ExtractedEntities(companies=[], years=[])
        result = extractor._resolve(entities)
        assert result == QueryFilters(tickers=[], periods=[])

    def test_deduplicates_tickers(self, extractor: QueryFilterExtractor) -> None:
        entities = ExtractedEntities(companies=["AAPL", "Apple"], years=[])
        result = extractor._resolve(entities)
        assert result.tickers == ["AAPL"]

    def test_alphabet_matches_googl(self, extractor: QueryFilterExtractor) -> None:
        entities = ExtractedEntities(companies=["Alphabet"], years=[])
        result = extractor._resolve(entities)
        assert result.tickers == ["GOOGL"]

    def test_google_matches_via_substring(self, extractor: QueryFilterExtractor) -> None:
        entities = ExtractedEntities(companies=["Google"], years=[])
        result = extractor._resolve(entities)
        assert result.tickers == ["GOOGL"]


class TestToQdrantFilters:
    @pytest.fixture
    def extractor(self) -> QueryFilterExtractor:
        extractor = object.__new__(QueryFilterExtractor)
        return extractor

    def test_single_ticker_single_period(self, extractor: QueryFilterExtractor) -> None:
        filters = QueryFilters(tickers=["AAPL"], periods=["FY2024"])
        result = extractor._to_qdrant_filters(filters)
        assert result == {"ticker": "AAPL", "period": "FY2024"}

    def test_multiple_tickers(self, extractor: QueryFilterExtractor) -> None:
        filters = QueryFilters(tickers=["AAPL", "MSFT"], periods=["FY2024"])
        result = extractor._to_qdrant_filters(filters)
        assert result == {"ticker": ["AAPL", "MSFT"], "period": "FY2024"}

    def test_no_filters(self, extractor: QueryFilterExtractor) -> None:
        filters = QueryFilters(tickers=[], periods=[])
        result = extractor._to_qdrant_filters(filters)
        assert result == {}

    def test_multiple_periods(self, extractor: QueryFilterExtractor) -> None:
        filters = QueryFilters(tickers=["AAPL"], periods=["FY2023", "FY2024"])
        result = extractor._to_qdrant_filters(filters)
        assert result == {"ticker": "AAPL", "period": ["FY2023", "FY2024"]}
