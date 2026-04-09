from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from research_assistant.retrieval.query_filter import (
    ExtractedEntities,
    QueryFilterExtractor,
    QueryFilters,
    YearRange,
)


class TestYearRangeExpand:
    @pytest.fixture
    def valid_years(self) -> set[int]:
        return {2022, 2023, 2024, 2025}

    def test_closed_range(self, valid_years: set[int]) -> None:
        assert YearRange(start=2022, end=2024).expand(valid_years) == [2022, 2023, 2024]

    def test_single_year(self, valid_years: set[int]) -> None:
        assert YearRange(start=2024, end=2024).expand(valid_years) == [2024]

    def test_open_end(self, valid_years: set[int]) -> None:
        assert YearRange(start=2024, end=None).expand(valid_years) == [2024, 2025]

    def test_open_start(self, valid_years: set[int]) -> None:
        assert YearRange(start=None, end=2023).expand(valid_years) == [2022, 2023]

    def test_both_open(self, valid_years: set[int]) -> None:
        assert YearRange(start=None, end=None).expand(valid_years) == [2022, 2023, 2024, 2025]

    def test_start_after_end(self, valid_years: set[int]) -> None:
        assert YearRange(start=2025, end=2022).expand(valid_years) == []

    def test_empty_valid_years(self) -> None:
        assert YearRange(start=2022, end=2024).expand(set()) == [2022, 2023, 2024]

    def test_open_end_empty_valid_years(self) -> None:
        assert YearRange(start=2022, end=None).expand(set()) == []

    def test_range_extends_beyond_valid(self, valid_years: set[int]) -> None:
        assert YearRange(start=2024, end=2027).expand(valid_years) == [2024, 2025, 2026, 2027]


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
        store = MagicMock()
        store.get_field_values.return_value = [
            MagicMock(value=y) for y in [2022, 2023, 2024, 2025]
        ]
        store.get_latest_fiscal_year.return_value = 2025
        extractor._store = store
        return extractor

    def test_single_ticker_single_year(self, extractor: QueryFilterExtractor) -> None:
        entities = ExtractedEntities(
            companies=["Apple"], year_range=YearRange(start=2024, end=2024),
        )
        result = extractor._resolve(entities)
        assert result == QueryFilters(tickers=["AAPL"], fiscal_years=[2024])

    def test_single_ticker_by_symbol(self, extractor: QueryFilterExtractor) -> None:
        entities = ExtractedEntities(
            companies=["MSFT"], year_range=YearRange(start=2023, end=2023),
        )
        result = extractor._resolve(entities)
        assert result == QueryFilters(tickers=["MSFT"], fiscal_years=[2023])

    def test_multiple_tickers(self, extractor: QueryFilterExtractor) -> None:
        entities = ExtractedEntities(
            companies=["Apple", "Microsoft"], year_range=YearRange(start=2024, end=2024),
        )
        result = extractor._resolve(entities)
        assert result == QueryFilters(tickers=["AAPL", "MSFT"], fiscal_years=[2024])

    def test_company_name_with_suffix(self, extractor: QueryFilterExtractor) -> None:
        entities = ExtractedEntities(companies=["Apple Inc."])
        result = extractor._resolve(entities)
        assert result.tickers == ["AAPL"]

    def test_unknown_company_skipped(self, extractor: QueryFilterExtractor) -> None:
        entities = ExtractedEntities(
            companies=["Tesla"], year_range=YearRange(start=2024, end=2024),
        )
        result = extractor._resolve(entities)
        assert result.tickers == []
        assert result.fiscal_years == [2024]

    def test_empty_entities(self, extractor: QueryFilterExtractor) -> None:
        entities = ExtractedEntities()
        result = extractor._resolve(entities)
        assert result == QueryFilters(tickers=[], fiscal_years=[])

    def test_deduplicates_tickers(self, extractor: QueryFilterExtractor) -> None:
        entities = ExtractedEntities(companies=["AAPL", "Apple"])
        result = extractor._resolve(entities)
        assert result.tickers == ["AAPL"]

    def test_year_range_fills_gap(self, extractor: QueryFilterExtractor) -> None:
        entities = ExtractedEntities(
            companies=["AAPL"], year_range=YearRange(start=2022, end=2024),
        )
        result = extractor._resolve(entities)
        assert result.fiscal_years == [2022, 2023, 2024]

    def test_year_range_open_ended(self, extractor: QueryFilterExtractor) -> None:
        entities = ExtractedEntities(
            companies=["AAPL"], year_range=YearRange(start=2023, end=None),
        )
        result = extractor._resolve(entities)
        assert result.fiscal_years == [2023, 2024, 2025]

    def test_year_range_open_start(self, extractor: QueryFilterExtractor) -> None:
        entities = ExtractedEntities(
            companies=["AAPL"], year_range=YearRange(start=None, end=2023),
        )
        result = extractor._resolve(entities)
        assert result.fiscal_years == [2022, 2023]

    def test_latest_resolves_per_ticker(self, extractor: QueryFilterExtractor) -> None:
        extractor._store.get_latest_fiscal_year.side_effect = (
            lambda t: 2025 if t == "AAPL" else 2026
        )
        entities = ExtractedEntities(companies=["Apple", "NVDA"], latest=True)
        result = extractor._resolve(entities)
        assert result.fiscal_years == [2025, 2026]

    def test_latest_without_ticker_no_fiscal_year(
        self, extractor: QueryFilterExtractor,
    ) -> None:
        entities = ExtractedEntities(latest=True)
        result = extractor._resolve(entities)
        assert result.fiscal_years == []

    def test_latest_ignored_when_year_range_present(
        self, extractor: QueryFilterExtractor,
    ) -> None:
        entities = ExtractedEntities(
            companies=["AAPL"], year_range=YearRange(start=2023, end=2023), latest=True,
        )
        result = extractor._resolve(entities)
        assert result.fiscal_years == [2023]
        extractor._store.get_latest_fiscal_year.assert_not_called()

    def test_alphabet_matches_googl(self, extractor: QueryFilterExtractor) -> None:
        entities = ExtractedEntities(companies=["Alphabet"])
        result = extractor._resolve(entities)
        assert result.tickers == ["GOOGL"]

    def test_google_matches_via_substring(self, extractor: QueryFilterExtractor) -> None:
        entities = ExtractedEntities(companies=["Google"])
        result = extractor._resolve(entities)
        assert result.tickers == ["GOOGL"]


class TestExtract:
    @pytest.fixture
    def extractor(self) -> QueryFilterExtractor:
        extractor = object.__new__(QueryFilterExtractor)
        extractor._name_to_ticker = {"aapl": "AAPL"}
        store = MagicMock()
        store.get_field_values.return_value = [MagicMock(value=2025)]
        extractor._store = store
        extractor._agent = AsyncMock()
        return extractor

    @pytest.mark.anyio
    async def test_rejected_query(self, extractor: QueryFilterExtractor) -> None:
        mock_result = MagicMock()
        mock_result.output = ExtractedEntities(
            reject_reason="Question is outside the scope of SEC financial filings.",
        )
        extractor._agent.run.return_value = mock_result

        result = await extractor.extract("What is the capital of France?")
        assert result.reject_reason == "Question is outside the scope of SEC financial filings."
        assert result.filters == {}

    @pytest.mark.anyio
    async def test_valid_query(self, extractor: QueryFilterExtractor) -> None:
        mock_result = MagicMock()
        mock_result.output = ExtractedEntities(
            companies=["AAPL"], year_range=YearRange(start=2025, end=2025),
        )
        extractor._agent.run.return_value = mock_result

        result = await extractor.extract("What was Apple's revenue in FY2025?")
        assert result.reject_reason is None
        assert result.filters == {"ticker": "AAPL", "fiscal_year": 2025}

    @pytest.mark.anyio
    async def test_agent_failure_returns_empty(
        self, extractor: QueryFilterExtractor,
    ) -> None:
        extractor._agent.run.side_effect = RuntimeError("LLM error")

        result = await extractor.extract("some query")
        assert result.reject_reason is None
        assert result.filters == {}


class TestToQdrantFilters:
    @pytest.fixture
    def extractor(self) -> QueryFilterExtractor:
        return object.__new__(QueryFilterExtractor)

    def test_single_ticker_single_year(self, extractor: QueryFilterExtractor) -> None:
        filters = QueryFilters(tickers=["AAPL"], fiscal_years=[2024])
        result = extractor._to_qdrant_filters(filters)
        assert result == {"ticker": "AAPL", "fiscal_year": 2024}

    def test_multiple_tickers(self, extractor: QueryFilterExtractor) -> None:
        filters = QueryFilters(tickers=["AAPL", "MSFT"], fiscal_years=[2024])
        result = extractor._to_qdrant_filters(filters)
        assert result == {"ticker": ["AAPL", "MSFT"], "fiscal_year": 2024}

    def test_no_filters(self, extractor: QueryFilterExtractor) -> None:
        filters = QueryFilters(tickers=[], fiscal_years=[])
        result = extractor._to_qdrant_filters(filters)
        assert result == {}

    def test_multiple_years(self, extractor: QueryFilterExtractor) -> None:
        filters = QueryFilters(tickers=["AAPL"], fiscal_years=[2023, 2024])
        result = extractor._to_qdrant_filters(filters)
        assert result == {"ticker": "AAPL", "fiscal_year": [2023, 2024]}
