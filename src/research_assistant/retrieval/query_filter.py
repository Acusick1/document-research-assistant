from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from qdrant_client.models import FieldCondition, Filter, MatchValue

from research_assistant.retrieval.vector_store import QdrantStore

logger = logging.getLogger(__name__)

FILTER_EXTRACTION_PROMPT = """\
You are a query preprocessor for a financial document search system. Your job is to extract \
structured entities from a user's natural language query.

Extract:
- **companies**: Any company names or ticker symbols mentioned. Output them exactly as they \
appear in the query (e.g. "Apple", "MSFT", "Alphabet"). Do not guess or infer companies that \
aren't mentioned.
- **years**: Any fiscal years mentioned. Extract as integers (e.g. "FY2023" -> 2023, \
"in 2024" -> 2024, "last year" -> leave empty unless the exact year is clear). \
Only extract years you are confident about.

If the query doesn't mention any companies or years, return empty lists.\
"""


class ExtractedEntities(BaseModel):
    companies: list[str] = Field(
        default_factory=list,
        description="Company names or ticker symbols as they appear in the query",
    )
    years: list[int] = Field(
        default_factory=list,
        description="Fiscal years mentioned in the query as integers",
    )


class QueryFilters(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    periods: list[str] = Field(default_factory=list)


class QueryFilterExtractor:
    def __init__(self, store: QdrantStore, model: str) -> None:
        self._agent: Agent[None, ExtractedEntities] = Agent(
            model,
            system_prompt=FILTER_EXTRACTION_PROMPT,
            output_type=ExtractedEntities,
        )
        self._name_to_ticker = self._build_name_mapping(store)
        self._valid_periods = self._load_valid_periods(store)

    def _build_name_mapping(self, store: QdrantStore) -> dict[str, str]:
        mapping: dict[str, str] = {}
        ticker_hits = store.get_field_values("ticker")
        tickers = [str(h.value) for h in ticker_hits]

        # One scroll per ticker to pair ticker → company_name (N+1, fine for small corpus)
        for ticker in tickers:
            mapping[ticker.lower()] = ticker

            points, _ = store.client.scroll(
                collection_name=store.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="ticker", match=MatchValue(value=ticker))]
                ),
                limit=1,
                with_payload=["company_name"],
            )
            if points and points[0].payload:
                name = points[0].payload.get("company_name", "")
                if name:
                    mapping[str(name).lower()] = ticker

        logger.info("Built name->ticker mapping: %s", mapping)
        return mapping

    def _load_valid_periods(self, store: QdrantStore) -> set[str]:
        hits = store.get_field_values("period")
        periods = {str(h.value) for h in hits}
        logger.info("Valid periods: %s", periods)
        return periods

    async def extract(self, query: str) -> dict[str, Any]:
        try:
            result = await self._agent.run(query)
        except Exception:
            logger.exception("Filter extraction failed, falling back to unfiltered: %r", query)
            return {}
        entities = result.output
        logger.debug("Extracted entities: %s", entities)

        filters = self._resolve(entities)
        logger.info("Query: %r -> filters: %s", query, filters)
        return self._to_qdrant_filters(filters)

    def _resolve(self, entities: ExtractedEntities) -> QueryFilters:
        tickers: list[str] = []
        for company in entities.companies:
            ticker = self._match_ticker(company)
            if ticker and ticker not in tickers:
                tickers.append(ticker)

        periods: list[str] = []
        for year in entities.years:
            period = f"FY{year}"
            if period in self._valid_periods and period not in periods:
                periods.append(period)

        return QueryFilters(tickers=tickers, periods=periods)

    def _match_ticker(self, company: str) -> str | None:
        key = company.strip().lower()

        if key in self._name_to_ticker:
            return self._name_to_ticker[key]

        # Substring match: extracted key must appear in a mapping key, longest match wins
        best_match: str | None = None
        best_len = 0
        for name, ticker in self._name_to_ticker.items():
            if key in name and len(name) > best_len:
                best_match = ticker
                best_len = len(name)
        if best_match:
            return best_match

        logger.warning("Could not resolve company %r to a ticker", company)
        return None

    def _to_qdrant_filters(self, filters: QueryFilters) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if len(filters.tickers) == 1:
            result["ticker"] = filters.tickers[0]
        elif len(filters.tickers) > 1:
            result["ticker"] = filters.tickers
        if len(filters.periods) == 1:
            result["period"] = filters.periods[0]
        elif len(filters.periods) > 1:
            result["period"] = filters.periods
        return result
