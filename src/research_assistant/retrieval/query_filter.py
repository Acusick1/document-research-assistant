from __future__ import annotations

import logging
from datetime import date

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from qdrant_client.http.models.models import Condition
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from research_assistant.retrieval.vector_store import QdrantStore

logger = logging.getLogger(__name__)

FILTER_EXTRACTION_PROMPT_TEMPLATE = """\
You are a query preprocessor for a financial document search system. Your job is to extract \
structured entities from a user's natural language query.

The current fiscal year is {current_year}.

Extract:
- **companies**: Any company names or ticker symbols mentioned. Output them exactly as they \
appear in the query (e.g. "Apple", "MSFT", "Alphabet"). Do not guess or infer companies that \
aren't mentioned.
- **year_range**: The fiscal year or years the query refers to, expressed as a range with \
start and/or end. Both are inclusive. Examples:
  - "FY2023" / "in 2024" -> start=2023, end=2023 (or 2024)
  - "FY2022 to FY2024" -> start=2022, end=2024
  - "FY2022 and FY2024" -> start=2022, end=2024 (fills gap — includes 2023)
  - "since 2022" / "from 2022 onwards" -> start=2022, end=null
  - "up to 2024" / "before 2025" -> start=null, end=2024
  - "last 3 years" / "past 3 years" -> start={current_year} - 3 + 1, end={current_year}
  - "over the past 5 years" -> start={current_year} - 5 + 1, end={current_year}
  Leave null if no year information is mentioned or implied.
- **latest**: Set to true if the query asks for the most recent data without specifying a year \
(e.g. "latest revenue", "most recent filing", "current net income"). \
Do not set latest if a year_range is provided.
- **reject_reason**: If the query is not a valid financial question, set this to a short \
explanation. Examples of invalid queries:
  - Gibberish or random text -> "Query is not a valid question."
  - Off-topic (not about financial data) -> "Question is outside the scope of financial filings."
  - Completely ambiguous with no actionable intent -> "Query is too vague to process."
  Leave null for valid financial queries, even if the answer might not be in the corpus.\
"""


class YearRange(BaseModel):
    start: int | None = Field(
        default=None, description="Start year of the range (inclusive)",
    )
    end: int | None = Field(
        default=None, description="End year of the range (inclusive)",
    )

    def expand(self, valid_years: set[int]) -> list[int]:
        lo = self.start if self.start is not None else min(valid_years, default=None)
        hi = self.end if self.end is not None else max(valid_years, default=None)
        if lo is None or hi is None or lo > hi:
            return []
        return list(range(lo, hi + 1))


class ExtractedEntities(BaseModel):
    companies: list[str] = Field(
        default_factory=list,
        description="Company names or ticker symbols as they appear in the query",
    )
    year_range: YearRange | None = Field(
        default=None,
        description="Fiscal year(s) as a range (start and end inclusive, either may be null)",
    )
    latest: bool = Field(
        default=False,
        description="True if the query asks for the most recent data without a specific year",
    )
    reject_reason: str | None = Field(
        default=None,
        description="Reason the query was rejected, or null if valid",
    )


class QueryFilters(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    fiscal_years: list[int] = Field(default_factory=list)

    def to_qdrant_filter(self) -> Filter | None:
        conditions: list[Condition] = []
        if len(self.tickers) == 1:
            conditions.append(
                FieldCondition(key="ticker", match=MatchValue(value=self.tickers[0])),
            )
        elif len(self.tickers) > 1:
            conditions.append(
                FieldCondition(key="ticker", match=MatchAny(any=self.tickers)),
            )
        if len(self.fiscal_years) == 1:
            conditions.append(
                FieldCondition(
                    key="fiscal_year", match=MatchValue(value=self.fiscal_years[0]),
                ),
            )
        elif len(self.fiscal_years) > 1:
            conditions.append(
                FieldCondition(
                    key="fiscal_year", match=MatchAny(any=self.fiscal_years),
                ),
            )
        return Filter(must=conditions) if conditions else None


class FilterResult(BaseModel):
    qdrant_filter: Filter | None = None
    reject_reason: str | None = None


class QueryFilterExtractor:
    def __init__(self, store: QdrantStore, model: str) -> None:
        prompt = FILTER_EXTRACTION_PROMPT_TEMPLATE.format(current_year=date.today().year)
        self._agent: Agent[None, ExtractedEntities] = Agent(
            model,
            system_prompt=prompt,
            output_type=ExtractedEntities,
        )
        self._store = store
        self._name_to_ticker = self._build_name_mapping(store)

    def _build_name_mapping(self, store: QdrantStore) -> dict[str, str]:
        mapping: dict[str, str] = {}
        ticker_hits = store.get_field_values("ticker")
        tickers = [str(h.value) for h in ticker_hits]

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

    async def extract(self, query: str) -> FilterResult:
        try:
            result = await self._agent.run(query)
        except Exception:
            logger.exception("Filter extraction failed, falling back to unfiltered: %r", query)
            return FilterResult()
        entities = result.output
        logger.debug("Extracted entities: %s", entities)

        if entities.reject_reason:
            logger.info("Query rejected: %r -> %s", query, entities.reject_reason)
            return FilterResult(reject_reason=entities.reject_reason)

        query_filters = self._resolve(entities)
        logger.info("Query: %r -> filters: %s", query, query_filters)
        return FilterResult(qdrant_filter=query_filters.to_qdrant_filter())

    def _resolve(self, entities: ExtractedEntities) -> QueryFilters:
        tickers: list[str] = []
        for company in entities.companies:
            ticker = self._match_ticker(company)
            if ticker and ticker not in tickers:
                tickers.append(ticker)

        fiscal_years: set[int] = set()

        if entities.year_range:
            valid_years = self._get_valid_years()
            fiscal_years.update(entities.year_range.expand(valid_years))
        elif entities.latest and tickers:
            for ticker in tickers:
                latest = self._store.get_latest_fiscal_year(ticker)
                if latest is not None:
                    fiscal_years.add(latest)

        return QueryFilters(tickers=tickers, fiscal_years=sorted(fiscal_years))

    def _get_valid_years(self) -> set[int]:
        hits = self._store.get_field_values("fiscal_year")
        return {int(h.value) for h in hits}

    def _match_ticker(self, company: str) -> str | None:
        key = company.strip().lower()

        if key in self._name_to_ticker:
            return self._name_to_ticker[key]

        best_match: str | None = None
        best_len = 0
        for name, ticker in self._name_to_ticker.items():
            if (key in name or name in key) and len(name) > best_len:
                best_match = ticker
                best_len = len(name)
        if best_match:
            return best_match

        logger.warning("Could not resolve company %r to a ticker", company)
        return None

