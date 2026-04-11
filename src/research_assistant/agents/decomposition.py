from __future__ import annotations

import logging

from pydantic import BaseModel, Field
from pydantic_ai import Agent

logger = logging.getLogger(__name__)

DECOMPOSITION_PROMPT_TEMPLATE = """\
You are a query decomposition preprocessor for a financial document search system.

Your job is to decide whether a user's query needs to be broken into multiple independent \
sub-queries for retrieval, and if so, produce those sub-queries.

**When to decompose:**
- The query asks about multiple companies (e.g. "Compare Apple and Microsoft's revenue")
- The query asks about multiple time periods (e.g. "How did revenue trend from 2022 to 2024")
- The query requires multiple distinct pieces of information (e.g. "What was the revenue growth \
and what risk factors were cited")

**When NOT to decompose:**
- Simple single-company, single-metric questions (e.g. "What was Apple's revenue in FY2024")
- Questions that are already focused on one retrieval target

**Rules:**
- Each sub-query must be a self-contained question that can be answered independently.
- For each sub-query, list the company names or tickers it targets.
- Never produce more than {max_sub_queries} sub-queries.
- If decomposition is not needed, set sub_queries to null.
- Do not rephrase the query if it doesn't need decomposition — just return null.\
"""


class SubQuery(BaseModel):
    query: str = Field(description="Self-contained sub-query for independent retrieval")
    companies: list[str] = Field(
        default_factory=list,
        description="Company names or tickers this sub-query targets",
    )


class DecompositionResult(BaseModel):
    sub_queries: list[SubQuery] | None = Field(
        default=None,
        description=(
            "List of sub-queries for independent retrieval, "
            "or null if the original query should be used as-is"
        ),
    )


class QueryDecomposer:
    def __init__(self, model: str, max_sub_queries: int) -> None:
        prompt = DECOMPOSITION_PROMPT_TEMPLATE.format(max_sub_queries=max_sub_queries)
        self._agent: Agent[None, DecompositionResult] = Agent(
            model,
            system_prompt=prompt,
            output_type=DecompositionResult,
        )
        self._max_sub_queries = max_sub_queries

    async def decompose(self, query: str) -> DecompositionResult:
        try:
            result = await self._agent.run(query)
        except Exception:
            logger.exception("Query decomposition failed, falling back to original query: %r", query)
            return DecompositionResult()

        output = result.output
        if output.sub_queries is not None:
            output.sub_queries = output.sub_queries[: self._max_sub_queries]
        logger.info(
            "Decomposition: %r -> %s",
            query,
            output.sub_queries if output.sub_queries else "pass-through",
        )
        return output
