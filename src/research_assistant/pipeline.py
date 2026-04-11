from __future__ import annotations

import logging

from pydantic_ai.settings import ModelSettings
from qdrant_client.models import Filter

from research_assistant.agents.decomposition import QueryDecomposer, SubQuery
from research_assistant.agents.simple import create_agent
from research_assistant.config import Settings, get_settings
from research_assistant.eval.models import EvalInput, EvalOutput
from research_assistant.retrieval.embeddings import FastEmbedEmbedder, FastEmbedSparseEmbedder
from research_assistant.retrieval.query_filter import QueryFilterExtractor, QueryFilters
from research_assistant.retrieval.reranker import CrossEncoderReranker
from research_assistant.retrieval.vector_store import (
    QdrantStore,
    SearchResult,
    create_qdrant_client,
)

logger = logging.getLogger(__name__)


def _round_robin_merge(
    result_groups: list[list[SearchResult]], top_k: int,
) -> list[SearchResult]:
    from itertools import zip_longest

    merged: list[SearchResult] = []
    seen_ids: set[str] = set()
    for row in zip_longest(*result_groups):
        for result in row:
            if result is not None and result.id not in seen_ids:
                merged.append(result)
                seen_ids.add(result.id)
                if len(merged) >= top_k:
                    return merged
    return merged


def _format_context(results: list[SearchResult]) -> str:
    if not results:
        return "(No relevant documents found)"
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        header = f"[{i}] {r.ticker} FY{r.fiscal_year} — {r.section_name} (score: {r.score:.3f})"
        parts.append(f"{header}\n{r.text}")
    return "\n\n---\n\n".join(parts)


def _sources_from_results(results: list[SearchResult]) -> list[str]:
    return [f"[{r.ticker} FY{r.fiscal_year} {r.section_name}] {r.text[:200]}" for r in results]


class RagPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self._embedder = FastEmbedEmbedder(model_name=settings.embedding_model)
        client = create_qdrant_client(settings)
        self._store = QdrantStore(
            client,
            settings.collection_name,
            self._embedder.dim,
            enable_sparse=bool(settings.sparse_model),
        )
        self._agent = create_agent(model=settings.llm_model)
        self._filter_extractor = QueryFilterExtractor(
            store=self._store, model=settings.filter_model,
        )
        self._decomposer = QueryDecomposer(
            model=settings.filter_model, max_sub_queries=settings.top_k,
        )
        self._sparse_embedder: FastEmbedSparseEmbedder | None = None
        if settings.sparse_model:
            self._sparse_embedder = FastEmbedSparseEmbedder(model_name=settings.sparse_model)
        self._reranker: CrossEncoderReranker | None = None
        if settings.rerank_model:
            self._reranker = CrossEncoderReranker(model_name=settings.rerank_model)
        self._top_k = settings.top_k
        self._rerank_top_k = settings.rerank_top_k
        self._prefetch_limit = settings.prefetch_limit
        self._max_tokens = settings.max_tokens

    def _build_sub_query_filter(self, sub_query: SubQuery) -> Filter | None:
        tickers: list[str] = []
        for company in sub_query.companies:
            ticker = self._filter_extractor.match_ticker(company)
            if ticker and ticker not in tickers:
                tickers.append(ticker)
        if not tickers:
            return None
        return QueryFilters(tickers=tickers).to_qdrant_filter()

    def _retrieve(
        self, query: str, qdrant_filter: object | None,
    ) -> list[SearchResult]:
        query_vector = self._embedder.embed([query])[0].tolist()

        sparse_query = None
        if self._sparse_embedder:
            sparse_query = self._sparse_embedder.embed([query])[0]

        search_top_k = self._rerank_top_k if self._reranker else self._top_k
        results = self._store.search(
            query_vector,
            top_k=search_top_k,
            qdrant_filter=qdrant_filter,
            sparse_vector=sparse_query,
            prefetch_limit=self._prefetch_limit,
        )
        if self._reranker:
            results = self._reranker.rerank(query, results, top_k=self._top_k)
        return results

    async def __call__(self, eval_input: EvalInput) -> EvalOutput:
        filter_result = await self._filter_extractor.extract(eval_input.query)
        if filter_result.reject_reason:
            return EvalOutput(answer=filter_result.reject_reason, sources=[])

        decomposition = await self._decomposer.decompose(eval_input.query)

        if decomposition.sub_queries:
            result_groups = [
                self._retrieve(sq.query, self._build_sub_query_filter(sq))
                for sq in decomposition.sub_queries
            ]
            results = _round_robin_merge(result_groups, self._top_k)
        else:
            results = self._retrieve(eval_input.query, filter_result.qdrant_filter)

        context = _format_context(results)
        sources = _sources_from_results(results)

        prompt = f"Question: {eval_input.query}\n\nRetrieved Context:\n{context}"
        logger.debug("RAG prompt (%d retrieved chunks):\n%s", len(results), prompt[:500])

        response = await self._agent.run(
            prompt, model_settings=ModelSettings(max_tokens=self._max_tokens),
        )
        output = response.output

        return EvalOutput(
            answer=output.answer,
            numeric_answer=output.numeric_answer,
            sources=sources,
        )
