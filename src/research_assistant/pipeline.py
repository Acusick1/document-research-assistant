from __future__ import annotations

import logging

from pydantic_ai.settings import ModelSettings

from research_assistant.agents.simple import create_agent
from research_assistant.config import Settings, get_settings
from research_assistant.eval.models import EvalInput, EvalOutput
from research_assistant.retrieval.embeddings import FastEmbedEmbedder
from research_assistant.retrieval.query_filter import QueryFilterExtractor
from research_assistant.retrieval.reranker import CrossEncoderReranker
from research_assistant.retrieval.vector_store import (
    QdrantStore,
    SearchResult,
    create_qdrant_client,
)

logger = logging.getLogger(__name__)


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
        self._store = QdrantStore(client, settings.collection_name, self._embedder.dim)
        self._agent = create_agent(model=settings.llm_model)
        self._filter_extractor = QueryFilterExtractor(
            store=self._store, model=settings.filter_model,
        )
        self._reranker: CrossEncoderReranker | None = None
        if settings.rerank_model:
            self._reranker = CrossEncoderReranker(model_name=settings.rerank_model)
        self._top_k = settings.top_k
        self._rerank_top_k = settings.rerank_top_k
        self._max_tokens = settings.max_tokens

    async def __call__(self, eval_input: EvalInput) -> EvalOutput:
        filters = await self._filter_extractor.extract(eval_input.query)
        query_vector = self._embedder.embed([eval_input.query])[0].tolist()
        search_top_k = self._rerank_top_k if self._reranker else self._top_k
        results = self._store.search(query_vector, top_k=search_top_k, filters=filters)
        if self._reranker:
            results = self._reranker.rerank(eval_input.query, results, top_k=self._top_k)

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
