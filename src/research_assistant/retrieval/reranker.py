from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from research_assistant.retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from fastembed.rerank.cross_encoder import TextCrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._encoder: TextCrossEncoder | None = None

    def _load(self) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        logger.info("Loading cross-encoder model: %s", self._model_name)
        self._encoder = TextCrossEncoder(model_name=self._model_name)

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        if not results:
            return []
        if self._encoder is None:
            self._load()

        scores = list(self._encoder.rerank(query, [r.text for r in results]))  # type: ignore[union-attr]
        scored = sorted(zip(scores, results, strict=True), key=lambda x: x[0], reverse=True)
        reranked = [r.model_copy(update={"score": s}) for s, r in scored[:top_k]]
        logger.info(
            "Reranked %d -> %d results, top score: %.3f",
            len(results), len(reranked), reranked[0].score if reranked else 0.0,
        )
        return reranked
