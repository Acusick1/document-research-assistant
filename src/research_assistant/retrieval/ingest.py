from __future__ import annotations

import logging

from research_assistant.corpus.models import Chunk
from research_assistant.retrieval.embeddings import Embedder
from research_assistant.retrieval.vector_store import QdrantStore

logger = logging.getLogger(__name__)


def ingest_chunks(
    chunks: list[Chunk],
    embedder: Embedder,
    store: QdrantStore,
) -> int:
    if not chunks:
        return 0
    texts = [c.text for c in chunks]
    vectors = embedder.embed(texts)
    store.upsert(chunks, vectors.tolist())
    return len(chunks)
