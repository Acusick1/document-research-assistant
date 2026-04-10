from __future__ import annotations

import logging

from research_assistant.corpus.models import Chunk
from research_assistant.retrieval.embeddings import Embedder, SparseEmbedder
from research_assistant.retrieval.vector_store import QdrantStore

logger = logging.getLogger(__name__)


def ingest_chunks(
    chunks: list[Chunk],
    embedder: Embedder,
    store: QdrantStore,
    sparse_embedder: SparseEmbedder | None = None,
) -> int:
    if not chunks:
        return 0
    texts = [c.text for c in chunks]
    vectors = embedder.embed(texts)
    sparse_vectors = sparse_embedder.embed(texts) if sparse_embedder else None
    store.upsert(chunks, vectors.tolist(), sparse_vectors=sparse_vectors)
    return len(chunks)
