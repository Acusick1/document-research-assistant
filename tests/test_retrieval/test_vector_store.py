from __future__ import annotations

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from research_assistant.corpus.models import Chunk
from research_assistant.retrieval.embeddings import SparseVector
from research_assistant.retrieval.vector_store import QdrantStore


@pytest.fixture
def store() -> QdrantStore:
    client = QdrantClient(":memory:")
    s = QdrantStore(client, "test_collection", vector_dim=4)
    s.ensure_collection()
    return s


@pytest.fixture
def sparse_store() -> QdrantStore:
    client = QdrantClient(":memory:")
    s = QdrantStore(client, "test_collection", vector_dim=4, enable_sparse=True)
    s.ensure_collection()
    return s


@pytest.fixture
def sample_chunks_with_vectors() -> tuple[list[Chunk], list[list[float]], list[SparseVector]]:
    from datetime import date

    from research_assistant.corpus.edgar.metadata import EdgarMetadata

    meta = EdgarMetadata(
        source="edgar",
        ticker="AAPL",
        company_name="Apple Inc.",
        filing_type="10-K",
        fiscal_year=2024,
        section_name="Item 1",
        filing_date=date(2024, 11, 1),
    )
    chunks = [
        Chunk(
            id="c1",
            document_id="doc1",
            text="Apple makes iPhones",
            section_name="Item 1",
            metadata=meta,
            chunk_index=0,
        ),
        Chunk(
            id="c2",
            document_id="doc1",
            text="Revenue was $391 billion",
            section_name="Item 7",
            metadata=meta.model_copy(update={"section_name": "Item 7", "fiscal_year": 2025}),
            chunk_index=1,
        ),
    ]
    vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    sparse_vectors = [
        SparseVector(indices=[0, 1], values=[0.5, 0.3]),
        SparseVector(indices=[1, 2], values=[0.7, 0.2]),
    ]
    return chunks, vectors, sparse_vectors


class TestQdrantStore:
    def test_ensure_collection_creates(self, store: QdrantStore) -> None:
        collections = [c.name for c in store.client.get_collections().collections]
        assert "test_collection" in collections

    def test_ensure_collection_idempotent(self, store: QdrantStore) -> None:
        store.ensure_collection()
        collections = [c.name for c in store.client.get_collections().collections]
        assert collections.count("test_collection") == 1

    def test_upsert_and_count(
        self,
        store: QdrantStore,
        sample_chunks_with_vectors: tuple[list[Chunk], list[list[float]], list[SparseVector]],
    ) -> None:
        chunks, vectors, _ = sample_chunks_with_vectors
        store.upsert(chunks, vectors)
        assert store.count() == 2

    def test_search(
        self,
        store: QdrantStore,
        sample_chunks_with_vectors: tuple[list[Chunk], list[list[float]], list[SparseVector]],
    ) -> None:
        chunks, vectors, _ = sample_chunks_with_vectors
        store.upsert(chunks, vectors)

        results = store.search([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].text == "Apple makes iPhones"

    def test_search_with_filter(
        self,
        store: QdrantStore,
        sample_chunks_with_vectors: tuple[list[Chunk], list[list[float]], list[SparseVector]],
    ) -> None:
        chunks, vectors, _ = sample_chunks_with_vectors
        store.upsert(chunks, vectors)

        results = store.search(
            [0.5, 0.5, 0.0, 0.0],
            top_k=2,
            qdrant_filter=Filter(must=[
                FieldCondition(key="ticker", match=MatchValue(value="AAPL")),
                FieldCondition(key="fiscal_year", match=MatchValue(value=2025)),
            ]),
        )
        assert len(results) == 1
        assert results[0].fiscal_year == 2025


class TestQdrantStoreHybrid:
    def test_upsert_with_sparse(
        self,
        sparse_store: QdrantStore,
        sample_chunks_with_vectors: tuple[list[Chunk], list[list[float]], list[SparseVector]],
    ) -> None:
        chunks, vectors, sparse_vectors = sample_chunks_with_vectors
        sparse_store.upsert(chunks, vectors, sparse_vectors=sparse_vectors)
        assert sparse_store.count() == 2

    def test_hybrid_search(
        self,
        sparse_store: QdrantStore,
        sample_chunks_with_vectors: tuple[list[Chunk], list[list[float]], list[SparseVector]],
    ) -> None:
        chunks, vectors, sparse_vectors = sample_chunks_with_vectors
        sparse_store.upsert(chunks, vectors, sparse_vectors=sparse_vectors)

        results = sparse_store.search(
            [1.0, 0.0, 0.0, 0.0],
            top_k=2,
            sparse_vector=SparseVector(indices=[0, 1], values=[0.5, 0.3]),
            prefetch_limit=10,
        )
        assert len(results) == 2

    def test_hybrid_search_with_filter(
        self,
        sparse_store: QdrantStore,
        sample_chunks_with_vectors: tuple[list[Chunk], list[list[float]], list[SparseVector]],
    ) -> None:
        chunks, vectors, sparse_vectors = sample_chunks_with_vectors
        sparse_store.upsert(chunks, vectors, sparse_vectors=sparse_vectors)

        results = sparse_store.search(
            [0.5, 0.5, 0.0, 0.0],
            top_k=2,
            qdrant_filter=Filter(must=[
                FieldCondition(key="fiscal_year", match=MatchValue(value=2025)),
            ]),
            sparse_vector=SparseVector(indices=[1, 2], values=[0.7, 0.2]),
            prefetch_limit=10,
        )
        assert len(results) == 1
        assert results[0].fiscal_year == 2025

    def test_dense_only_fallback_on_sparse_store(
        self,
        sparse_store: QdrantStore,
        sample_chunks_with_vectors: tuple[list[Chunk], list[list[float]], list[SparseVector]],
    ) -> None:
        chunks, vectors, sparse_vectors = sample_chunks_with_vectors
        sparse_store.upsert(chunks, vectors, sparse_vectors=sparse_vectors)

        results = sparse_store.search([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].text == "Apple makes iPhones"
