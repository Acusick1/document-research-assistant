from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from qdrant_client import QdrantClient

from research_assistant.corpus.edgar.metadata import EdgarMetadata
from research_assistant.corpus.models import Chunk
from research_assistant.retrieval.ingest import ingest_chunks
from research_assistant.retrieval.vector_store import QdrantStore


class FakeEmbedder:
    @property
    def dim(self) -> int:
        return 4

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.random.default_rng(42).random((len(texts), 4)).astype(np.float32)


@pytest.fixture
def store() -> QdrantStore:
    client = QdrantClient(":memory:")
    s = QdrantStore(client, "test", vector_dim=4)
    s.ensure_collection()
    return s


@pytest.fixture
def chunks() -> list[Chunk]:
    meta = EdgarMetadata(
        source="edgar",
        ticker="AAPL",
        filing_type="10-K",
        period="FY2024",
        section_name="Item 1",
        filing_date=date(2024, 11, 1),
    )
    return [
        Chunk(
            id=f"chunk_{i}",
            document_id="doc1",
            text=f"Text {i}",
            section_name="Item 1",
            metadata=meta,
            chunk_index=i,
        )
        for i in range(3)
    ]


class TestIngestChunks:
    def test_ingests_and_returns_count(self, chunks: list[Chunk], store: QdrantStore) -> None:
        result = ingest_chunks(chunks, FakeEmbedder(), store)
        assert result == 3
        assert store.count() == 3

    def test_empty_chunks(self, store: QdrantStore) -> None:
        result = ingest_chunks([], FakeEmbedder(), store)
        assert result == 0
        assert store.count() == 0
