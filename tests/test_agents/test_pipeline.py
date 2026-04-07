from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from qdrant_client import QdrantClient

from research_assistant.agents.simple import AgentResponse
from research_assistant.config import Settings
from research_assistant.corpus.edgar.metadata import EdgarMetadata
from research_assistant.corpus.models import Chunk
from research_assistant.eval.models import EvalInput
from research_assistant.pipeline import RagPipeline, _format_context, _sources_from_results
from research_assistant.retrieval.ingest import ingest_chunks
from research_assistant.retrieval.vector_store import QdrantStore, SearchResult


def _result(**overrides: object) -> SearchResult:
    defaults = {
        "id": "abc",
        "score": 0.9,
        "text": "Some text",
        "chunk_id": "c1",
        "document_id": "doc1",
        "section_name": "Item 7",
        "chunk_index": 0,
        "ticker": "AAPL",
        "period": "FY2024",
        "filing_type": "10-K",
        "source": "edgar",
    }
    return SearchResult(**(defaults | overrides))  # type: ignore[arg-type]


class TestFormatContext:
    def test_empty_results(self) -> None:
        assert _format_context([]) == "(No relevant documents found)"

    def test_formats_results(self) -> None:
        r = _result(score=0.912, text="Revenue increased year over year.")
        formatted = _format_context([r])
        assert "AAPL" in formatted
        assert "FY2024" in formatted
        assert "Item 7" in formatted
        assert "0.912" in formatted
        assert "Revenue increased" in formatted

    def test_multiple_results_separated(self) -> None:
        r1 = _result(ticker="AAPL", score=0.9, text="A")
        r2 = _result(ticker="MSFT", section_name="Item 1", score=0.8, text="B")
        formatted = _format_context([r1, r2])
        assert "---" in formatted
        assert "[1]" in formatted
        assert "[2]" in formatted


class TestSourcesFromResults:
    def test_includes_ticker_and_text(self) -> None:
        r = _result(text="Revenue was $391B.")
        sources = _sources_from_results([r])
        assert len(sources) == 1
        assert "AAPL" in sources[0]
        assert "Revenue" in sources[0]


class FakeEmbedder:
    @property
    def dim(self) -> int:
        return 4

    def embed(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(hash(texts[0]) % 2**31)
        return rng.random((len(texts), 4)).astype(np.float32)


class TestRagPipeline:
    @pytest.fixture
    def settings(self) -> Settings:
        return Settings(qdrant_mode="memory", top_k=2)

    @pytest.fixture
    def populated_store(self) -> QdrantStore:
        client = QdrantClient(":memory:")
        store = QdrantStore(client, "documents", vector_dim=4)
        store.ensure_collection()
        meta = EdgarMetadata(
            source="edgar",
            ticker="AAPL",
            filing_type="10-K",
            period="FY2024",
            section_name="Item 7",
            filing_date=date(2024, 11, 1),
        )
        chunks = [
            Chunk(
                id=f"chunk_{i}",
                document_id="doc1",
                text=f"Apple revenue data chunk {i}",
                section_name="Item 7",
                metadata=meta,
                chunk_index=i,
            )
            for i in range(3)
        ]
        ingest_chunks(chunks, FakeEmbedder(), store)
        return store

    @pytest.mark.anyio
    async def test_wiring(self, settings: Settings, populated_store: QdrantStore) -> None:
        mock_response = AgentResponse(
            answer="Apple's revenue was $391B",
            numeric_answer=391_035_000_000.0,
            reasoning="Found in Item 7",
            cited_sections=["AAPL 10-K FY2024 Item 7"],
            confidence=0.9,
        )
        mock_result = MagicMock()
        mock_result.output = mock_response

        with (
            patch(
                "research_assistant.pipeline.FastEmbedEmbedder",
                return_value=FakeEmbedder(),
            ),
            patch(
                "research_assistant.pipeline.create_qdrant_client",
                return_value=populated_store.client,
            ),
            patch("research_assistant.pipeline.create_agent") as mock_create_agent,
        ):
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_result
            mock_create_agent.return_value = mock_agent

            pipeline = RagPipeline(settings)
            result = await pipeline(EvalInput(query="What was Apple's revenue in FY2024?"))

        assert result.answer == "Apple's revenue was $391B"
        assert result.numeric_answer == 391_035_000_000.0
        assert len(result.sources) > 0
        assert any("AAPL" in s for s in result.sources)
        mock_agent.run.assert_called_once()
        prompt_arg = mock_agent.run.call_args[0][0]
        assert "Apple" in prompt_arg
