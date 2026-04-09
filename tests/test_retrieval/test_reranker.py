from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from research_assistant.retrieval.reranker import CrossEncoderReranker
from research_assistant.retrieval.vector_store import SearchResult

_PAYLOAD = {
    "chunk_id": "c1",
    "document_id": "d1",
    "section_name": "s1",
    "chunk_index": 0,
    "ticker": "AAPL",
    "company_name": "Apple",
    "period": "FY2025",
    "filing_type": "10-K",
    "source": "edgar",
}


def _make_result(id: str, text: str, score: float) -> SearchResult:
    return SearchResult(id=id, text=text, score=score, **_PAYLOAD)


@pytest.fixture
def mock_encoder() -> MagicMock:
    return MagicMock()


@pytest.fixture
def reranker(mock_encoder: MagicMock) -> CrossEncoderReranker:
    r = CrossEncoderReranker(model_name="test-model")
    r._encoder = mock_encoder
    return r


class TestCrossEncoderReranker:
    def test_rerank_sorts_by_cross_encoder_score(
        self, reranker: CrossEncoderReranker, mock_encoder: MagicMock
    ) -> None:
        results = [
            _make_result("a", "low relevance", score=0.9),
            _make_result("b", "high relevance", score=0.1),
            _make_result("c", "mid relevance", score=0.5),
        ]
        mock_encoder.rerank.return_value = [0.2, 0.9, 0.5]

        reranked = reranker.rerank("query", results, top_k=3)

        assert [r.id for r in reranked] == ["b", "c", "a"]
        assert reranked[0].score == pytest.approx(0.9)
        assert reranked[1].score == pytest.approx(0.5)
        assert reranked[2].score == pytest.approx(0.2)

    def test_rerank_truncates_to_top_k(
        self, reranker: CrossEncoderReranker, mock_encoder: MagicMock
    ) -> None:
        results = [_make_result(str(i), f"text {i}", score=0.5) for i in range(5)]
        mock_encoder.rerank.return_value = [0.1, 0.5, 0.3, 0.9, 0.7]

        reranked = reranker.rerank("query", results, top_k=2)

        assert len(reranked) == 2
        assert [r.id for r in reranked] == ["3", "4"]

    def test_rerank_fewer_results_than_top_k(
        self, reranker: CrossEncoderReranker, mock_encoder: MagicMock
    ) -> None:
        results = [_make_result("a", "text", score=0.5)]
        mock_encoder.rerank.return_value = [0.8]

        reranked = reranker.rerank("query", results, top_k=5)

        assert len(reranked) == 1
        assert reranked[0].score == pytest.approx(0.8)

    def test_rerank_empty_results(self, reranker: CrossEncoderReranker) -> None:
        reranked = reranker.rerank("query", [], top_k=5)

        assert reranked == []

    def test_rerank_replaces_original_score(
        self, reranker: CrossEncoderReranker, mock_encoder: MagicMock
    ) -> None:
        results = [_make_result("a", "text", score=0.99)]
        mock_encoder.rerank.return_value = [0.01]

        reranked = reranker.rerank("query", results, top_k=1)

        assert reranked[0].score == pytest.approx(0.01)

    def test_rerank_passes_texts_to_encoder(
        self, reranker: CrossEncoderReranker, mock_encoder: MagicMock
    ) -> None:
        results = [
            _make_result("a", "first doc", score=0.5),
            _make_result("b", "second doc", score=0.5),
        ]
        mock_encoder.rerank.return_value = [0.5, 0.5]

        reranker.rerank("my query", results, top_k=2)

        mock_encoder.rerank.assert_called_once_with("my query", ["first doc", "second doc"])

    def test_lazy_init_loads_model_on_first_call(self) -> None:
        reranker = CrossEncoderReranker(model_name="test-model")
        assert reranker._encoder is None

        with patch(
            "fastembed.rerank.cross_encoder.TextCrossEncoder"
        ) as mock_cls:
            mock_cls.return_value.rerank.return_value = [0.5]
            results = [_make_result("a", "text", score=0.5)]

            reranker.rerank("query", results, top_k=1)

            mock_cls.assert_called_once_with(model_name="test-model")

    def test_lazy_init_does_not_reload(
        self, reranker: CrossEncoderReranker, mock_encoder: MagicMock
    ) -> None:
        mock_encoder.rerank.return_value = [0.5]
        results = [_make_result("a", "text", score=0.5)]

        reranker.rerank("query", results, top_k=1)
        reranker.rerank("query", results, top_k=1)

        assert reranker._encoder is mock_encoder
