from __future__ import annotations

import pytest

from research_assistant.agents.decomposition import DecompositionResult, QueryDecomposer, SubQuery
from research_assistant.pipeline import _round_robin_merge
from research_assistant.retrieval.vector_store import SearchResult


def _result(id: str, score: float = 0.9) -> SearchResult:
    return SearchResult(
        id=id,
        score=score,
        text=f"text {id}",
        chunk_id=id,
        document_id="doc1",
        section_name="Item 7",
        chunk_index=0,
        ticker="AAPL",
        company_name="Apple Inc.",
        fiscal_year=2024,
        filing_type="10-K",
        source="edgar",
    )


class TestDecompositionResult:
    def test_pass_through(self) -> None:
        result = DecompositionResult()
        assert result.sub_queries is None

    def test_decomposed(self) -> None:
        sqs = [SubQuery(query="q1", companies=["AAPL"]), SubQuery(query="q2", companies=["MSFT"])]
        result = DecompositionResult(sub_queries=sqs)
        assert len(result.sub_queries) == 2
        assert result.sub_queries[0].query == "q1"
        assert result.sub_queries[0].companies == ["AAPL"]


class TestRoundRobinMerge:
    def test_single_group(self) -> None:
        results = [_result("a"), _result("b"), _result("c")]
        merged = _round_robin_merge([results], top_k=2)
        assert [r.id for r in merged] == ["a", "b"]

    def test_two_groups_interleaved(self) -> None:
        g1 = [_result("a1"), _result("a2")]
        g2 = [_result("b1"), _result("b2")]
        merged = _round_robin_merge([g1, g2], top_k=4)
        assert [r.id for r in merged] == ["a1", "b1", "a2", "b2"]

    def test_respects_top_k(self) -> None:
        g1 = [_result("a1"), _result("a2")]
        g2 = [_result("b1"), _result("b2")]
        merged = _round_robin_merge([g1, g2], top_k=3)
        assert len(merged) == 3
        assert [r.id for r in merged] == ["a1", "b1", "a2"]

    def test_deduplicates(self) -> None:
        shared = _result("shared")
        g1 = [shared, _result("a1")]
        g2 = [_result("shared", score=0.8), _result("b1")]
        merged = _round_robin_merge([g1, g2], top_k=3)
        ids = [r.id for r in merged]
        assert ids.count("shared") == 1
        assert len(merged) == 3

    def test_uneven_groups(self) -> None:
        g1 = [_result("a1")]
        g2 = [_result("b1"), _result("b2"), _result("b3")]
        merged = _round_robin_merge([g1, g2], top_k=3)
        assert [r.id for r in merged] == ["a1", "b1", "b2"]

    def test_empty_groups(self) -> None:
        merged = _round_robin_merge([], top_k=3)
        assert merged == []

    def test_three_groups_top_k_3(self) -> None:
        g1 = [_result("a1"), _result("a2")]
        g2 = [_result("b1"), _result("b2")]
        g3 = [_result("c1"), _result("c2")]
        merged = _round_robin_merge([g1, g2, g3], top_k=3)
        assert [r.id for r in merged] == ["a1", "b1", "c1"]


class TestQueryDecomposer:
    @pytest.fixture
    def decomposer(self) -> QueryDecomposer:
        return QueryDecomposer(model="test", max_sub_queries=3)

    def test_truncates_excess_sub_queries(self) -> None:
        sqs = [SubQuery(query=f"q{i}") for i in range(4)]
        result = DecompositionResult(sub_queries=sqs)
        result.sub_queries = result.sub_queries[:3]
        assert len(result.sub_queries) == 3
