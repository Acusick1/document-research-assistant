from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from pydantic_evals.evaluators import EvaluatorContext

from research_assistant.eval.evaluators.context_precision import ContextPrecision
from research_assistant.eval.evaluators.faithfulness import Faithfulness
from research_assistant.eval.evaluators.numeric_match import NumericMatch
from research_assistant.eval.models import EvalInput, EvalMetadata, EvalOutput


def _make_context(
    output: EvalOutput,
    expected: EvalOutput | None = None,
    metadata: EvalMetadata | None = None,
) -> EvaluatorContext[EvalInput, EvalOutput, EvalMetadata]:
    return EvaluatorContext(
        name="test",
        inputs=EvalInput(query="test query"),
        metadata=metadata,
        expected_output=expected,
        output=output,
        duration=0.1,
        _span_tree=None,  # type: ignore[arg-type]
        attributes={},
        metrics={},
    )


class TestNumericMatch:
    def test_exact_match(self) -> None:
        evaluator = NumericMatch()
        ctx = _make_context(
            output=EvalOutput(numeric_answer=100.0),
            expected=EvalOutput(numeric_answer=100.0),
        )
        result = evaluator.evaluate(ctx)
        assert result.value is True

    def test_within_tolerance(self) -> None:
        evaluator = NumericMatch(tolerance_pct=0.05)
        ctx = _make_context(
            output=EvalOutput(numeric_answer=103.0),
            expected=EvalOutput(numeric_answer=100.0),
        )
        result = evaluator.evaluate(ctx)
        assert result.value is True

    def test_outside_tolerance(self) -> None:
        evaluator = NumericMatch(tolerance_pct=0.01)
        ctx = _make_context(
            output=EvalOutput(numeric_answer=110.0),
            expected=EvalOutput(numeric_answer=100.0),
        )
        result = evaluator.evaluate(ctx)
        assert result.value is False

    def test_no_expected(self) -> None:
        evaluator = NumericMatch()
        ctx = _make_context(
            output=EvalOutput(numeric_answer=100.0),
            expected=None,
        )
        result = evaluator.evaluate(ctx)
        assert result.value is False

    def test_no_actual(self) -> None:
        evaluator = NumericMatch()
        ctx = _make_context(
            output=EvalOutput(),
            expected=EvalOutput(numeric_answer=100.0),
        )
        result = evaluator.evaluate(ctx)
        assert result.value is False


class TestContextPrecision:
    def test_all_relevant(self) -> None:
        evaluator = ContextPrecision()
        ctx = _make_context(
            output=EvalOutput(sources=["AAPL_10K_2024:Item1", "AAPL_10K_2023:Item7"]),
            metadata=EvalMetadata(category="factual", company="AAPL"),
        )
        result = evaluator.evaluate(ctx)
        assert result.value == 1.0

    def test_partial_relevant(self) -> None:
        evaluator = ContextPrecision()
        ctx = _make_context(
            output=EvalOutput(sources=["AAPL_chunk1", "MSFT_chunk2"]),
            metadata=EvalMetadata(category="factual", company="AAPL"),
        )
        result = evaluator.evaluate(ctx)
        assert result.value == 0.5

    def test_no_sources(self) -> None:
        evaluator = ContextPrecision()
        ctx = _make_context(
            output=EvalOutput(sources=[]),
            metadata=EvalMetadata(category="factual", company="AAPL"),
        )
        result = evaluator.evaluate(ctx)
        assert result.value == 0.0

    def test_multi_company_all_relevant(self) -> None:
        evaluator = ContextPrecision()
        ctx = _make_context(
            output=EvalOutput(sources=["AAPL_chunk1", "MSFT_chunk2"]),
            metadata=EvalMetadata(category="comparison", companies=["AAPL", "MSFT"]),
        )
        result = evaluator.evaluate(ctx)
        assert result.value == 1.0

    def test_multi_company_partial(self) -> None:
        evaluator = ContextPrecision()
        ctx = _make_context(
            output=EvalOutput(sources=["AAPL_chunk1", "NVDA_chunk2"]),
            metadata=EvalMetadata(category="comparison", companies=["AAPL", "MSFT"]),
        )
        result = evaluator.evaluate(ctx)
        assert result.value == 0.5


class TestFaithfulness:
    @pytest.mark.anyio
    async def test_no_answer(self) -> None:
        evaluator = Faithfulness()
        ctx = _make_context(
            output=EvalOutput(answer=None, sources=["some source text"]),
        )
        result = await evaluator.evaluate(ctx)
        assert result.value == 0.0
        assert "No answer" in (result.reason or "")

    @pytest.mark.anyio
    async def test_no_sources(self) -> None:
        evaluator = Faithfulness()
        ctx = _make_context(
            output=EvalOutput(answer="Revenue was $100B", sources=[]),
        )
        result = await evaluator.evaluate(ctx)
        assert result.value == 0.0
        assert "No sources" in (result.reason or "")

    @pytest.mark.anyio
    async def test_calls_judge(self) -> None:
        mock_grading = AsyncMock()
        mock_grading.return_value.score = 0.8
        mock_grading.return_value.reason = "Most claims supported"

        evaluator = Faithfulness()
        ctx = _make_context(
            output=EvalOutput(
                answer="Revenue was $100B",
                sources=["Annual revenue: $100 billion"],
            ),
        )

        with patch(
            "research_assistant.eval.evaluators.faithfulness.judge_output",
            mock_grading,
        ):
            result = await evaluator.evaluate(ctx)

        assert result.value == 0.8
        assert result.reason == "Most claims supported"
        mock_grading.assert_called_once()
        call_kwargs = mock_grading.call_args
        assert "faithfulness" in call_kwargs.kwargs["rubric"].lower()
