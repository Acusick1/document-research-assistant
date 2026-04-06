from __future__ import annotations

from dataclasses import dataclass

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from research_assistant.eval.models import EvalInput, EvalMetadata, EvalOutput


@dataclass
class NumericMatch(Evaluator[EvalInput, EvalOutput, EvalMetadata]):
    tolerance_pct: float = 0.01

    def evaluate(
        self, ctx: EvaluatorContext[EvalInput, EvalOutput, EvalMetadata]
    ) -> EvaluationReason:
        expected = ctx.expected_output
        actual = ctx.output

        if expected is None or expected.numeric_answer is None:
            return EvaluationReason(value=False, reason="No expected numeric answer")

        if actual.numeric_answer is None:
            return EvaluationReason(value=False, reason="No numeric answer produced")

        expected_val = expected.numeric_answer
        actual_val = actual.numeric_answer

        if expected_val == 0:
            match = actual_val == 0
        else:
            pct_diff = abs(actual_val - expected_val) / abs(expected_val)
            match = pct_diff <= self.tolerance_pct

        reason = f"expected={expected_val}, actual={actual_val}"
        return EvaluationReason(value=match, reason=reason)
