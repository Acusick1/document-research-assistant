from __future__ import annotations

from dataclasses import dataclass

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from research_assistant.eval.models import EvalInput, EvalMetadata, EvalOutput


@dataclass
class AnswerContains(Evaluator[EvalInput, EvalOutput, EvalMetadata]):
    case_sensitive: bool = False

    def evaluate(
        self, ctx: EvaluatorContext[EvalInput, EvalOutput, EvalMetadata]
    ) -> EvaluationReason:
        expected = ctx.expected_output
        actual = ctx.output

        if expected is None or not expected.answer:
            return EvaluationReason(value=False, reason="No expected answer")

        if not actual.answer:
            return EvaluationReason(value=False, reason="No answer produced")

        expected_str = str(expected.answer)
        actual_str = str(actual.answer)

        if not self.case_sensitive:
            expected_str = expected_str.lower()
            actual_str = actual_str.lower()

        match = expected_str in actual_str
        reason = f"expected '{expected.answer}' in answer: {'found' if match else 'not found'}"
        return EvaluationReason(value=match, reason=reason)
