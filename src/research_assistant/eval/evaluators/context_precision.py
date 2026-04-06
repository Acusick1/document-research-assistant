from __future__ import annotations

from dataclasses import dataclass

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from research_assistant.eval.models import EvalInput, EvalMetadata, EvalOutput


@dataclass
class ContextPrecision(Evaluator[EvalInput, EvalOutput, EvalMetadata]):
    def evaluate(
        self, ctx: EvaluatorContext[EvalInput, EvalOutput, EvalMetadata]
    ) -> EvaluationReason:
        sources = ctx.output.sources
        if not sources:
            return EvaluationReason(value=0.0, reason="No sources retrieved")

        metadata = ctx.metadata
        if metadata is None:
            return EvaluationReason(value=0.0, reason="No metadata to check relevance")

        target_company = metadata.company
        if not target_company:
            return EvaluationReason(value=0.0, reason="No target company in metadata")

        relevant = sum(1 for s in sources if target_company.upper() in s.upper())
        precision = relevant / len(sources)

        return EvaluationReason(
            value=precision,
            reason=f"{relevant}/{len(sources)} sources relevant to {target_company}",
        )
