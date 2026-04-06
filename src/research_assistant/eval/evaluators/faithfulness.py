from __future__ import annotations

from dataclasses import dataclass

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext
from pydantic_evals.evaluators.llm_as_a_judge import judge_output

from research_assistant.eval.models import EvalInput, EvalMetadata, EvalOutput


@dataclass
class Faithfulness(Evaluator[EvalInput, EvalOutput, EvalMetadata]):
    model: str | None = None

    async def evaluate(
        self, ctx: EvaluatorContext[EvalInput, EvalOutput, EvalMetadata]
    ) -> EvaluationReason:
        answer = ctx.output.answer
        sources = ctx.output.sources

        if not answer:
            return EvaluationReason(value=0.0, reason="No answer produced")

        if not sources:
            return EvaluationReason(value=0.0, reason="No sources retrieved")

        sources_text = "\n---\n".join(sources)
        output_for_judge = (
            f"<Answer>{answer}</Answer>\n"
            f"<Sources>\n{sources_text}\n</Sources>"
        )
        rubric = (
            "Evaluate faithfulness: what fraction of factual claims in the Answer "
            "are directly supported by the Sources? Ignore stylistic differences. "
            "Score 1.0 if all claims are supported, 0.0 if none are, and proportionally "
            "for partial support. A claim counts as supported if the Sources contain "
            "information that logically entails it."
        )

        result = await judge_output(
            output=output_for_judge,
            rubric=rubric,
            model=self.model,
        )

        return EvaluationReason(value=result.score, reason=result.reason)
