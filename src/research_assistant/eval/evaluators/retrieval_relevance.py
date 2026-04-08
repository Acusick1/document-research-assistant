from __future__ import annotations

from dataclasses import dataclass

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext
from pydantic_evals.evaluators.llm_as_a_judge import judge_output

from research_assistant.eval.models import EvalInput, EvalMetadata, EvalOutput


@dataclass
class RetrievalRelevance(Evaluator[EvalInput, EvalOutput, EvalMetadata]):
    model: str | None = None

    async def evaluate(
        self, ctx: EvaluatorContext[EvalInput, EvalOutput, EvalMetadata]
    ) -> EvaluationReason:
        sources = ctx.output.sources
        if not sources:
            return EvaluationReason(value=0.0, reason="No sources retrieved")

        sources_text = "\n---\n".join(sources)
        output_for_judge = (
            f"<Question>{ctx.inputs.query}</Question>\n"
            f"<RetrievedSources>\n{sources_text}\n</RetrievedSources>"
        )
        rubric = (
            "Score how well the retrieved sources cover the information needed to "
            "answer the question. 1.0 = the answer is clearly and directly present "
            "in the sources. 0.5 = partial information that would require inference "
            "or is from a related but not exact context. 0.0 = the sources are "
            "irrelevant, from the wrong time period, or do not contain the requested "
            "data at all."
        )

        result = await judge_output(
            output=output_for_judge,
            rubric=rubric,
            model=self.model,
        )

        return EvaluationReason(value=result.score, reason=result.reason)
