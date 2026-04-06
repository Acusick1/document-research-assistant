from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from pydantic_evals import Dataset
from pydantic_evals.evaluators.llm_as_a_judge import set_default_judge_model
from pydantic_evals.reporting import EvaluationReport

from research_assistant.config import get_settings
from research_assistant.eval.evaluators.context_precision import ContextPrecision
from research_assistant.eval.evaluators.faithfulness import Faithfulness
from research_assistant.eval.evaluators.numeric_match import NumericMatch
from research_assistant.eval.models import EvalInput, EvalMetadata, EvalOutput

type TaskFn = Callable[[EvalInput], Awaitable[EvalOutput]]

CUSTOM_EVALUATORS = (NumericMatch, ContextPrecision, Faithfulness)


def _configure_judge_model() -> None:
    settings = get_settings()
    set_default_judge_model(settings.eval_judge_model)


async def stub_task(_inputs: EvalInput) -> EvalOutput:
    return EvalOutput()


async def run_eval(
    dataset_path: Path,
    task: TaskFn | None = None,
) -> EvaluationReport[EvalInput, EvalOutput, EvalMetadata]:
    _configure_judge_model()
    dataset = Dataset[EvalInput, EvalOutput, EvalMetadata].from_file(
        dataset_path,
        custom_evaluator_types=CUSTOM_EVALUATORS,
    )
    task_fn = task or stub_task
    return await dataset.evaluate(task_fn)


async def run_all_evals(
    task: TaskFn | None = None,
    datasets_dir: Path | None = None,
) -> dict[str, EvaluationReport[EvalInput, EvalOutput, EvalMetadata]]:
    datasets_dir = datasets_dir or get_settings().datasets_dir
    results: dict[str, EvaluationReport[EvalInput, EvalOutput, EvalMetadata]] = {}
    if not datasets_dir.exists():
        return results

    for yaml_file in sorted(datasets_dir.glob("*.yaml")):
        result = await run_eval(yaml_file, task=task)
        results[yaml_file.stem] = result

    return results
