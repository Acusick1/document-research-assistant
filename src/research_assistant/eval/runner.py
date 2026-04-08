from __future__ import annotations

import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic_evals import Dataset
from pydantic_evals.evaluators.llm_as_a_judge import set_default_judge_model
from pydantic_evals.reporting import EvaluationReport

from research_assistant.config import Settings, get_settings
from research_assistant.eval.evaluators.answer_contains import AnswerContains
from research_assistant.eval.evaluators.context_precision import ContextPrecision
from research_assistant.eval.evaluators.faithfulness import Faithfulness
from research_assistant.eval.evaluators.numeric_match import NumericMatch
from research_assistant.eval.models import EvalInput, EvalMetadata, EvalOutput

type TaskFn = Callable[[EvalInput], Awaitable[EvalOutput]]

CUSTOM_EVALUATORS = (AnswerContains, NumericMatch, ContextPrecision, Faithfulness)


def _configure_judge_model() -> None:
    settings = get_settings()
    set_default_judge_model(settings.eval_judge_model)


def _build_metadata(settings: Settings) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
        ).strip()
    except subprocess.SubprocessError:
        commit = "unknown"
    return {
        "commit": commit,
        "model": settings.llm_model,
        "top_k": settings.top_k,
        "max_tokens": settings.max_tokens,
    }


async def run_eval(
    dataset_path: Path,
    task: TaskFn,
    task_name: str = "RagPipeline",
    max_concurrency: int = 1,
    max_cases: int | None = None,
    experiment_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvaluationReport[EvalInput, EvalOutput, EvalMetadata]:
    _configure_judge_model()
    dataset = Dataset[EvalInput, EvalOutput, EvalMetadata].from_file(
        dataset_path,
        custom_evaluator_types=CUSTOM_EVALUATORS,
    )
    if max_cases is not None:
        dataset = Dataset(name=dataset.name, cases=dataset.cases[:max_cases])
    return await dataset.evaluate(
        task,
        name=experiment_name,
        task_name=task_name,
        max_concurrency=max_concurrency,
        metadata=metadata,
    )


async def run_all_evals(
    task: TaskFn,
    datasets_dir: Path | None = None,
    dataset_name: str | None = None,
    max_cases: int | None = None,
    max_concurrency: int = 1,
    experiment_name: str | None = None,
) -> dict[str, EvaluationReport[EvalInput, EvalOutput, EvalMetadata]]:
    settings = get_settings()
    datasets_dir = datasets_dir or settings.datasets_dir
    metadata = _build_metadata(settings)
    results: dict[str, EvaluationReport[EvalInput, EvalOutput, EvalMetadata]] = {}
    if not datasets_dir.exists():
        return results

    for yaml_file in sorted(datasets_dir.glob("*.yaml")):
        if dataset_name is not None and yaml_file.stem != dataset_name:
            continue
        result = await run_eval(
            yaml_file, task=task, max_cases=max_cases,
            max_concurrency=max_concurrency,
            experiment_name=experiment_name, metadata=metadata,
        )
        results[yaml_file.stem] = result

    return results
