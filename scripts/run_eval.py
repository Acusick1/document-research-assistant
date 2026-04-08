from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from pydantic_evals.reporting import EvaluationReport, EvaluationReportAdapter

from research_assistant.config import configure_logfire, get_settings
from research_assistant.eval.models import EvalInput, EvalMetadata, EvalOutput
from research_assistant.eval.runner import run_all_evals
from research_assistant.pipeline import RagPipeline

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run eval suite")
    parser.add_argument(
        "--dataset",
        help="Run only the named dataset (e.g. 'factual', 'comparison')",
    )
    parser.add_argument(
        "--max-cases", type=int, help="Limit the number of cases per dataset",
    )
    parser.add_argument(
        "--name", help="Experiment name for Logfire tracking",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Max concurrent eval cases (default: 1)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print expected vs actual for failing cases",
    )
    return parser.parse_args()


type ReportDict = dict[str, EvaluationReport[EvalInput, EvalOutput, EvalMetadata]]


def _write_results(
    results: ReportDict,
    results_dir: Path,
    experiment_name: str | None,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    name = experiment_name or "run"
    for dataset_name, report in results.items():
        filename = f"{name}_{dataset_name}_{timestamp}.json"
        path = results_dir / filename
        data = json.loads(EvaluationReportAdapter.dump_json(report))
        path.write_text(json.dumps(data, indent=2))
        print(f"\n  Results written to {path}")


async def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logfire(settings)
    logging.basicConfig(level=settings.log_level)

    pipeline = RagPipeline(settings)

    logger.info("Running eval suite (concurrency=%d)", args.concurrency)

    results = await run_all_evals(
        task=pipeline.__call__, dataset_name=args.dataset, max_cases=args.max_cases,
        max_concurrency=args.concurrency, experiment_name=args.name,
    )

    for name, report in results.items():
        print(f"\n{'=' * 60}")
        print(f"Dataset: {name} ({len(report.cases)} cases)")
        print("=" * 60)

        report.print()

        if args.verbose:
            failing = [c for c in report.cases if any(not a.value for a in c.assertions.values())]
            if failing:
                print(f"\n  Failing cases ({len(failing)}):")
                for case in failing:
                    print(f"\n    {case.name}:")
                    if case.expected_output is not None:
                        print(f"      expected: {case.expected_output}")
                    print(f"      actual:   {case.output}")
                    for aname, assertion in case.assertions.items():
                        if not assertion.value and assertion.reason:
                            print(f"      {aname}: {assertion.reason}")

        if report.failures:
            print(f"\n  Task failures: {len(report.failures)}")

        by_category: dict[str, list[float | None]] = defaultdict(list)
        for case in report.cases:
            category = "unknown"
            if isinstance(case.metadata, EvalMetadata):
                category = case.metadata.category
            assertion_pass = (
                sum(1 for a in case.assertions.values() if a.value) / len(case.assertions)
                if case.assertions
                else None
            )
            by_category[category].append(assertion_pass)

        if by_category:
            print("\n  Per-category pass rates:")
            for category, rates in sorted(by_category.items()):
                valid = [r for r in rates if r is not None]
                if valid:
                    avg = sum(valid) / len(valid)
                    print(f"    {category}: {avg:.0%} ({len(valid)}/{len(rates)} evaluated)")
                else:
                    print(f"    {category}: no evaluators ({len(rates)} cases)")

    if results:
        _write_results(results, settings.eval_results_dir, args.name)

    if not results:
        print("No datasets found.")


if __name__ == "__main__":
    asyncio.run(main())
