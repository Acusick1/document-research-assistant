from __future__ import annotations

import argparse
import asyncio
import logging
from collections import defaultdict

from research_assistant.config import configure_logfire, get_settings
from research_assistant.eval.models import EvalMetadata
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
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logfire(settings)
    logging.basicConfig(level=settings.log_level)

    pipeline = RagPipeline(settings)

    logger.info("Running eval suite (concurrency=%d)", args.concurrency)

    results = await run_all_evals(
        task=pipeline, dataset_name=args.dataset, max_cases=args.max_cases,
        max_concurrency=args.concurrency, experiment_name=args.name,
    )

    for name, report in results.items():
        print(f"\n{'=' * 60}")
        print(f"Dataset: {name} ({len(report.cases)} cases)")
        print("=" * 60)

        report.print()

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

    if not results:
        print("No datasets found.")


if __name__ == "__main__":
    asyncio.run(main())
