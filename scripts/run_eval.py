from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from research_assistant.config import Settings, configure_logfire
from research_assistant.eval.models import EvalMetadata
from research_assistant.eval.runner import run_all_evals

logger = logging.getLogger(__name__)


async def main() -> None:
    settings = Settings()
    configure_logfire(settings)
    logging.basicConfig(level=settings.log_level)

    logger.info("Running eval suite with stub task (baseline)")
    results = await run_all_evals()

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
                sum(1 for a in case.assertions.values() if a.value)
                / len(case.assertions)
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
