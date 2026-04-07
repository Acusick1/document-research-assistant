from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pydantic_evals import Dataset

from research_assistant.config import get_settings
from research_assistant.corpus.edgar.cache import create_cache
from research_assistant.eval.generate import generate_comparison_cases, generate_factual_cases
from research_assistant.eval.models import EvalInput, EvalMetadata, EvalOutput

logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Generate eval datasets from XBRL data")
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols")
    parser.add_argument("--identity", default="ResearchAssistant research@example.com")
    parser.add_argument("--output-dir", type=Path, default=settings.datasets_dir)
    parser.add_argument("--min-year", type=int, default=2022)
    parser.add_argument(
        "--no-cache", action="store_true", help="Bypass the disk cache for EDGAR API responses"
    )
    parser.add_argument(
        "--max-comparison-pairs", type=int, default=3,
        help="Max ticker pairs per concept in comparison cases",
    )
    args = parser.parse_args()

    logging.basicConfig(level="INFO")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache = None if args.no_cache else create_cache(settings.cache_dir)

    logger.info("Generating factual cases for %s", args.tickers)
    factual_cases = generate_factual_cases(
        args.tickers, identity=args.identity, min_year=args.min_year, cache=cache
    )
    logger.info("Generated %d factual cases", len(factual_cases))

    factual_ds = Dataset[EvalInput, EvalOutput, EvalMetadata](name="factual", cases=factual_cases)
    factual_path = args.output_dir / "factual.yaml"
    factual_ds.to_file(factual_path)
    logger.info("Wrote %s", factual_path)

    logger.info("Generating comparison cases for %s", args.tickers)
    comparison_cases = generate_comparison_cases(
        args.tickers, identity=args.identity, min_year=args.min_year,
        cache=cache, max_pairs_per_concept=args.max_comparison_pairs,
    )
    logger.info("Generated %d comparison cases", len(comparison_cases))

    comparison_ds = Dataset[EvalInput, EvalOutput, EvalMetadata](
        name="comparison", cases=comparison_cases
    )
    comparison_path = args.output_dir / "comparison.yaml"
    comparison_ds.to_file(comparison_path)
    logger.info("Wrote %s", comparison_path)


if __name__ == "__main__":
    main()
