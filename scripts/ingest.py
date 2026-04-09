from __future__ import annotations

import argparse
import logging

from research_assistant.config import Settings, configure_logfire
from research_assistant.corpus.edgar.cache import create_cache
from research_assistant.corpus.edgar.chunker import EdgarChunker
from research_assistant.corpus.edgar.parser import EdgarParser
from research_assistant.retrieval.embeddings import FastEmbedEmbedder
from research_assistant.retrieval.ingest import ingest_chunks
from research_assistant.retrieval.vector_store import QdrantStore, create_qdrant_client

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest EDGAR filings into Qdrant")
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols")
    parser.add_argument(
        "--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025, 2026],
        help="Fiscal years (failures for unavailable years are logged and skipped)",
    )
    parser.add_argument("--identity", default="ResearchAssistant research@example.com")
    parser.add_argument(
        "--no-cache", action="store_true", help="Bypass the disk cache for EDGAR API responses"
    )
    parser.add_argument(
        "--fresh", action="store_true", help="Delete and recreate the collection before ingesting"
    )
    args = parser.parse_args()

    settings = Settings()
    configure_logfire(settings)
    logging.basicConfig(level=settings.log_level)

    cache = None if args.no_cache else create_cache(settings.cache_dir)
    edgar_parser = EdgarParser(identity=args.identity, cache=cache)
    chunker = EdgarChunker(max_tokens=settings.chunk_max_tokens)
    embedder = FastEmbedEmbedder(model_name=settings.embedding_model)
    client = create_qdrant_client(settings)
    store = QdrantStore(client, settings.collection_name, embedder.dim)
    if args.fresh:
        logger.info("Deleting collection '%s' for fresh ingestion", settings.collection_name)
        client.delete_collection(settings.collection_name)
    store.ensure_collection()

    total_chunks = 0
    for ticker in args.tickers:
        for year in args.years:
            logger.info("Parsing %s 10-K for FY%d", ticker, year)
            try:
                doc = edgar_parser.parse(ticker, year)
            except Exception as e:
                logger.warning("Failed to parse %s FY%d: %s", ticker, year, e)
                continue

            chunks = chunker.chunk(doc)
            logger.info("  %d chunks from %s FY%d", len(chunks), ticker, year)
            total_chunks += ingest_chunks(chunks, embedder, store)

    logger.info(
        "Ingested %d total chunks into collection '%s'", total_chunks, settings.collection_name
    )
    logger.info("Collection count: %d", store.count())


if __name__ == "__main__":
    main()
