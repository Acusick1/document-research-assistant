from __future__ import annotations

from datetime import date

from research_assistant.corpus.edgar.chunker import (
    CONTEXT_PREFIX_TOKEN_BUDGET,
    EdgarChunker,
    _build_context_prefix,
)
from research_assistant.corpus.edgar.metadata import EdgarMetadata
from research_assistant.corpus.models import Document


class TestBuildContextPrefix:
    def test_known_section(self) -> None:
        meta = EdgarMetadata(
            source="edgar",
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_type="10-K",
            fiscal_year=2024,
            section_name="",
            filing_date=date(2024, 11, 1),
        )
        prefix = _build_context_prefix(meta, "Item 7")
        assert "Apple Inc. (AAPL)" in prefix
        assert "FY2024" in prefix
        assert "Item 7 (MD&A)" in prefix

    def test_unknown_section(self) -> None:
        meta = EdgarMetadata(
            source="edgar",
            ticker="MSFT",
            company_name="Microsoft Corporation",
            filing_type="10-K",
            fiscal_year=2023,
            section_name="",
            filing_date=date(2023, 7, 1),
        )
        prefix = _build_context_prefix(meta, "Custom Section")
        assert "Custom Section" in prefix
        assert "()" not in prefix


class TestEdgarChunker:
    def test_small_sections_stay_whole(self, sample_document: Document) -> None:
        chunker = EdgarChunker(max_tokens=5000, tokenizer="character")
        chunks = chunker.chunk(sample_document)
        assert len(chunks) == len(sample_document.sections)

    def test_large_section_gets_split(self) -> None:
        long_text = "\n\n".join(f"Paragraph {i}. " + "word " * 100 for i in range(20))
        meta = EdgarMetadata(
            source="edgar",
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_type="10-K",
            fiscal_year=2024,
            section_name="",
            filing_date=date(2024, 11, 1),
        )
        doc = Document(
            id="test",
            source="edgar",
            sections={"Item 1": long_text},
            metadata=meta,
            raw_text=long_text,
        )
        max_tokens = 512
        content_budget = max_tokens - CONTEXT_PREFIX_TOKEN_BUDGET
        chunker = EdgarChunker(max_tokens=max_tokens, tokenizer="character")
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1
        prefix = _build_context_prefix(meta, "Item 1")
        for chunk in chunks:
            assert chunk.section_name == "Item 1"
            assert chunk.document_id == "test"
            content = chunk.text.removeprefix(prefix)
            assert len(content) <= content_budget

    def test_chunk_ids_are_unique(self, sample_document: Document) -> None:
        chunker = EdgarChunker(max_tokens=5000, tokenizer="character")
        chunks = chunker.chunk(sample_document)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_metadata_inherits_from_document(self, sample_document: Document) -> None:
        chunker = EdgarChunker(max_tokens=5000, tokenizer="character")
        chunks = chunker.chunk(sample_document)
        for chunk in chunks:
            assert isinstance(chunk.metadata, EdgarMetadata)
            assert chunk.metadata.ticker == "AAPL"

    def test_context_prefix_prepended(self, sample_document: Document) -> None:
        chunker = EdgarChunker(max_tokens=5000, tokenizer="character")
        chunks = chunker.chunk(sample_document)
        for chunk in chunks:
            assert chunk.text.startswith("Apple Inc. (AAPL)")
            assert "FY2024" in chunk.text
