from __future__ import annotations

from research_assistant.corpus.edgar.chunker import (
    EdgarChunker,
    _estimate_tokens,
    _split_at_paragraphs,
)
from research_assistant.corpus.models import Document


class TestEstimateTokens:
    def test_basic_estimate(self) -> None:
        assert _estimate_tokens("abcd") == 1
        assert _estimate_tokens("abcdefgh") == 2

    def test_empty(self) -> None:
        assert _estimate_tokens("") == 0


class TestSplitAtParagraphs:
    def test_single_paragraph_under_limit(self) -> None:
        text = "Short paragraph."
        result = _split_at_paragraphs(text, max_tokens=100)
        assert len(result) == 1
        assert result[0] == "Short paragraph."

    def test_multiple_paragraphs_split(self) -> None:
        para1 = "a" * 400  # 100 tokens
        para2 = "b" * 400  # 100 tokens
        text = f"{para1}\n\n{para2}"
        result = _split_at_paragraphs(text, max_tokens=150)
        assert len(result) == 2

    def test_paragraphs_merged_when_under_limit(self) -> None:
        text = "Short one.\n\nShort two.\n\nShort three."
        result = _split_at_paragraphs(text, max_tokens=100)
        assert len(result) == 1


class TestEdgarChunker:
    def test_small_sections_stay_whole(self, sample_document: Document) -> None:
        chunker = EdgarChunker(max_tokens=5000)
        chunks = chunker.chunk(sample_document)
        assert len(chunks) == len(sample_document.sections)

    def test_large_section_gets_split(self) -> None:
        from datetime import date

        from research_assistant.corpus.edgar.metadata import EdgarMetadata

        long_text = "\n\n".join(f"Paragraph {i} " * 50 for i in range(20))
        meta = EdgarMetadata(
            source="edgar",
            ticker="AAPL",
            filing_type="10-K",
            period="FY2024",
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
        chunker = EdgarChunker(max_tokens=512)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.section_name == "Item 1"
            assert chunk.document_id == "test"

    def test_chunk_ids_are_unique(self, sample_document: Document) -> None:
        chunker = EdgarChunker()
        chunks = chunker.chunk(sample_document)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_metadata_inherits_from_document(self, sample_document: Document) -> None:
        chunker = EdgarChunker()
        chunks = chunker.chunk(sample_document)
        for chunk in chunks:
            from research_assistant.corpus.edgar.metadata import EdgarMetadata
            assert isinstance(chunk.metadata, EdgarMetadata)
            assert chunk.metadata.ticker == "AAPL"
