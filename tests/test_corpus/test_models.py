from __future__ import annotations

from datetime import date

from research_assistant.corpus.edgar.metadata import EdgarMetadata
from research_assistant.corpus.models import Chunk, Document


class TestDocument:
    def test_create_document(self) -> None:
        meta = EdgarMetadata(
            source="edgar",
            ticker="AAPL",
            filing_type="10-K",
            period="FY2024",
            section_name="",
            filing_date=date(2024, 11, 1),
        )
        doc = Document(
            id="AAPL_10K_2024",
            source="edgar",
            sections={"Item 1": "Business description"},
            metadata=meta,
            raw_text="Business description",
        )
        assert doc.id == "AAPL_10K_2024"
        assert "Item 1" in doc.sections
        assert doc.metadata.source == "edgar"

    def test_document_multiple_sections(self) -> None:
        meta = EdgarMetadata(
            source="edgar",
            ticker="MSFT",
            filing_type="10-K",
            period="FY2024",
            section_name="",
            filing_date=date(2024, 10, 30),
        )
        doc = Document(
            id="MSFT_10K_2024",
            source="edgar",
            sections={
                "Item 1": "Microsoft develops software.",
                "Item 7": "Revenue grew significantly.",
            },
            metadata=meta,
            raw_text="",
        )
        assert len(doc.sections) == 2


class TestChunk:
    def test_create_chunk(self, sample_metadata: EdgarMetadata) -> None:
        chunk = Chunk(
            id="abc123",
            document_id="AAPL_10K_2024",
            text="Some text content",
            section_name="Item 1",
            metadata=sample_metadata,
            chunk_index=0,
        )
        assert chunk.chunk_index == 0
        assert chunk.section_name == "Item 1"
