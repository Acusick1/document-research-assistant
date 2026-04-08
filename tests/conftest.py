from __future__ import annotations

from datetime import date

import pytest

from research_assistant.config import Settings
from research_assistant.corpus.edgar.metadata import EdgarMetadata
from research_assistant.corpus.models import Chunk, Document


@pytest.fixture
def settings() -> Settings:
    return Settings(qdrant_mode="memory", log_level="DEBUG")


@pytest.fixture
def sample_metadata() -> EdgarMetadata:
    return EdgarMetadata(
        source="edgar",
        ticker="AAPL",
        company_name="Apple Inc.",
        filing_type="10-K",
        period="FY2024",
        section_name="Item 1",
        filing_date=date(2024, 11, 1),
    )


@pytest.fixture
def sample_document(sample_metadata: EdgarMetadata) -> Document:
    return Document(
        id="AAPL_10K_2024",
        source="edgar",
        sections={
            "Item 1": "Apple designs and manufactures consumer electronics.",
            "Item 1A": "Risk factors include supply chain disruptions and competition.",
            "Item 7": "Revenue increased year over year driven by iPhone and Services.",
        },
        metadata=sample_metadata,
        raw_text="Apple designs and manufactures consumer electronics.",
    )


@pytest.fixture
def sample_chunks(sample_document: Document) -> list[Chunk]:
    meta = sample_document.metadata
    assert isinstance(meta, EdgarMetadata)
    return [
        Chunk(
            id="chunk_0",
            document_id=sample_document.id,
            text="Apple designs and manufactures consumer electronics.",
            section_name="Item 1",
            metadata=meta.model_copy(update={"section_name": "Item 1"}),
            chunk_index=0,
        ),
        Chunk(
            id="chunk_1",
            document_id=sample_document.id,
            text="Risk factors include supply chain disruptions.",
            section_name="Item 1A",
            metadata=meta.model_copy(update={"section_name": "Item 1A"}),
            chunk_index=1,
        ),
    ]
