from __future__ import annotations

import hashlib

from chonkie import RecursiveChunker

from research_assistant.corpus.edgar.metadata import EdgarMetadata
from research_assistant.corpus.models import Chunk, Document

SECTION_LABELS: dict[str, str] = {
    "Item 1": "Business",
    "Item 1A": "Risk Factors",
    "Item 1B": "Unresolved Staff Comments",
    "Item 2": "Properties",
    "Item 3": "Legal Proceedings",
    "Item 4": "Mine Safety Disclosures",
    "Item 5": "Market for Common Equity",
    "Item 6": "Selected Financial Data",
    "Item 7": "MD&A",
    "Item 7A": "Market Risk Disclosures",
    "Item 8": "Financial Statements",
    "Item 9": "Disagreements with Accountants",
    "Item 9A": "Controls and Procedures",
    "Item 9B": "Other Information",
    "Item 10": "Directors and Officers",
    "Item 11": "Executive Compensation",
    "Item 12": "Security Ownership",
    "Item 13": "Related Transactions",
    "Item 14": "Principal Accountant Fees",
    "Item 15": "Exhibits and Financial Statements",
}


def _build_context_prefix(metadata: EdgarMetadata, section_name: str) -> str:
    label = SECTION_LABELS.get(section_name, "")
    section_display = f"{section_name} ({label})" if label else section_name
    return (
        f"{metadata.company_name} ({metadata.ticker}) "
        f"{metadata.filing_type} FY{metadata.fiscal_year}, "
        f"{section_display}:\n\n"
    )


CONTEXT_PREFIX_TOKEN_BUDGET = 30


class EdgarChunker:
    def __init__(
        self,
        max_tokens: int = 512,
        tokenizer: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.max_tokens = max_tokens
        self._chunker = RecursiveChunker(
            tokenizer=tokenizer,
            chunk_size=max_tokens - CONTEXT_PREFIX_TOKEN_BUDGET,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_index = 0

        base_metadata = document.metadata
        assert isinstance(base_metadata, EdgarMetadata)

        for section_name, text in document.sections.items():
            prefix = _build_context_prefix(base_metadata, section_name)
            chonkie_chunks = self._chunker.chunk(text)

            for cc in chonkie_chunks:
                chunk_id = hashlib.sha256(
                    f"{document.id}:{section_name}:{chunk_index}".encode()
                ).hexdigest()[:16]

                chunk_metadata = base_metadata.model_copy(
                    update={"section_name": section_name},
                )

                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_id=document.id,
                        text=f"{prefix}{cc.text}",
                        section_name=section_name,
                        metadata=chunk_metadata,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

        return chunks
