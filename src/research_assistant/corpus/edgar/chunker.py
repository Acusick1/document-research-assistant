from __future__ import annotations

import hashlib

from research_assistant.corpus.edgar.metadata import EdgarMetadata
from research_assistant.corpus.models import Chunk, Document

CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _split_at_paragraphs(text: str, max_tokens: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)
        if current and current_tokens + para_tokens > max_tokens:
            chunks.append("\n\n".join(current))
            current = [para]
            current_tokens = para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))
    return chunks


class EdgarChunker:
    def __init__(self, max_tokens: int = 512) -> None:
        self.max_tokens = max_tokens

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_index = 0

        base_metadata = document.metadata
        assert isinstance(base_metadata, EdgarMetadata)

        for section_name, text in document.sections.items():
            if _estimate_tokens(text) <= self.max_tokens:
                text_parts = [text]
            else:
                text_parts = _split_at_paragraphs(text, self.max_tokens)

            for part in text_parts:
                chunk_id = hashlib.sha256(
                    f"{document.id}:{section_name}:{chunk_index}".encode()
                ).hexdigest()[:16]

                chunk_metadata = base_metadata.model_copy(
                    update={"section_name": section_name}
                )

                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_id=document.id,
                        text=part,
                        section_name=section_name,
                        metadata=chunk_metadata,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

        return chunks
