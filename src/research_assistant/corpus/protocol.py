from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from research_assistant.corpus.models import Chunk, Document


@runtime_checkable
class DocumentParser(Protocol):
    def parse(self, ticker: str, filing_date: str) -> Document: ...


@runtime_checkable
class ChunkingStrategy(Protocol):
    def chunk(self, document: Document) -> list[Chunk]: ...
