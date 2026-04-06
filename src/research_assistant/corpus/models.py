from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class Metadata(BaseModel):
    source: str


class Document(BaseModel):
    id: str
    source: str
    sections: dict[str, str]
    metadata: Metadata
    raw_text: str


class Chunk(BaseModel):
    id: str
    document_id: str
    text: str
    section_name: str
    metadata: Metadata
    chunk_index: int


class XBRLFact(BaseModel):
    concept: str
    value: float
    unit: str
    period_end: date
    fiscal_year: int
    fiscal_period: str
