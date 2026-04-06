from __future__ import annotations

from pydantic import BaseModel


class EvalInput(BaseModel):
    query: str


class EvalOutput(BaseModel):
    answer: str | float | None = None
    numeric_answer: float | None = None
    sources: list[str] = []


class EvalMetadata(BaseModel):
    category: str
    company: str | None = None
    companies: list[str] | None = None
    metric: str | None = None
