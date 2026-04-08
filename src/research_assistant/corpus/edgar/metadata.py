from __future__ import annotations

from datetime import date

from research_assistant.corpus.models import Metadata


class EdgarMetadata(Metadata):
    ticker: str
    company_name: str
    filing_type: str
    period: str
    section_name: str
    filing_date: date
