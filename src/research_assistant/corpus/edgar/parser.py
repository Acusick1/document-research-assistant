from __future__ import annotations

import logging
from datetime import date

from edgar import Company, set_identity
from edgar.company_reports.ten_k import TenK
from edgar.entity.filings import EntityFiling, EntityFilings

from research_assistant.corpus.edgar.cache import EdgarCache, FilingCacheEntry
from research_assistant.corpus.edgar.metadata import EdgarMetadata
from research_assistant.corpus.models import Document

logger = logging.getLogger(__name__)

# Standard 10-K sections mandated by the SEC (Items 1-16).
# This subset covers business overview, risks, MD&A, and financials.
TARGET_ITEMS = ["Item 1", "Item 1A", "Item 7", "Item 7A", "Item 8"]


class EdgarParser:
    def __init__(
        self,
        identity: str = "ResearchAssistant research@example.com",
        cache: EdgarCache | None = None,
    ) -> None:
        set_identity(identity)
        self._cache = cache

    def parse(self, ticker: str, year: int) -> Document:
        cached = self._cache.get_filing(ticker, year) if self._cache else None
        if cached is not None:
            logger.info("Cache hit for %s FY%d", ticker, year)
            return self._build_document(ticker, year, cached)

        sections, filing_date = self._fetch_filing(ticker, year)
        data: FilingCacheEntry = {"sections": sections, "filing_date": filing_date.isoformat()}

        if self._cache is not None:
            self._cache.put_filing(ticker, year, data)

        return self._build_document(ticker, year, data)

    def _fetch_filing(self, ticker: str, year: int) -> tuple[dict[str, str], date]:
        company = Company(ticker)
        filings = company.get_filings(form="10-K")

        filing = self._find_filing_for_year(filings, year)
        obj = filing.obj()
        if not isinstance(obj, TenK):
            msg = f"Expected TenK filing, got {type(obj).__name__}"
            raise TypeError(msg)
        tenk = obj

        sections: dict[str, str] = {}
        for item_name in TARGET_ITEMS:
            try:
                text = tenk[item_name]
                if text:
                    sections[item_name] = str(text).strip()
            except (KeyError, IndexError):
                continue

        filing_date = date.fromisoformat(str(filing.filing_date))
        return sections, filing_date

    def _build_document(self, ticker: str, year: int, data: FilingCacheEntry) -> Document:
        sections: dict[str, str] = data["sections"]
        filing_date = date.fromisoformat(data["filing_date"])

        metadata = EdgarMetadata(
            source="edgar",
            ticker=ticker.upper(),
            filing_type="10-K",
            period=f"FY{year}",
            section_name="",
            filing_date=filing_date,
        )

        raw_text = "\n\n".join(f"## {name}\n\n{text}" for name, text in sections.items())
        doc_id = f"{ticker.upper()}_10K_{year}"

        return Document(
            id=doc_id,
            source="edgar",
            sections=sections,
            metadata=metadata,
            raw_text=raw_text,
        )

    def _find_filing_for_year(self, filings: EntityFilings, year: int) -> EntityFiling:
        for filing in filings:
            filing_date = date.fromisoformat(str(filing.filing_date))
            if filing_date.year == year or filing_date.year == year + 1:
                obj = filing.obj()
                if not isinstance(obj, TenK):
                    continue
                period = obj.period_of_report
                if period and date.fromisoformat(str(period)).year == year:
                    return filing
        msg = f"No 10-K filing found for fiscal year {year}"
        raise ValueError(msg)
