from __future__ import annotations

import logging

import pandas as pd
from diskcache import Cache
from edgar import Company, set_identity
from pydantic_evals import Case

from research_assistant.corpus.edgar.cache import facts_key
from research_assistant.eval.evaluators.numeric_match import NumericMatch
from research_assistant.eval.models import EvalInput, EvalMetadata, EvalOutput

logger = logging.getLogger(__name__)

FACTUAL_CONCEPTS = [
    ("revenue", "total revenue"),
    ("net_income", "net income"),
    ("total_assets", "total assets"),
    ("total_liabilities", "total liabilities"),
    ("operating_income", "operating income"),
    ("cost_of_revenue", "cost of revenue"),
    ("gross_profit", "gross profit"),
    ("research_and_development", "R&D spending"),
    ("stockholders_equity", "stockholders' equity"),
    ("operating_cash_flow", "operating cash flow"),
]

XBRL_TAG_MAP: dict[str, list[str]] = {
    "revenue": [
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:Revenues",
    ],
    "net_income": ["us-gaap:NetIncomeLoss"],
    "total_assets": ["us-gaap:Assets"],
    "total_liabilities": ["us-gaap:Liabilities"],
    "operating_income": ["us-gaap:OperatingIncomeLoss"],
    "cost_of_revenue": ["us-gaap:CostOfGoodsAndServicesSold", "us-gaap:CostOfRevenue"],
    "gross_profit": ["us-gaap:GrossProfit"],
    "research_and_development": ["us-gaap:ResearchAndDevelopmentExpense"],
    "stockholders_equity": [
        "us-gaap:StockholdersEquity",
        "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "operating_cash_flow": [
        "us-gaap:NetCashProvidedByUsedInOperatingActivities",
    ],
}


MIN_ANNUAL_DAYS = 300


def _get_company_facts(ticker: str, cache: Cache | None) -> tuple[str, pd.DataFrame] | None:
    key = facts_key(ticker)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            logger.info("Cache hit for %s XBRL facts", ticker)
            return cached["name"], cached["facts_df"]

    company = Company(ticker)
    name: str = company.name
    facts = company.get_facts()
    if facts is None:
        return None
    facts_df: pd.DataFrame = facts.to_dataframe()

    if cache is not None:
        cache.set(key, {"name": name, "facts_df": facts_df})

    return name, facts_df


def _get_annual_values(
    facts_df: pd.DataFrame,
    concept: str,
    min_year: int | None = None,
) -> dict[str, float]:
    tags = XBRL_TAG_MAP.get(concept, [])
    if not tags:
        return {}

    usd_rows = facts_df[(facts_df["unit"] == "USD") & (facts_df["concept"].isin(tags))].copy()

    if usd_rows.empty:
        return {}

    has_start = usd_rows["period_start"].notna()
    # Duration items (income statement, cash flow): filter by span >= 300 days
    duration_rows = usd_rows[has_start].copy()
    if not duration_rows.empty:
        start = pd.to_datetime(duration_rows["period_start"])
        end = pd.to_datetime(duration_rows["period_end"])
        duration_rows = duration_rows[(end - start).dt.days >= MIN_ANNUAL_DAYS]
        usd_rows = duration_rows
    else:
        # Instant items (balance sheet): use fiscal_period == FY to filter
        usd_rows = usd_rows[usd_rows["fiscal_period"] == "FY"]

    if usd_rows.empty:
        return {}

    if min_year is not None:
        usd_rows = usd_rows[pd.to_datetime(usd_rows["period_end"]).dt.year >= min_year]

    # Take max per period_end to get totals (not segments), then deduplicate by fiscal year
    deduped = usd_rows.groupby("period_end")["numeric_value"].max()
    by_fy: dict[str, float] = {}
    for period_end, value in sorted(deduped.items()):
        fy = _fiscal_year_label(str(period_end))
        by_fy[fy] = float(value)
    return by_fy


def _fiscal_year_label(period_end: str) -> str:
    year = period_end[:4]
    return f"FY{year}"


def generate_factual_cases(
    tickers: list[str],
    identity: str = "ResearchAssistant research@example.com",
    min_year: int = 2022,
    cache: Cache | None = None,
) -> list[Case[EvalInput, EvalOutput, EvalMetadata]]:
    set_identity(identity)
    cases: list[Case[EvalInput, EvalOutput, EvalMetadata]] = []

    for ticker in tickers:
        result = _get_company_facts(ticker, cache)
        if result is None:
            logger.warning("No XBRL facts for %s, skipping", ticker)
            continue
        name, facts_df = result

        for concept, label in FACTUAL_CONCEPTS:
            annual = _get_annual_values(facts_df, concept, min_year=min_year)
            if not annual:
                logger.debug("No annual data for %s/%s", ticker, concept)
                continue

            for fy, value in annual.items():
                case = Case(
                    name=f"{ticker.lower()}_{concept}_{fy.lower()}",
                    inputs=EvalInput(
                        query=f"What was {name}'s {label} in {fy}?",
                    ),
                    expected_output=EvalOutput(
                        numeric_answer=value,
                        answer=str(value),
                    ),
                    metadata=EvalMetadata(
                        category="factual",
                        company=ticker.upper(),
                        metric=concept,
                    ),
                    evaluators=(NumericMatch(tolerance_pct=0.05),),
                )
                cases.append(case)

    _log_coverage(cases, tickers)
    return cases


def _log_coverage(
    cases: list[Case[EvalInput, EvalOutput, EvalMetadata]],
    tickers: list[str],
) -> None:
    if not tickers:
        return

    by_ticker: dict[str, set[str]] = {}
    for case in cases:
        meta = case.metadata
        if meta and meta.company:
            by_ticker.setdefault(meta.company, set()).add(meta.metric or "unknown")

    total_concepts = len(FACTUAL_CONCEPTS)
    logger.info(
        "Generated %d cases across %d tickers (%d concepts each)",
        len(cases),
        len(by_ticker),
        total_concepts,
    )

    for ticker in tickers:
        ticker_upper = ticker.upper()
        metrics = by_ticker.get(ticker_upper, set())
        missing = [concept for concept, _ in FACTUAL_CONCEPTS if concept not in metrics]
        if missing:
            logger.warning(
                "%s: missing %d metrics: %s",
                ticker_upper,
                len(missing),
                ", ".join(missing),
            )
        else:
            logger.info("%s: all %d metrics covered", ticker_upper, len(metrics))


def generate_comparison_cases(
    tickers: list[str],
    identity: str = "ResearchAssistant research@example.com",
    cache: Cache | None = None,
) -> list[Case[EvalInput, EvalOutput, EvalMetadata]]:
    set_identity(identity)

    # Collect the latest annual value per company per concept
    company_values: dict[str, dict[str, tuple[float, str, str]]] = {}
    for ticker in tickers:
        result = _get_company_facts(ticker, cache)
        if result is None:
            logger.warning("No XBRL facts for %s, skipping", ticker)
            continue
        name, facts_df = result

        values: dict[str, tuple[float, str, str]] = {}
        for concept, _ in FACTUAL_CONCEPTS:
            annual = _get_annual_values(facts_df, concept)
            if annual:
                latest_fy = max(annual.keys())
                values[concept] = (annual[latest_fy], name, latest_fy)
        company_values[ticker.upper()] = values

    cases: list[Case[EvalInput, EvalOutput, EvalMetadata]] = []
    ticker_list = list(company_values.keys())

    for i in range(len(ticker_list)):
        for j in range(i + 1, len(ticker_list)):
            t1, t2 = ticker_list[i], ticker_list[j]
            v1, v2 = company_values[t1], company_values[t2]

            for concept, label in FACTUAL_CONCEPTS[:5]:
                if concept not in v1 or concept not in v2:
                    continue

                val1, name1, fy1 = v1[concept]
                val2, name2, fy2 = v2[concept]

                higher = name1 if val1 > val2 else name2
                higher_ticker = t1 if val1 > val2 else t2

                case = Case(
                    name=f"{t1.lower()}_vs_{t2.lower()}_{concept}",
                    inputs=EvalInput(
                        query=(
                            f"Which company had higher {label}, {name1} ({fy1}) or {name2} ({fy2})?"
                        ),
                    ),
                    expected_output=EvalOutput(answer=higher),
                    metadata=EvalMetadata(
                        category="comparison",
                        companies=[t1, t2],
                        company=higher_ticker,
                        metric=concept,
                    ),
                )
                cases.append(case)

    return cases
