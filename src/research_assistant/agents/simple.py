from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent

EDGAR_INSTRUCTIONS = """\
You are a financial research assistant specializing in SEC EDGAR filings (10-K and 10-Q reports).

Answer questions using ONLY the provided context from retrieved filing sections. Do not rely on \
prior knowledge about company financials — if the context doesn't contain the information, say so.

Rules:
- For numerical questions, extract the EXACT figure from the filings. Do not round.
- Always populate numeric_answer when the answer is a number (revenue, income, cash flow, etc.).
- For comparison questions, state which entity has the higher/lower value and by how much.
- For trend questions, describe the direction and approximate magnitude of changes over time.
- Cite the specific filing sections and companies you used in cited_sections.
- If multiple filings are relevant, synthesize across them.
- Keep your reasoning concise but show your work for calculations.\
"""


class AgentResponse(BaseModel):
    answer: str = Field(description="Direct answer to the question")
    numeric_answer: float | None = Field(
        default=None,
        description=(
            "Extracted numeric value when the answer is a number"
            " (in original units, e.g. dollars not millions)"
        ),
    )
    reasoning: str = Field(description="Brief explanation of how you arrived at the answer")
    cited_sections: list[str] = Field(
        default_factory=list,
        description="List of source identifiers used (e.g. 'AAPL 10-K FY2024 Item 7')",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the answer from 0 (no confidence) to 1 (certain)",
    )


def create_agent(model: str = "anthropic:claude-sonnet-4-6") -> Agent[None, AgentResponse]:
    return Agent(  # type: ignore[return-value]
        model,
        system_prompt=EDGAR_INSTRUCTIONS,
        output_type=AgentResponse,
    )
