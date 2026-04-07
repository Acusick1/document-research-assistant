from __future__ import annotations

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from research_assistant.agents.simple import EDGAR_INSTRUCTIONS, AgentResponse


class TestAgentResponse:
    def test_minimal(self) -> None:
        resp = AgentResponse(
            answer="Revenue was $100B",
            reasoning="Found in Item 7",
            confidence=0.9,
        )
        assert resp.numeric_answer is None
        assert resp.cited_sections == []

    def test_with_numeric(self) -> None:
        resp = AgentResponse(
            answer="$93.7 billion",
            numeric_answer=93_736_000_000.0,
            reasoning="Extracted from income statement",
            cited_sections=["AAPL 10-K FY2024 Item 8"],
            confidence=0.95,
        )
        assert resp.numeric_answer == 93_736_000_000.0
        assert len(resp.cited_sections) == 1

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            AgentResponse(answer="x", reasoning="y", confidence=1.5)
        with pytest.raises(ValidationError):
            AgentResponse(answer="x", reasoning="y", confidence=-0.1)


class TestCreateAgent:
    def test_default_system_prompt(self) -> None:
        agent = Agent(TestModel(), system_prompt=EDGAR_INSTRUCTIONS, output_type=AgentResponse)
        assert EDGAR_INSTRUCTIONS in agent._system_prompts
