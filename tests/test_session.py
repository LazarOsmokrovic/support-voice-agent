"""Phase 7: tests for the session/turn orchestration extracted out of
transport/text_cli.py into agent/session.py. This is the logic every
transport (text, and now voice) shares — tested once here rather than
implicitly through whichever transport happens to call it.

Escalation's own LLM call (classify_turn) is monkeypatched directly rather
than mocking a second Anthropic client through the full chain — same
approach test_escalation.py already uses to test EscalationTracker against
fabricated classifications without a real call.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.session import create_session, run_turn
from agent.tools import escalation
from agent.tools.escalation import TurnClassification
from data import mock_db


def _text_response(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.stop_reason = "end_turn"
    return response


def _tool_use_response(name: str, tool_input: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.id = "toolu_test123"
    block.name = name
    block.input = tool_input
    response = MagicMock()
    response.content = [block]
    response.stop_reason = "tool_use"
    return response


def _calm_classification():
    return TurnClassification(intent="chitchat", sentiment="neutral", policy_restricted=False)


def test_create_session_registers_every_tool():
    from agent.session import TOOLS

    session = create_session("CUST-1001")

    tool_names = {t["name"] for t in TOOLS}
    assert tool_names == set(session.handlers.keys())


@pytest.mark.asyncio
async def test_run_turn_returns_reply_and_does_not_end_by_default(monkeypatch):
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=_text_response("Happy to help!"))
    monkeypatch.setattr(escalation, "classify_turn", AsyncMock(return_value=_calm_classification()))
    session = create_session("CUST-1001", client=fake_client)

    outcome = await run_turn(session, "Hi there")

    assert outcome.reply == "Happy to help!"
    assert outcome.ended is False
    assert outcome.end_reason is None
    assert outcome.notice is None
    assert outcome.llm_latency_seconds >= 0
    assert outcome.warnings == []


@pytest.mark.asyncio
async def test_run_turn_detects_the_model_ending_the_conversation(monkeypatch):
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(
        side_effect=[
            _tool_use_response("end_conversation", {}),
            _text_response("Take care!"),
        ]
    )
    monkeypatch.setattr(escalation, "classify_turn", AsyncMock(return_value=_calm_classification()))
    session = create_session("CUST-1001", client=fake_client)

    outcome = await run_turn(session, "Thanks, that's all!")

    assert outcome.reply == "Take care!"
    assert outcome.ended is True
    assert outcome.end_reason == "model_ended"
    assert outcome.notice is None


@pytest.mark.asyncio
async def test_run_turn_escalates_and_produces_a_notice(monkeypatch, tmp_path):
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_session.db")
    mock_db.reset_and_seed()

    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=_text_response("Sure, one moment."))
    monkeypatch.setattr(
        escalation, "classify_turn",
        AsyncMock(return_value=TurnClassification(intent="request_human", sentiment="neutral", policy_restricted=False)),
    )
    monkeypatch.setattr(
        escalation, "create_handoff_packet",
        AsyncMock(return_value={"escalation_id": 42, "reason": "explicit request for a human"}),
    )
    session = create_session("CUST-1001", client=fake_client)

    outcome = await run_turn(session, "I want to talk to a human")

    assert outcome.ended is True
    assert outcome.end_reason == "escalated"
    assert outcome.notice is not None
    assert "handoff #42" in outcome.notice


@pytest.mark.asyncio
async def test_run_turn_survives_a_classifier_failure_without_crashing(monkeypatch):
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=_text_response("Here you go."))
    monkeypatch.setattr(escalation, "classify_turn", AsyncMock(side_effect=RuntimeError("no credit")))
    session = create_session("CUST-1001", client=fake_client)

    outcome = await run_turn(session, "Hi")

    assert outcome.reply == "Here you go."
    assert outcome.ended is False
    assert any("no credit" in w for w in outcome.warnings)
