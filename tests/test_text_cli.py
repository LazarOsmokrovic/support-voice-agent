"""Phase 1/2/3/4 checkpoints for the REPL: tools are actually wired into the
Phase 0 tool-use loop, not just callable on their own, and the loop knows
to stop when the model signals the conversation is over.

Uses a mocked Claude client (no network, no API key) that scripts a
two-turn exchange: first Claude asks for a tool, then it replies with text
once the tool result comes back — exercising Agent.send end to end with the
real dispatch_tool from transport/text_cli.py.

The live tests (gated on a real ANTHROPIC_API_KEY) are the full-pipeline
checkpoints: Phase 3's (a genuinely uncovered policy question, checking the
model doesn't invent an answer once retrieval correctly comes back empty)
and Phase 4's (scripted conversations, checking escalation fires neither
too eagerly nor too late) — both run through the actual Agent + real Claude
+ real tools (local embedding backend for search_policy — no Voyage key
needed).
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core import Agent
from agent.prompts import SYSTEM_PROMPT
from agent.tools import escalation
from data import mock_db
from transport.text_cli import TOOL_HANDLERS, TOOLS, dispatch_tool, should_end_session


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


def _text_response(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.stop_reason = "end_turn"
    return response


@pytest.mark.asyncio
async def test_order_status_tool_is_wired_into_the_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_wiring.db")
    mock_db.reset_and_seed()
    order_id = mock_db.ORDERS[0][0]

    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(
        side_effect=[
            _tool_use_response("get_order_status", {"order_id": order_id}),
            _text_response("Your order has shipped!"),
        ]
    )

    agent = Agent(client=fake_client, tools=TOOLS, tool_executor=dispatch_tool)
    result = await agent.send(f"Where's my order {order_id}?")

    assert result.reply == "Your order has shipped!"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "get_order_status"
    assert result.tool_calls[0]["input"] == {"order_id": order_id}
    assert result.tool_calls[0]["output"]["found"] is True  # Phase 4 needs the raw output, not just name/input
    assert fake_client.messages.create.await_count == 2


@pytest.mark.asyncio
async def test_end_conversation_tool_is_wired_into_the_loop():
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(
        side_effect=[
            _tool_use_response("end_conversation", {}),
            _text_response("Glad that's sorted — have a great day!"),
        ]
    )

    agent = Agent(client=fake_client, tools=TOOLS, tool_executor=dispatch_tool)
    result = await agent.send("Thanks, that's all, have a nice day!")

    assert result.reply == "Glad that's sorted — have a great day!"
    assert should_end_session(result.tool_calls) is True


def test_should_end_session_is_false_without_end_conversation_call():
    tool_calls = [{"name": "get_order_status", "input": {"order_id": "112-3487561-2938471"}}]
    assert should_end_session(tool_calls) is False


def test_should_end_session_is_false_with_no_tool_calls():
    assert should_end_session([]) is False


@pytest.mark.asyncio
async def test_search_policy_tool_is_wired_into_the_loop(monkeypatch):
    # Stub the handler rather than hit the real retrieval pipeline — this
    # test is about the wiring (does a search_policy tool_use call actually
    # reach transport.text_cli's dispatch_tool), not retrieval quality,
    # which tests/test_policy_rag.py already covers on its own.
    monkeypatch.setitem(
        TOOL_HANDLERS,
        "search_policy",
        lambda query: {
            "found": True,
            "results": [{"source": "returns_policy.md", "title": "Returns Policy", "text": "...", "distance": 0.1}],
        },
    )
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(
        side_effect=[
            _tool_use_response("search_policy", {"query": "How long do I have to return something?"}),
            _text_response("You have 30 days to return most items."),
        ]
    )

    agent = Agent(client=fake_client, tools=TOOLS, tool_executor=dispatch_tool)
    result = await agent.send("How long do I have to return something?")

    assert result.reply == "You have 30 days to return most items."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "search_policy"
    assert result.tool_calls[0]["input"] == {"query": "How long do I have to return something?"}
    assert result.tool_calls[0]["output"]["found"] is True


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="requires a real ANTHROPIC_API_KEY to hit the live Claude API",
)
@pytest.mark.asyncio
async def test_agent_does_not_invent_an_answer_for_an_uncovered_policy_question():
    """Phase 3 checkpoint, full pipeline: a real Claude call, asked a policy
    question none of the 16 real policy docs cover, correctly calls
    search_policy, sees found: False (verified separately and more directly
    in tests/test_policy_rag.py), and doesn't fabricate a confident answer
    instead. The keyword check below is a best-effort automated proxy for
    "didn't hallucinate," not a full substitute for reading the reply —
    this was also checked by hand in the live REPL, see the phase write-up.
    """
    agent = Agent(system=SYSTEM_PROMPT, tools=TOOLS, tool_executor=dispatch_tool)

    result = await agent.send("Do you offer price matching with other stores?")

    search_calls = [c for c in result.tool_calls if c["name"] == "search_policy"]
    assert search_calls, "expected the model to call search_policy for a policy question"

    honest_signal = any(
        phrase in result.reply.lower()
        for phrase in [
            "don't have",
            "not sure",
            "check",
            "get back to",
            "no information",
            "not something",
            "don't currently",
            "unable to find",
        ]
    )
    assert honest_signal, f"reply doesn't read like an honest 'don't know' — got: {result.reply!r}"


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="requires a real ANTHROPIC_API_KEY to hit the live Claude API"
)
@pytest.mark.asyncio
async def test_escalation_fires_immediately_on_explicit_human_request():
    """Phase 4 checkpoint — not too late: an explicit ask for a human
    escalates on the very first turn, not after several more exchanges."""
    agent = Agent(system=SYSTEM_PROMPT, tools=TOOLS, tool_executor=dispatch_tool)
    tracker = escalation.EscalationTracker()

    result = await agent.send("I don't want to talk to a bot, please connect me with a real person.")
    reason = await escalation.check_escalation(tracker, agent.messages, result.tool_calls)

    assert reason == "explicit request for a human"


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="requires a real ANTHROPIC_API_KEY to hit the live Claude API"
)
@pytest.mark.asyncio
async def test_escalation_fires_on_sustained_frustration_not_on_the_first_complaint():
    """Phase 4 checkpoint — neither too eager nor too late: one grumpy
    message shouldn't escalate on its own, but frustration sustained across
    turns should, by the second consecutive negative turn."""
    agent = Agent(system=SYSTEM_PROMPT, tools=TOOLS, tool_executor=dispatch_tool)
    tracker = escalation.EscalationTracker()

    turns = [
        "My order is 3 days late, that's kind of annoying.",
        "This is ridiculous, it's been late every single time and nobody seems to care.",
    ]
    reasons = []
    for turn in turns:
        result = await agent.send(turn)
        reasons.append(await escalation.check_escalation(tracker, agent.messages, result.tool_calls))

    assert reasons[0] is None, "a single mildly annoyed message shouldn't escalate on its own"
    assert reasons[1] == "sustained negative sentiment across multiple turns"


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="requires a real ANTHROPIC_API_KEY to hit the live Claude API"
)
@pytest.mark.asyncio
async def test_escalation_never_fires_for_a_calm_satisfied_conversation():
    """Phase 4 checkpoint — not too eager: an ordinary, friendly
    conversation should never trip any escalation trigger."""
    agent = Agent(system=SYSTEM_PROMPT, tools=TOOLS, tool_executor=dispatch_tool)
    tracker = escalation.EscalationTracker()

    turns = [
        "Hi! Can you tell me when order 112-3487561-2938471 will arrive?",
        "Great, thanks so much for checking!",
    ]
    for turn in turns:
        result = await agent.send(turn)
        reason = await escalation.check_escalation(tracker, agent.messages, result.tool_calls)
        assert reason is None
