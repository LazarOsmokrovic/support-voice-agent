"""Phase 1/2/3/4/5/6 checkpoints for the tool-use loop: tools are actually
wired into it, not just callable on their own, and the loop knows to stop
when the model signals the conversation is over. Since Phase 7, the tool
registry/dispatcher these tests exercise (TOOLS, build_dispatch_tool,
should_end_session) lives in agent/session.py, shared with
transport/voice_local.py — see that module's docstring for why.

Uses a mocked Claude client (no network, no API key) that scripts a
two-turn exchange: first Claude asks for a tool, then it replies with text
once the tool result comes back — exercising Agent.send end to end with a
real dispatch_tool.

The live full-pipeline checkpoints that used to live here — Phase 3's
uncovered policy question, Phase 4's three escalation conversations, Phase
5's book-then-reschedule, and Phase 6's two refund conversations — moved to
eval/scenarios.py in Phase 10c. They are now recorded once and replayed
offline, which retires the calendar expiry that silently broke the
high-value refund test for a week: the seeded delivery dates aged past the
30-day return window, so issue_refund returned outside_window BEFORE
reaching the escalation branch the test asserted on, and a calendar expiry
looked exactly like a logic regression. The eval harness freezes the clock
to each recording's timestamp instead. Everything remaining in this file is
offline and mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core import Agent
from agent.session import TOOLS, build_dispatch_tool, should_end_session
from data import mock_db


def _fresh_dispatch_tool(customer_id: str = "CUST-1001"):
    """Most tests here don't care about scheduling state or which customer
    is acting — this just saves repeating build_dispatch_tool's 3-tuple
    unpacking everywhere. Tests that DO care call build_dispatch_tool
    directly.
    """
    dispatch_tool, _handlers, _state = build_dispatch_tool(customer_id)
    return dispatch_tool


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

    agent = Agent(client=fake_client, tools=TOOLS, tool_executor=_fresh_dispatch_tool())
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

    agent = Agent(client=fake_client, tools=TOOLS, tool_executor=_fresh_dispatch_tool())
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
    # reach dispatch_tool), not retrieval quality, which
    # tests/test_policy_rag.py already covers on its own.
    dispatch_tool, handlers, _state = build_dispatch_tool("CUST-1001")
    monkeypatch.setitem(
        handlers,
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


