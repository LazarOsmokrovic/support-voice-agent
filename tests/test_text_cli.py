"""Phase 1/2 checkpoints for the REPL: tools are actually wired into the
Phase 0 tool-use loop, not just callable on their own, and the loop knows
to stop when the model signals the conversation is over.

Uses a mocked Claude client (no network, no API key) that scripts a
two-turn exchange: first Claude asks for a tool, then it replies with text
once the tool result comes back — exercising Agent.send end to end with the
real dispatch_tool from transport/text_cli.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import data.mock_db as mock_db
from agent.core import Agent
from transport.text_cli import TOOLS, dispatch_tool, should_end_session


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
    assert result.tool_calls == [{"name": "get_order_status", "input": {"order_id": order_id}}]
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
