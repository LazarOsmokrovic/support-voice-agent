"""Phase 1 checkpoint: get_order_status is actually wired into the Phase 0
tool-use loop, not just callable on its own.

Uses a mocked Claude client (no network, no API key) that scripts a
two-turn exchange: first Claude asks for the tool, then it replies with
text once the tool result comes back — exercising Agent.send end to end
with the real dispatch_tool from transport/text_cli.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import data.mock_db as mock_db
from agent.core import Agent
from transport.text_cli import TOOLS, dispatch_tool


def _tool_use_response(order_id: str):
    block = MagicMock()
    block.type = "tool_use"
    block.id = "toolu_test123"
    block.name = "get_order_status"
    block.input = {"order_id": order_id}
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
            _tool_use_response(order_id),
            _text_response("Your order has shipped!"),
        ]
    )

    agent = Agent(client=fake_client, tools=TOOLS, tool_executor=dispatch_tool)
    result = await agent.send(f"Where's my order {order_id}?")

    assert result.reply == "Your order has shipped!"
    assert result.tool_calls == [{"name": "get_order_status", "input": {"order_id": order_id}}]
    assert fake_client.messages.create.await_count == 2
