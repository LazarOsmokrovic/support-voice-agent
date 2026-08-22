"""Phase 0 checkpoint: agent/core.py's tool-use loop returns a reply.

Two tests:
  - the trivial, always-run one: a mocked client, no network, no API key
    needed — proves the loop's control flow (call -> no tool_use -> return
    text) is correct.
  - the live one: a real "hello" round trip to Claude, skipped automatically
    when ANTHROPIC_API_KEY isn't set (e.g. in CI) since it costs a real call.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core import Agent


def _text_response(text: str, stop_reason: str = "end_turn"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.stop_reason = stop_reason
    return response


@pytest.mark.asyncio
async def test_send_returns_text_when_claude_replies_without_tool_use():
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=_text_response("Hello! How can I help?"))

    agent = Agent(client=fake_client)
    result = await agent.send("hello")

    assert result.reply == "Hello! How can I help?"
    assert result.tool_calls == []
    fake_client.messages.create.assert_awaited_once()


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="requires a real ANTHROPIC_API_KEY to hit the live Claude API",
)
@pytest.mark.asyncio
async def test_hello_live_smoke():
    """Phase 0 checkpoint, live half: send 'hello', get an actual reply back."""
    agent = Agent()
    result = await agent.send("hello")

    assert isinstance(result.reply, str)
    assert result.reply.strip()
