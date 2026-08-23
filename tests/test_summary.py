"""Phase 2 checkpoint: structured session summaries always validate against
the schema, and get written to the tickets table correctly.

Offline tests (no network) cover the plumbing; the live checkpoint test
itself calls summarize_session 20+ times against a real transcript and
confirms every result validates — skipped automatically without a real
ANTHROPIC_API_KEY, same pattern as the live tests in Phase 0/1.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

import data.mock_db as mock_db
from agent.tools.summary import (
    SessionSummary,
    _format_transcript,
    close_session,
    log_ticket,
    summarize_session,
)


def test_format_transcript_renders_text_and_collapses_tool_blocks():
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Your order has shipped."

    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "get_order_status"

    messages = [
        {"role": "user", "content": "Where's my order?"},
        {"role": "assistant", "content": [tool_use_block]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "..."}]},
        {"role": "assistant", "content": [text_block]},
    ]

    transcript = _format_transcript(messages)

    assert "user: Where's my order?" in transcript
    assert "assistant: [used tool: get_order_status]" in transcript
    assert "user: [tool result received]" in transcript
    assert "assistant: Your order has shipped." in transcript


def test_log_ticket_writes_row_to_tickets_table(tmp_path, monkeypatch):
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_summary.db")
    mock_db.reset_and_seed()
    customer_id = mock_db.CUSTOMERS[0][0]
    summary = SessionSummary(
        issue="Order never arrived.",
        resolution="Replacement shipped at no charge.",
        sentiment="neutral",
        follow_up_needed=False,
    )

    ticket_id = log_ticket(customer_id, summary, created_at="2026-08-23T00:00:00+00:00")

    with mock_db.get_connection() as conn:
        row = conn.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()

    assert row["customer_id"] == customer_id
    assert row["issue"] == summary.issue
    assert row["resolution"] == summary.resolution
    assert row["sentiment"] == "neutral"
    assert row["follow_up_needed"] == 0


@pytest.mark.asyncio
async def test_summarize_session_returns_parsed_summary_from_mocked_client():
    fake_summary = SessionSummary(
        issue="Wanted a refund.",
        resolution="Refund approved.",
        sentiment="positive",
        follow_up_needed=False,
    )
    fake_response = MagicMock()
    fake_response.parsed_output = fake_summary
    fake_client = MagicMock()
    fake_client.messages.parse = AsyncMock(return_value=fake_response)

    result = await summarize_session([{"role": "user", "content": "I want a refund."}], client=fake_client)

    assert result is fake_summary
    fake_client.messages.parse.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_session_summarizes_and_logs_a_ticket(tmp_path, monkeypatch):
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_close_session.db")
    mock_db.reset_and_seed()
    customer_id = mock_db.CUSTOMERS[0][0]

    fake_summary = SessionSummary(
        issue="Asked about shipping.",
        resolution="Provided tracking info.",
        sentiment="neutral",
        follow_up_needed=False,
    )
    fake_response = MagicMock()
    fake_response.parsed_output = fake_summary
    fake_client = MagicMock()
    fake_client.messages.parse = AsyncMock(return_value=fake_response)

    result_summary, ticket_id = await close_session(
        customer_id, [{"role": "user", "content": "Where's my package?"}], client=fake_client
    )

    assert result_summary is fake_summary
    with mock_db.get_connection() as conn:
        row = conn.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
    assert row["customer_id"] == customer_id


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="requires a real ANTHROPIC_API_KEY to hit the live Claude API",
)
@pytest.mark.asyncio
async def test_summarize_session_always_validates_against_schema():
    """Phase 2 checkpoint: run the real structured-output call 20+ times and
    confirm every result validates against SessionSummary. LLM structured
    output can occasionally drift, so this is an empirical check across many
    calls, not just trusting the API's schema guarantee on a single one.
    """
    transcript = [
        {"role": "user", "content": "My order 112-3487561-2938471 never arrived, it's been 2 weeks."},
        {
            "role": "assistant",
            "content": (
                "I'm sorry about the delay! I checked your order and it shows as delayed "
                "in transit. I've gone ahead and issued a replacement shipment at no extra "
                "cost, which should arrive within 3-5 business days. Anything else I can help with?"
            ),
        },
        {"role": "user", "content": "That's great, thank you!"},
    ]

    for _ in range(20):
        result = await summarize_session(transcript)
        assert isinstance(result, SessionSummary)
        assert result.sentiment in {"positive", "neutral", "negative"}
        assert isinstance(result.follow_up_needed, bool)
        assert result.issue.strip()
        assert result.resolution.strip()
