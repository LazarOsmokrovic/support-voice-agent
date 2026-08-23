"""Post-session summary & CRM logging.

This file has two distinct halves:

1. END_CONVERSATION_SCHEMA / end_conversation() — a normal tool the model
   calls mid-conversation, exactly like get_order_status, when it judges the
   customer's issue is resolved and they're signing off. Recognizing "this
   conversation is naturally over" from a goodbye/thanks is genuinely
   ambiguous (not a fixed keyword check), so it's handed to model judgment
   via a tool rather than pattern-matched in the transport layer.

2. Once a conversation ends (by that tool or by the user typing quit/exit),
   call Claude exactly once with a *separate* structured-output request to
   produce a summary (issue, resolution, sentiment, follow-up needed), then
   write it to the `tickets` table. This half is not part of the tool-use
   loop at all — the application calls it after the session is over.

The summary's schema is expressed as a Pydantic model rather than a
hand-written JSON schema dict: passed to client.messages.parse(), the SDK
builds the request and validates Claude's response against it for us,
handing back an already-validated SessionSummary instead of raw text to
parse ourselves.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

import anthropic
from pydantic import BaseModel

from agent.core import DEFAULT_MODEL
from agent.prompts import SUMMARY_PROMPT
from data.mock_db import get_connection

END_CONVERSATION_SCHEMA: dict[str, Any] = {
    "name": "end_conversation",
    "description": (
        "Call this when the customer's issue has been fully resolved and "
        "they've indicated they're done — thanking you, saying goodbye, or "
        "confirming everything is fine. This closes out the support "
        "session. Do not call it just because the customer said thanks for "
        "one thing while another part of their issue is still open — only "
        "call it when there is genuinely nothing left to help with."
    ),
    "input_schema": {"type": "object", "properties": {}},
}


def end_conversation() -> str:
    """No real work to do — the point is Claude choosing to call this tool at
    all. The transport layer watches for it in TurnResult.tool_calls and
    ends the session; the return value just satisfies the tool_result the
    API requires for every tool_use.
    """
    return "Session marked complete."


class SessionSummary(BaseModel):
    issue: str
    resolution: str
    sentiment: Literal["positive", "neutral", "negative"]
    follow_up_needed: bool


def format_transcript(messages: list[dict[str, Any]]) -> str:
    """Render an Agent's message history as plain text for a summary/
    classification prompt. Public and shared — agent/tools/escalation.py
    (Phase 4) reuses this too, rather than duplicating it.

    `Agent.messages` mixes plain dicts (user turns, tool_result turns, built
    by our own code) with SDK content-block objects (assistant turns, set
    directly from `response.content`) — so each block is read via getattr
    first, falling back to dict access. Tool calls/results are collapsed to
    a short marker rather than dumped as raw JSON, since this transcript is
    meant to read like a plain-English conversation.
    """

    def block_type(block: Any) -> str | None:
        return getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")

    def block_field(block: Any, field: str) -> Any:
        return getattr(block, field, None) if not isinstance(block, dict) else block.get(field)

    lines: list[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"]

        if isinstance(content, str):
            lines.append(f"{role}: {content}")
            continue

        for block in content:
            kind = block_type(block)
            if kind == "text":
                lines.append(f"{role}: {block_field(block, 'text')}")
            elif kind == "tool_use":
                lines.append(f"{role}: [used tool: {block_field(block, 'name')}]")
            elif kind == "tool_result":
                lines.append(f"{role}: [tool result received]")

    return "\n".join(lines)


async def summarize_session(
    messages: list[dict[str, Any]],
    client: anthropic.AsyncAnthropic | None = None,
    model: str = DEFAULT_MODEL,
) -> SessionSummary:
    """Call Claude once to produce a structured summary of a finished conversation."""
    client = client or anthropic.AsyncAnthropic()
    transcript = format_transcript(messages)

    response = await client.messages.parse(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": SUMMARY_PROMPT.format(transcript=transcript)}],
        output_format=SessionSummary,
    )
    return response.parsed_output


def log_ticket(customer_id: str, summary: SessionSummary, created_at: str | None = None) -> int:
    """Write a session summary to the tickets table. Returns the new ticket_id.

    `customer_id` must already exist in the customers table — tickets has a
    foreign-key constraint on it (enforced via PRAGMA foreign_keys = ON in
    data/mock_db.py's get_connection()).
    """
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO tickets "
            "(customer_id, issue, resolution, sentiment, follow_up_needed, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                customer_id,
                summary.issue,
                summary.resolution,
                summary.sentiment,
                int(summary.follow_up_needed),
                created_at,
            ),
        )
        return cursor.lastrowid


async def close_session(
    customer_id: str,
    messages: list[dict[str, Any]],
    client: anthropic.AsyncAnthropic | None = None,
) -> tuple[SessionSummary, int]:
    """Summarize a finished conversation and log it as a ticket in one call.

    This is what transport/text_cli.py (and later transports) call when a
    session ends. Returns (summary, ticket_id).
    """
    summary = await summarize_session(messages, client=client)
    ticket_id = log_ticket(customer_id, summary)
    return summary, ticket_id
