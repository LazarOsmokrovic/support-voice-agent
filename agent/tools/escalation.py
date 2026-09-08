"""Per-turn triage classification, escalation tracking, and human handoff.

Three pieces:

1. TurnClassification / classify_turn — a lightweight, one-shot
   structured-output call after every turn, judging intent, sentiment, and
   whether the topic needs human review regardless of tone. Recognizing
   these from natural language is genuinely ambiguous, so it's handed to
   model judgment (CLAUDE.md rule 7) rather than pattern-matched.

2. EscalationTracker — the deterministic half. "Repeated failed attempts"
   and "sustained negative sentiment" are objective, countable things, so
   they're tracked as plain counters in code rather than re-judged by the
   model every turn. record_turn() returns an escalation reason the moment
   a trigger actually fires, or None otherwise. A tool can also signal
   escalation directly (Phase 6's issue_refund does, for high-value
   refunds) via a generic escalate/escalation_reason convention in its
   output — this file doesn't need to know anything refund-specific to
   honor it.

3. HandoffFields / create_handoff_packet — once EscalationTracker decides
   to escalate, this assembles the actual packet (customer intent, summary,
   verified account info, actions taken, reason, sentiment) via one more
   structured-output call, and persists it to the escalations table.
   Deliberately NOT a tool the model calls itself, unlike
   get_order_status / search_policy / end_conversation: the decision to
   escalate has already been made by the time this runs, so there's
   nothing left for the model to decide by calling it. It's triggered by
   the application, the same way Phase 2's close_session is — "for now,
   transfer to human just logs the packet" (PROJECT_PLAN.md); a real
   transfer arrives in Phase 10.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import anthropic
from pydantic import BaseModel

from agent.core import DEFAULT_MODEL
from agent.prompts import CLASSIFICATION_PROMPT, HANDOFF_PROMPT
from agent.tools.notifications import notify_escalation
from agent.tools.summary import format_transcript
from data.mock_db import get_connection
from guardrails.pii import HANDOFF_TEXT_FIELDS, redact_fields

logger = logging.getLogger("agent.tools.escalation")

# How many consecutive turns of the same bad signal before actually
# escalating — chosen to avoid firing on one grumpy word or one bad lookup,
# while still catching genuine sustained trouble. Checked against the
# scripted conversations in tests/test_escalation.py and
# tests/test_text_cli.py, not just picked arbitrarily.
NEGATIVE_SENTIMENT_ESCALATION_THRESHOLD = 2
FAILED_LOOKUP_ESCALATION_THRESHOLD = 2

# Phase 10a: how many consecutive replies the grounding detector
# (guardrails/validators.py) may flag before handing off. Matched to the two
# thresholds above for consistency, but deliberately a starting value — a
# hallucination is weaker evidence of trouble than two consecutively angry
# messages, so 3 is arguable. Sub-phase 10c instruments the threshold and
# records a baseline — roughly a dozen labellable turns, which cannot settle
# the value on statistical grounds but does make a later change measurable
# as a delta rather than argued as a hunch.
UNGROUNDED_REPLY_ESCALATION_THRESHOLD = 2


class TurnClassification(BaseModel):
    intent: Literal[
        "order_status",
        "policy_question",
        "refund_or_return",
        "complaint",
        "request_human",
        "chitchat",
        "other",
    ]
    sentiment: Literal["positive", "neutral", "negative"]
    policy_restricted: bool


async def classify_turn(
    messages: list[dict[str, Any]],
    client: anthropic.AsyncAnthropic | None = None,
    model: str = DEFAULT_MODEL,
) -> TurnClassification:
    """One lightweight structured-output call judging the latest turn."""
    client = client or anthropic.AsyncAnthropic()
    transcript = format_transcript(messages)

    response = await client.messages.parse(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": CLASSIFICATION_PROMPT.format(transcript=transcript)}],
        output_format=TurnClassification,
    )
    return response.parsed_output


def _turn_tool_outcomes(tool_calls: list[dict[str, Any]]) -> list[bool]:
    """Extract "found" booleans from this turn's tool calls, for whichever
    ones recorded an output shaped like {"found": bool, ...}
    (get_order_status, search_policy). Tools without that shape (e.g.
    end_conversation) are skipped — they're not lookups that can "fail"
    in this sense.
    """
    outcomes = []
    for call in tool_calls:
        output = call.get("output")
        if isinstance(output, dict) and isinstance(output.get("found"), bool):
            outcomes.append(output["found"])
    return outcomes


def _tool_signaled_escalation(tool_calls: list[dict[str, Any]]) -> str | None:
    """Phase 6: any tool can ask for escalation directly by returning a
    truthy "escalate" in its output (issue_refund does this for high-value
    refunds) — this is a deliberately generic convention, not specific to
    refunds, so future tools can reuse it the same way without this file
    needing to know about them.
    """
    for call in tool_calls:
        output = call.get("output")
        if isinstance(output, dict) and output.get("escalate"):
            return output.get("escalation_reason", "a high-value action requires human approval")
    return None


@dataclass
class EscalationTracker:
    """One instance per session. Call record_turn() after every turn."""

    consecutive_negative_turns: int = 0
    consecutive_failed_lookups: int = 0
    consecutive_ungrounded_replies: int = 0

    def record_turn(
        self,
        classification: TurnClassification,
        tool_calls: list[dict[str, Any]],
        ungrounded: bool = False,
    ) -> str | None:
        """Update counters from this turn; return an escalation reason the
        moment a trigger fires, else None.

        Immediate triggers (a tool directly signaling escalation, an
        explicit human request, or a policy-restricted topic) short-circuit
        without touching the streak counters — nothing else matters once
        one fires.

        ungrounded — whether guardrails/validators.py flagged this turn's
        reply as unsupported by tool output; two consecutive flags hand off
        to a human.
        """
        tool_escalation = _tool_signaled_escalation(tool_calls)
        if tool_escalation:
            return tool_escalation
        if classification.intent == "request_human":
            return "explicit request for a human"
        if classification.policy_restricted:
            return "policy-restricted topic"

        if classification.sentiment == "negative":
            self.consecutive_negative_turns += 1
        else:
            self.consecutive_negative_turns = 0
        if self.consecutive_negative_turns >= NEGATIVE_SENTIMENT_ESCALATION_THRESHOLD:
            return "sustained negative sentiment across multiple turns"

        outcomes = _turn_tool_outcomes(tool_calls)
        if outcomes:
            if any(outcomes):  # at least one lookup succeeded this turn — clean slate
                self.consecutive_failed_lookups = 0
            else:
                self.consecutive_failed_lookups += 1
        if self.consecutive_failed_lookups >= FAILED_LOOKUP_ESCALATION_THRESHOLD:
            return "repeated failed lookups"

        if ungrounded:
            self.consecutive_ungrounded_replies += 1
        else:
            self.consecutive_ungrounded_replies = 0
        if self.consecutive_ungrounded_replies >= UNGROUNDED_REPLY_ESCALATION_THRESHOLD:
            return "repeated ungrounded replies"

        return None


async def check_escalation(
    tracker: EscalationTracker,
    messages: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    ungrounded: bool = False,
    client: anthropic.AsyncAnthropic | None = None,
) -> str | None:
    """classify_turn + tracker.record_turn in one call — shared by
    transport/text_cli.py's real loop and by tests, so the two can't drift
    apart from each other.
    """
    classification = await classify_turn(messages, client=client)
    return tracker.record_turn(classification, tool_calls, ungrounded=ungrounded)


class HandoffFields(BaseModel):
    customer_intent: str
    conversation_summary: str
    verified_account_info: str
    actions_taken: str
    sentiment: Literal["positive", "neutral", "negative"]


async def _infer_handoff_fields(
    customer_id: str,
    messages: list[dict[str, Any]],
    client: anthropic.AsyncAnthropic | None = None,
    model: str = DEFAULT_MODEL,
) -> HandoffFields:
    client = client or anthropic.AsyncAnthropic()
    transcript = format_transcript(messages)

    response = await client.messages.parse(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": HANDOFF_PROMPT.format(customer_id=customer_id, transcript=transcript)}
        ],
        output_format=HandoffFields,
    )
    return response.parsed_output


def log_escalation(
    customer_id: str, reason: str, fields: HandoffFields, created_at: str | None = None
) -> int:
    """Persist a handoff packet to the escalations table. Returns the new escalation_id."""
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO escalations "
            "(customer_id, reason, customer_intent, conversation_summary, "
            "verified_account_info, actions_taken, sentiment, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                customer_id,
                reason,
                fields.customer_intent,
                fields.conversation_summary,
                fields.verified_account_info,
                fields.actions_taken,
                fields.sentiment,
                created_at,
            ),
        )
        return cursor.lastrowid


def mark_notified(escalation_id: int, delivered: bool, notified_at: str | None = None) -> None:
    """Record whether notify_escalation actually delivered this handoff to
    the automation platform. Always called after log_escalation, whether or
    not delivery succeeded — the escalations table is the durable record of
    both what happened and whether a human was actually told.
    """
    notified_at = notified_at or datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            "UPDATE escalations SET notified = ?, notified_at = ? WHERE escalation_id = ?",
            (int(delivered), notified_at, escalation_id),
        )


async def create_handoff_packet(
    customer_id: str,
    messages: list[dict[str, Any]],
    reason: str,
    client: anthropic.AsyncAnthropic | None = None,
) -> dict[str, Any]:
    """Assemble a structured handoff packet, persist it, and notify an
    external automation platform (Phase 11) — see agent/tools/notifications.py.

    Not a tool the model calls itself — see the module docstring. Returns
    the full packet, including its escalation_id, for the transport layer
    to relay (e.g. print a transfer notice). Notification delivery never
    affects this return value — persisting the packet must not depend on
    whether anyone was actually told about it.
    """
    inferred = await _infer_handoff_fields(customer_id, messages, client=client)
    # Redact ONCE, here, so the DB row and the outbound webhook carry
    # identical text. notify_escalation redacts again defensively for any
    # future caller; redaction is idempotent, so that second pass is a no-op.
    fields = HandoffFields(**redact_fields(inferred.model_dump(), HANDOFF_TEXT_FIELDS))
    escalation_id = log_escalation(customer_id, reason, fields)
    packet = {"escalation_id": escalation_id, "reason": reason, **fields.model_dump()}

    try:
        delivered = await notify_escalation(packet)
    except Exception:  # noqa: BLE001 — a broken webhook must never break escalation
        logger.exception("notify_escalation raised unexpectedly")
        delivered = False

    # Separate try/except from the notify call above: mark_notified is a
    # second, independent thing that can fail (e.g. a pre-existing DB
    # missing the notified/notified_at columns, or write-lock contention
    # under simultaneous escalations) and it must not discard a packet that
    # log_escalation already durably persisted. Kept as its own except block
    # so the two distinct failure modes stay distinguishable in logs.
    try:
        mark_notified(escalation_id, delivered)
    except Exception:  # noqa: BLE001 — recording delivery status must never break escalation
        logger.exception("mark_notified raised unexpectedly")

    return packet
