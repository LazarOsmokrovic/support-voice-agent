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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import anthropic
from pydantic import BaseModel

from agent.core import DEFAULT_MODEL
from agent.prompts import CLASSIFICATION_PROMPT, HANDOFF_PROMPT
from agent.tools.notifications import notify_escalation
from agent.tools.scheduling import find_available_slots
from agent.tools.summary import format_transcript
from data.mock_db import get_connection
from guardrails.pii import HANDOFF_TEXT_FIELDS, redact_fields

logger = logging.getLogger("agent.tools.escalation")

# Named rather than repeated as literals: these values travel across
# escalation.py, handoff.py, session.py, turn_log.py and a DB column, and a
# typo in any one fails silently as "this handover is somehow neither open
# nor resolved".
STATUS_NONE = "none"
STATUS_OPEN = "open"
STATUS_RESOLVED = "resolved"

RESOLUTION_CALLBACK = "callback"
RESOLUTION_SELF = "customer_will_reach_out"
RESOLUTION_TRANSFER = "transfer"
RESOLUTION_UNRESOLVED = "unresolved"

# Whose decision each trigger represents. A mandatory trigger is the
# customer's or a rule's — the agent has no standing to second-guess it. A
# suggested trigger is the agent's own inference that it is failing, which is
# the judgement that hung up on a customer who had mis-dictated one digit,
# and on another who was calmly cancelling an order. Inferences get offered;
# they do not get imposed. CLAUDE.md rule 6's principle, applied to handoffs.
MANDATORY_REASONS = frozenset(
    {"explicit request for a human", "policy-restricted topic"}
)
SUGGESTED_REASONS = frozenset(
    {
        "repeated failed lookups",
        "sustained negative sentiment across multiple turns",
        "repeated ungrounded replies",
    }
)

MAX_END_REFUSALS = 2


@dataclass(frozen=True)
class EscalationSignal:
    reason: str
    mandatory: bool


@dataclass
class EscalationState:
    """One handover per session, for the life of the session.

    That is what happens on a real support line: a human ringing a customer
    back deals with everything that customer has, rather than booking three
    calls for three questions. So a second escalation-worthy issue becomes
    another ITEM on the same handover — which is why `items` is a list and not
    the single `reason` string it replaced.

    Mutated in place, never rebound: the tool handlers in build_dispatch_tool
    close over it before the Session that owns it exists.
    """

    status: str = STATUS_NONE
    escalation_id: int | None = None
    packet: dict[str, Any] | None = None
    items: list[str] = field(default_factory=list)
    resolution: str | None = None
    callback_time: str | None = None
    # Suggested triggers that have already made their offer. Being asked over
    # and over whether you want a human is its own kind of failure.
    offered: set[str] = field(default_factory=set)
    # Anything still unsent. close_session flushes on this, NOT on is_open —
    # a resolution recorded and then lost to a crashing turn is not open, and
    # guarding on is_open would drop it silently.
    pending_persist: bool = False
    refusals: int = 0
    _last_refusal_turn: int | None = None

    @property
    def is_open(self) -> bool:
        return self.status == STATUS_OPEN

    def open(self, reason: str) -> None:
        if self.status == STATUS_NONE:
            self.status = STATUS_OPEN
        if reason not in self.items:
            self.items.append(reason)
        self.pending_persist = True

    def amend(self, reason: str) -> None:
        """A second trigger during an existing handover. Never reopens a
        resolved one and never books a second callback — it adds an item the
        colleague taking it over can prepare for.
        """
        if reason in self.items:
            return
        self.items.append(reason)
        self.pending_persist = True

    def record_resolution(self, resolution: str, callback_time: str | None = None) -> None:
        self.status = STATUS_RESOLVED
        self.resolution = resolution
        self.callback_time = callback_time
        self.pending_persist = True

    def consume_refusal(self, turn: int) -> bool:
        """True if end_conversation should be refused. Budget is spent at most
        once per TURN.

        agent/core.py:91 allows 8 tool iterations, so a model that reads the
        refusal and retries can call end_conversation three times inside one
        agent.send(). Counting calls would exhaust the budget with no customer
        utterance in between — the nudge becomes a rubber stamp on exactly the
        turn it was meant to catch.
        """
        if self.refusals >= MAX_END_REFUSALS and turn != self._last_refusal_turn:
            return False
        if turn != self._last_refusal_turn:
            self.refusals += 1
            self._last_refusal_turn = turn
        return True

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
    ) -> EscalationSignal | None:
        """Update counters from this turn; return an escalation signal the
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
            return EscalationSignal(tool_escalation, mandatory=True)
        if classification.intent == "request_human":
            return EscalationSignal("explicit request for a human", mandatory=True)
        if classification.policy_restricted:
            return EscalationSignal("policy-restricted topic", mandatory=True)

        if classification.sentiment == "negative":
            self.consecutive_negative_turns += 1
        else:
            self.consecutive_negative_turns = 0
        if self.consecutive_negative_turns >= NEGATIVE_SENTIMENT_ESCALATION_THRESHOLD:
            return EscalationSignal("sustained negative sentiment across multiple turns", mandatory=False)

        outcomes = _turn_tool_outcomes(tool_calls)
        if outcomes:
            if any(outcomes):  # at least one lookup succeeded this turn — clean slate
                self.consecutive_failed_lookups = 0
            else:
                self.consecutive_failed_lookups += 1
            # INSIDE this block on purpose. It used to sit outside, re-reading
            # the counter every turn — so once the streak tripped, a turn with
            # no lookup at all still escalated. One live call produced rows 7,
            # 8 and 9 for one problem that way, the last from a turn that
            # merely asked the agent to repeat a number back.
            if self.consecutive_failed_lookups >= FAILED_LOOKUP_ESCALATION_THRESHOLD:
                return EscalationSignal("repeated failed lookups", mandatory=False)

        if ungrounded:
            self.consecutive_ungrounded_replies += 1
        else:
            self.consecutive_ungrounded_replies = 0
        if self.consecutive_ungrounded_replies >= UNGROUNDED_REPLY_ESCALATION_THRESHOLD:
            return EscalationSignal("repeated ungrounded replies", mandatory=False)

        return None

    def reset_streak(self, reason: str) -> None:
        """Reset the streak counter for the given reason."""
        if reason == "repeated failed lookups":
            self.consecutive_failed_lookups = 0
        elif reason == "sustained negative sentiment across multiple turns":
            self.consecutive_negative_turns = 0
        elif reason == "repeated ungrounded replies":
            self.consecutive_ungrounded_replies = 0


async def check_escalation(
    tracker: EscalationTracker,
    messages: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    ungrounded: bool = False,
    client: anthropic.AsyncAnthropic | None = None,
) -> EscalationSignal | None:
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


def _next_callback_slot() -> str | None:
    """The earliest open slot a human could call back on, or None.

    Read-only: it asks the real calendar so the time offered is genuinely
    free, but reserves nothing. Returns None rather than raising if the
    calendar is unreachable or fully booked — an escalation must never fail
    because scheduling did, so the caller falls back to wording that promises
    a callback without naming a time.
    """
    try:
        slots = find_available_slots()
    except Exception:  # noqa: BLE001 — escalation must not depend on the calendar
        logger.exception("find_available_slots raised while picking a callback time")
        return None
    available = slots.get("slots") or []
    return available[0] if available else None


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

    # The earliest slot a human could ring back on, so the customer hears a
    # specific time and the human agent's notification names the SAME one.
    # Without it the customer is told to hold for a transfer that (outside
    # Phase 10d's Twilio path) is not going to happen.
    #
    # Deliberately NOT booked. Reserving a slot is irreversible and CLAUDE.md
    # rule 6 requires an explicit confirmation turn — and escalation ends the
    # call, so there is no turn left to confirm in. The human agent confirms
    # the time when they actually ring. The cost is that the slot is not held,
    # so two escalations in quick succession would offer the same one; the
    # alternative is pausing an escalation to negotiate a booking with a
    # customer who has just asked to stop talking to a bot.
    packet["callback_time"] = _next_callback_slot()

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
