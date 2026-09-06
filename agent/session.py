"""Session assembly and per-turn orchestration, shared across transports.

Through Phase 6 this all lived inline in transport/text_cli.py, since it
was the only transport. Phase 7 adds a second one (transport/voice_local.py)
that needs the *exact* identical behavior — which tools exist, how a turn
advances the confirmation gates, checks escalation, maybe hands off, and
detects the model ending the conversation. A transport depending on
another transport module would be backwards, and duplicating this logic
risks the two drifting apart the moment a future tool gets added. This is
the same "extract on the second real use case" call made for
agent/confirmation.py in Phase 6, and precisely what CLAUDE.md rule 5
watches for: wiring up a new I/O layer forcing a change to how business
logic is organized, rather than the I/O layer just plugging into it.

A transport's job is now just: gather input, call run_turn(), render the
result (print it, speak it, whatever), repeat. Nothing here knows or cares
whether that input came from a keyboard or a microphone.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent.confirmation import PendingActionGate
from agent.core import Agent
from agent.prompts import SYSTEM_PROMPT
from agent.tools import escalation, orders, policy_rag, refunds, scheduling, summary
from agent.tools.summary import SessionSummary
from guardrails.injection import sanitize_user_text
from guardrails.validators import check_reply_grounding, hedge_for

TOOLS = [
    orders.TOOL_SCHEMA,
    policy_rag.TOOL_SCHEMA,
    scheduling.FIND_SLOTS_SCHEMA,
    scheduling.BOOK_APPOINTMENT_SCHEMA,
    scheduling.CANCEL_APPOINTMENT_SCHEMA,
    refunds.TOOL_SCHEMA,
    summary.END_CONVERSATION_SCHEMA,
]

# No login/auth phase exists yet, so — same simplification as Phase 1's
# get_order_status — every transport just asks for a customer ID up front.
# It must be one that already exists in the mock DB (e.g.
# CUST-1001..CUST-1005), since Phase 2's ticket logging has a foreign-key
# constraint on it.
DEFAULT_CUSTOMER_ID = "CUST-1001"


@dataclass
class SessionGates:
    """One PendingActionGate per gated-action family a session needs.
    Scheduling (booking/cancelling) and refunds each get their own — a
    pending refund shouldn't be clobbered by an unrelated pending booking,
    or vice versa.
    """

    scheduling: PendingActionGate = field(default_factory=PendingActionGate)
    refunds: PendingActionGate = field(default_factory=PendingActionGate)

    def advance_turn(self) -> None:
        self.scheduling.turn += 1
        self.refunds.turn += 1


def build_dispatch_tool(
    customer_id: str, gates: SessionGates | None = None
) -> tuple[Callable[[str, dict], Any], dict[str, Callable[..., Any]], SessionGates]:
    """Assemble one session's tool dispatcher.

    Returns (dispatch_tool, handlers, gates). `handlers` is returned too
    (not just the closure) so tests can stub an individual tool via
    monkeypatch.setitem — mutating the dict in place is visible to
    dispatch_tool since the closure captures it by reference, not by value.
    """
    gates = gates or SessionGates()
    handlers: dict[str, Callable[..., Any]] = {
        "get_order_status": orders.get_order_status,
        "search_policy": policy_rag.search_policy,
        "find_available_slots": scheduling.find_available_slots,
        "book_appointment": lambda **kw: scheduling.book_appointment(
            **kw, state=gates.scheduling, customer_id=customer_id
        ),
        "cancel_appointment": lambda **kw: scheduling.cancel_appointment(
            **kw, state=gates.scheduling, customer_id=customer_id
        ),
        "issue_refund": lambda **kw: refunds.issue_refund(**kw, state=gates.refunds, customer_id=customer_id),
        "end_conversation": summary.end_conversation,
    }

    def dispatch_tool(tool_name: str, tool_input: dict) -> Any:
        """Route one tool_use call to the function that implements it."""
        handler = handlers.get(tool_name)
        if handler is None:
            raise ValueError(f"unknown tool: {tool_name}")
        return handler(**tool_input)

    return dispatch_tool, handlers, gates


def should_end_session(tool_calls: list[dict]) -> bool:
    """True if this turn's tool calls included the model deciding to sign off."""
    return any(call["name"] == "end_conversation" for call in tool_calls)


@dataclass
class Session:
    """Everything one conversation needs, assembled once via create_session()."""

    customer_id: str
    agent: Agent
    tracker: escalation.EscalationTracker
    gates: SessionGates
    handlers: dict[str, Callable[..., Any]]


def create_session(customer_id: str, client: Any | None = None) -> Session:
    """`client` is only for tests — real callers never pass it, and Agent
    creates its own anthropic.AsyncAnthropic() by default, same as every
    other place in this project that accepts an injectable client.
    """
    dispatch_tool, handlers, gates = build_dispatch_tool(customer_id)
    agent = Agent(system=SYSTEM_PROMPT, tools=TOOLS, tool_executor=dispatch_tool, client=client)
    return Session(
        customer_id=customer_id,
        agent=agent,
        tracker=escalation.EscalationTracker(),
        gates=gates,
        handlers=handlers,
    )


@dataclass
class TurnOutcome:
    """What one call to run_turn() produced. A transport renders this —
    prints it, speaks it, whatever — rather than run_turn() doing that
    itself, which is what keeps this module transport-agnostic.
    """

    reply: str
    ended: bool = False
    end_reason: str | None = None  # "model_ended" | "escalated"
    notice: str | None = None  # pre-formatted transfer text, if escalated
    llm_latency_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)  # non-fatal issues to surface, not swallow


async def run_turn(session: Session, user_text: str) -> TurnOutcome:
    """Send one user turn through the agent and run the same per-turn
    orchestration every transport needs: sanitize the caller's text, advance
    the confirmation gates, time the LLM call, check the reply is grounded in
    what the tools actually returned, check escalation, maybe hand off, check
    whether the model ended the conversation.

    Phase 10a adds the guardrails (guardrails/), all of them at this single
    point so no transport and nothing in agent/core.py had to change.
    """
    session.gates.advance_turn()

    clean_text, warnings = sanitize_user_text(user_text)

    start = time.perf_counter()
    result = await session.agent.send(clean_text)
    llm_latency = time.perf_counter() - start

    # Grounding: a flagged reply is never spoken — the customer hears a hedge
    # instead, and a second consecutive flag hands off to a human
    # (EscalationTracker). The "retry" is simply the customer's next turn, so
    # this costs no extra LLM round-trip and no dead air on a live call.
    reply = result.reply
    try:
        findings = check_reply_grounding(reply, result.tool_calls)
    except Exception as exc:  # noqa: BLE001 — a guardrail must never break a turn
        findings = []
        warnings.append(f"Could not check reply grounding this turn: {exc}")
    if findings:
        warnings.extend(findings)
        # Rotate on how many consecutive ungrounded replies preceded this one.
        # The tracker's counter is still the PREVIOUS count here — it is
        # incremented inside check_escalation below — so a first flag gets
        # HEDGE_PHRASES[0] and a second consecutive flag gets a different
        # line, which is exactly the point of varying it.
        reply = hedge_for(session.tracker.consecutive_ungrounded_replies)

    # Check escalation before should_end_session — a trigger here always
    # outranks the model deciding on its own the chat is naturally over.
    try:
        reason = await escalation.check_escalation(
            session.tracker,
            session.agent.messages,
            result.tool_calls,
            ungrounded=bool(findings),
        )
    except Exception as exc:  # noqa: BLE001 — a classifier hiccup must not crash the turn
        reason = None
        warnings.append(f"Could not run triage classification this turn: {exc}")

    if reason:
        try:
            packet = await escalation.create_handoff_packet(session.customer_id, session.agent.messages, reason)
            notice = f"I'm connecting you with a human agent — {reason}. (handoff #{packet['escalation_id']})"
        except Exception as exc:  # noqa: BLE001 — exit path must never crash on this
            notice = None
            warnings.append(f"Escalation triggered ({reason}) but the handoff packet couldn't be logged: {exc}")
        return TurnOutcome(
            reply=reply,
            ended=True,
            end_reason="escalated",
            notice=notice,
            llm_latency_seconds=llm_latency,
            warnings=warnings,
        )

    if should_end_session(result.tool_calls):
        return TurnOutcome(
            reply=reply, ended=True, end_reason="model_ended", llm_latency_seconds=llm_latency, warnings=warnings
        )

    return TurnOutcome(reply=reply, llm_latency_seconds=llm_latency, warnings=warnings)


@dataclass
class SessionCloseResult:
    summary: SessionSummary | None = None
    ticket_id: int | None = None
    error: str | None = None


async def close_session(session: Session) -> SessionCloseResult:
    """Summarize and log the session as a ticket, once there was actually a
    conversation to summarize. Never raises — a failure here (e.g. no API
    credit) shouldn't blow up a transport's exit path; the transport
    decides how to surface `.error`.
    """
    if not session.agent.messages:
        return SessionCloseResult()
    try:
        session_summary, ticket_id = await summary.close_session(session.customer_id, session.agent.messages)
        return SessionCloseResult(summary=session_summary, ticket_id=ticket_id)
    except Exception as exc:  # noqa: BLE001 — exit path must never crash on this
        return SessionCloseResult(error=f"Could not log session summary: {exc}")
