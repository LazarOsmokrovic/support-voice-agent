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
import uuid
from collections.abc import Callable
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any

from agent.confirmation import PendingActionGate
from agent.core import Agent
from agent.prompts import SYSTEM_PROMPT
from agent.tools import escalation, orders, policy_rag, refunds, scheduling, summary
from agent.tools.summary import SessionSummary
from guardrails.injection import sanitize_user_text
from guardrails.validators import check_reply_grounding, hedge_for
from observability.turn_log import TurnRecord, log_turn

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


def _turn_proposed_a_confirmation(tool_calls: list[dict]) -> bool:
    """True if any tool call this turn was the *proposal* half of
    propose-then-confirm (agent/confirmation.py's PendingActionGate,
    used by issue_refund, book_appointment, cancel_appointment) — i.e. its
    output is a dict carrying status == "pending_confirmation".

    Used to keep a hedge substitution from stranding an armed confirmation
    gate: if the customer never hears the proposal ("This order is eligible
    for a $34.99 refund...") because a hedge was spoken instead, their next
    "okay" would commit the action on an uninformed confirmation.
    """
    return any(
        isinstance(call.get("output"), dict) and call["output"].get("status") == "pending_confirmation"
        for call in tool_calls
    )


def _substitute_hedge_in_history(messages: list[dict[str, Any]], hedge: str) -> str | None:
    """Overwrite the content of the last assistant message with `hedge`, so
    agent/core.py's conversation history matches what the customer actually
    heard, not the ungrounded reply that was suppressed. Without this, the
    model's real (flagged) reply lingers in `session.agent.messages`
    even though the hedge is what got spoken — so a later turn can restate
    the same claim, find it sitting right there in its own prior turn, and
    have the grounding check wrongly treat it as already-established.

    Safe specifically because this is only called after a turn that ended
    with stop_reason != "tool_use" (agent/core.py), so the final assistant
    message should hold nothing but text content blocks. Written
    defensively anyway: if that message isn't found, or its content isn't a
    list of text-only blocks, this leaves history untouched and returns a
    warning string instead of guessing. Never raises — a guardrail must
    never corrupt the very state it's trying to protect.
    """
    try:
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if not isinstance(content, list) or not content:
                return "Could not reconcile hedged reply in history: unexpected assistant message shape."
            block_types = {
                block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                for block in content
            }
            if block_types != {"text"}:
                return "Could not reconcile hedged reply in history: assistant message wasn't text-only."
            message["content"] = [{"type": "text", "text": hedge}]
            return None
        return "Could not reconcile hedged reply in history: no assistant message found."
    except Exception as exc:  # noqa: BLE001 — a guardrail must never corrupt history or crash the turn
        return f"Could not reconcile hedged reply in history: {exc}"


@dataclass
class Session:
    """Everything one conversation needs, assembled once via create_session()."""

    customer_id: str
    session_id: str
    transport: str
    agent: Agent
    tracker: escalation.EscalationTracker
    gates: SessionGates
    handlers: dict[str, Callable[..., Any]]
    # Telemetry's own turn index (observability/turn_log.py) — deliberately
    # separate from gates.refunds.turn / gates.scheduling.turn, which belong
    # to PendingActionGate (the confirmation mechanism) and would silently
    # change telemetry's meaning if that subsystem ever changes. Incremented
    # in run_turn, alongside gates.advance_turn() but not by it; the DTMF
    # handler (transport/pipecat_processors.py) advances it too, since that
    # path bypasses run_turn entirely and would otherwise collide with the
    # preceding spoken turn's number.
    turn: int = 0


def create_session(customer_id: str, client: Any | None = None, transport: str = "unknown") -> Session:
    """`client` is only for tests — real callers never pass it, and Agent
    creates its own anthropic.AsyncAnthropic() by default, same as every
    other place in this project that accepts an injectable client.

    `transport` labels which I/O layer is driving this conversation, purely
    so per-turn records (observability/turn_log.py) say which channel a turn
    came from. run_turn cannot infer it, and once telephony and CLI turns
    share one log file it is the difference between a readable record and an
    ambiguous one. It is a label, not behavior — nothing branches on it.
    """
    dispatch_tool, handlers, gates = build_dispatch_tool(customer_id)
    agent = Agent(system=SYSTEM_PROMPT, tools=TOOLS, tool_executor=dispatch_tool, client=client)
    return Session(
        customer_id=customer_id,
        session_id=uuid.uuid4().hex,
        transport=transport,
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
    # Phase 10d: the transport needs the packet itself, not just the notice —
    # transport/telephony.py whispers it to the human agent before bridging
    # the call. Same purpose as `notice` above: data for a transport to
    # render. run_turn built this and discarded it before 10d.
    escalation_packet: dict[str, Any] | None = None
    llm_latency_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)  # non-fatal issues to surface, not swallow


def _escalation_notice(callback_time: str | None) -> str:
    """What the customer actually hears when the agent hands them off.

    This is spoken aloud, which the original wording forgot: it read the
    INTERNAL escalation reason out loud ("sustained negative sentiment across
    multiple turns") and then recited a handoff number. That told an already
    frustrated customer they had been classified as angry, and gave them a
    ticket ID they cannot use. Both belong in the turn log and the escalations
    row, where they already are.

    It promises a callback rather than a transfer because, outside Phase 10d's
    Twilio path, no transfer happens — the call simply ends. Saying "connecting
    you now" and then hanging up is worse than saying nothing.
    """
    if callback_time:
        when = _spoken_time(callback_time)
        return (
            "Let me get one of my colleagues to call you back about this. "
            f"The earliest we have is {when}, and they'll have the full details of our conversation."
        )
    return (
        "Let me get one of my colleagues to call you back about this. "
        "They'll be in touch shortly, and they'll have the full details of our conversation."
    )


def _spoken_time(slot: str) -> str:
    """Turn an ISO slot into something a person would say.

    "2026-09-11T09:00:00" read aloud by a speech synthesiser is unintelligible;
    "Thursday at 9am" is what a human on a support line would say.
    """
    try:
        when = datetime.fromisoformat(slot)
    except (TypeError, ValueError):
        return slot
    hour = when.hour % 12 or 12
    meridiem = "am" if when.hour < 12 else "pm"
    minutes = f":{when.minute:02d}" if when.minute else ""
    return f"{when.strftime('%A')} at {hour}{minutes}{meridiem}"


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
    session.turn += 1

    clean_text, warnings = sanitize_user_text(user_text)

    start = time.perf_counter()
    try:
        result = await session.agent.send(clean_text)
    except Exception as exc:
        # The one call in this function with a real chance of raising: a live
        # API error (e.g. a 529) or agent/core.py's runaway-tool-loop
        # RuntimeError. Without this, the turn counter has already advanced
        # above but no record explains the gap — the next record would jump
        # straight from turn N-1 to turn N+1. Record what's known (empty
        # reply, no tool calls) and then let the exception propagate
        # unchanged: callers depend on it, and telemetry must never be the
        # thing that swallows a real failure. `except Exception`, never
        # BaseException — asyncio.CancelledError must keep propagating
        # untouched or Pipecat barge-in breaks.
        try:
            log_turn(
                TurnRecord(
                    session_id=session.session_id,
                    customer_id=session.customer_id,
                    transport=session.transport,
                    turn=session.turn,
                    user_text=user_text,
                    reply="",
                    original_reply=None,
                    grounding_flagged=False,
                    hedge_spoken=False,
                    tool_calls=[],
                    llm_latency_seconds=time.perf_counter() - start,
                    warnings=[*warnings, f"Turn failed before a reply was produced: {exc}"],
                    escalated=False,
                    escalation_reason=None,
                    escalation_id=None,
                    ended=True,
                    end_reason="error",
                )
            )
        except Exception:  # noqa: BLE001 — telemetry must never mask the real failure
            pass
        raise
    llm_latency = time.perf_counter() - start

    # Grounding: a flagged reply is never spoken — the customer hears a hedge
    # instead, and a second consecutive flag hands off to a human
    # (EscalationTracker). The "retry" is simply the customer's next turn, so
    # this costs no extra LLM round-trip and no dead air on a live call.
    reply = result.reply
    original_reply: str | None = None
    hedge_spoken = False
    try:
        findings = check_reply_grounding(reply, result.tool_calls)
    except Exception as exc:  # noqa: BLE001 — a guardrail must never break a turn
        findings = []
        warnings.append(f"Could not check reply grounding this turn: {exc}")
    grounding_flagged = bool(findings)
    if findings:
        warnings.extend(findings)
        # Never substitute the hedge when this turn proposed a confirmation
        # (issue_refund / book_appointment / cancel_appointment's first,
        # proposing call) — the customer must hear the real proposal text, or
        # their next "okay" commits an action they were never told about.
        # Detection and the escalation counter still run either way (below);
        # only the substitution is skipped. This is exactly why
        # grounding_flagged and hedge_spoken are two separate telemetry
        # fields, not one: this branch can flag without ever substituting.
        if not _turn_proposed_a_confirmation(result.tool_calls):
            # Rotate on how many consecutive ungrounded replies preceded this
            # one. The tracker's counter is still the PREVIOUS count here —
            # it is incremented inside check_escalation below — so a first
            # flag gets HEDGE_PHRASES[0] and a second consecutive flag gets a
            # different line, which is exactly the point of varying it.
            original_reply = reply
            reply = hedge_for(session.tracker.consecutive_ungrounded_replies)
            hedge_spoken = True
            # Keep session.agent.messages in sync with what the customer
            # actually heard — otherwise the suppressed reply lingers in
            # history for the model to build on next turn.
            reconcile_warning = _substitute_hedge_in_history(session.agent.messages, reply)
            if reconcile_warning:
                warnings.append(reconcile_warning)

    # Check escalation before should_end_session — a trigger here always
    # outranks the model deciding on its own the chat is naturally over.
    try:
        reason = await escalation.check_escalation(
            session.tracker,
            session.agent.messages,
            result.tool_calls,
            ungrounded=grounding_flagged,
        )
    except Exception as exc:  # noqa: BLE001 — a classifier hiccup must not crash the turn
        reason = None
        warnings.append(f"Could not run triage classification this turn: {exc}")

    escalation_id: int | None = None
    packet: dict[str, Any] | None = None
    if reason:
        try:
            packet = await escalation.create_handoff_packet(session.customer_id, session.agent.messages, reason)
            escalation_id = packet["escalation_id"]
            notice = _escalation_notice(packet.get("callback_time"))
        except Exception as exc:  # noqa: BLE001 — exit path must never crash on this
            notice = None
            warnings.append(f"Escalation triggered ({reason}) but the handoff packet couldn't be logged: {exc}")
        outcome = TurnOutcome(
            reply=reply,
            ended=True,
            end_reason="escalated",
            notice=notice,
            escalation_packet=packet,
            llm_latency_seconds=llm_latency,
            warnings=warnings,
        )
    elif should_end_session(result.tool_calls):
        outcome = TurnOutcome(
            reply=reply, ended=True, end_reason="model_ended", llm_latency_seconds=llm_latency, warnings=warnings
        )
    else:
        outcome = TurnOutcome(reply=reply, llm_latency_seconds=llm_latency, warnings=warnings)

    # One emit point, at the single exit. log_turn already guarantees it never
    # raises; this wrapper is belt-and-suspenders on top of that, the same
    # precedent create_handoff_packet sets for notify_escalation — telemetry
    # must never be the thing that breaks a live call.
    try:
        log_turn(
            TurnRecord(
                session_id=session.session_id,
                customer_id=session.customer_id,
                transport=session.transport,
                turn=session.turn,
                user_text=user_text,
                reply=outcome.reply,
                original_reply=original_reply,
                grounding_flagged=grounding_flagged,
                hedge_spoken=hedge_spoken,
                tool_calls=result.tool_calls,
                llm_latency_seconds=llm_latency,
                warnings=outcome.warnings,
                escalated=outcome.end_reason == "escalated",
                escalation_reason=reason,
                escalation_id=escalation_id,
                ended=outcome.ended,
                end_reason=outcome.end_reason,
            )
        )
    except Exception as exc:  # noqa: BLE001 — telemetry must never break a turn
        outcome.warnings.append(f"Could not write the turn log this turn: {exc}")

    return outcome


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
