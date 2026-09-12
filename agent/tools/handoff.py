"""Two tools that resolve an open (or offered) handover: booking a human
callback, or recording that the customer will reach out themselves.

Both are `async def`. agent/core.py:147 already awaits awaitable tool
output (`if hasattr(output, "__await__"): output = await output`), so an
`async def` handler works today with zero changes to agent/core.py or any
transport. That matters here specifically: these tools persist the
resolution and notify the human colleague INLINE, in one `await`, at the
moment the customer agrees to it — a deferred resolution (write it now,
notify it later on some other turn) is one a failing turn in between could
lose entirely. See EscalationState.pending_persist for the fallback (Session
close still flushes it) for the case where even this inline attempt fails.

Unlike agent/tools/escalation.py's open_escalation/resolve_escalation, these
two ARE tools the model calls itself — they're the resolution half of a
handover the model is actively negotiating with the customer
("would you like a callback, or would you rather reach out yourself?").
Both share one guard, `_ensure_open`: they are registered for EVERY
session, so without it a customer saying something as ordinary as "I'll get
back to you when I know my schedule" could make the model manufacture a
handover — an escalations row, a Slack notification, a changed end_reason —
out of an entirely unremarkable conversation.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from agent.confirmation import PendingActionGate
from agent.tools import escalation as escalation_module
from agent.tools import scheduling
from agent.tools.escalation import EscalationState, RESOLUTION_CALLBACK, RESOLUTION_SELF, STATUS_NONE

logger = logging.getLogger("agent.tools.handoff")

# Not one of escalation.py's MANDATORY_REASONS/SUGGESTED_REASONS — this one
# is never a trigger a tracker fires on its own. It exists purely to give
# _ensure_open's state.open() call a human-readable `items` entry for the
# one case that reaches it with nothing already recorded: a suggested
# trigger OFFERED a callback (state.offered), and the customer just said yes.
ACCEPTED_OFFER_REASON = "customer accepted an offer of a callback"


def _ensure_open(state: EscalationState) -> bool:
    """Guard and open. Returns False when there is no handover to resolve.

    Only a real offer, or an already-open handover, authorises opening one.
    Both tools are registered for every session, so without this guard a
    model could call one during an entirely ordinary conversation — "I'll
    get back to you when I know my schedule" is a plausible prompt — and
    manufacture an escalations row, a Slack message and a changed end_reason
    out of nothing.
    """
    if state.status != STATUS_NONE:
        return True
    if not state.offered:
        return False
    state.open(ACCEPTED_OFFER_REASON)
    return True


def _spoken_time(slot: str) -> str:
    """Turn an ISO slot into something a person would say.

    Duplicated from agent/session.py's private helper of the same name
    rather than imported: agent/session.py imports this module to register
    its tools, so importing back from session.py here would be circular.
    """
    try:
        when = datetime.fromisoformat(slot)
    except (TypeError, ValueError):
        return slot
    hour = when.hour % 12 or 12
    meridiem = "am" if when.hour < 12 else "pm"
    minutes = f":{when.minute:02d}" if when.minute else ""
    return f"{when.strftime('%A')} at {hour}{minutes}{meridiem}"


async def _persist_resolution(state: EscalationState, customer_id: str, messages: list[dict[str, Any]]) -> None:
    """Open the packet if there is none, then finalize and notify.

    Callers wrap this in try/except and must NOT clear
    `state.pending_persist` themselves on failure — record_resolution()
    already set it True, and leaving it True is what lets close_session
    retry the flush later (D-5). Only this function's own success path
    clears it.
    """
    if state.packet is None:
        # Normally the handover was already opened (a MANDATORY/SUGGESTED
        # trigger calls escalation.open_escalation with the real
        # transcript) by the time one of these tools runs — this branch only
        # exists for the accepted-offer path, where _ensure_open merely
        # flipped a status flag and never persisted anything. Both callers
        # pass the session's real, live transcript here (see each tool's
        # dispatch wiring in agent/session.py), so the inferred packet is
        # built from the actual conversation, not a blank one.
        state.packet = await escalation_module.open_escalation(
            customer_id, messages, "; ".join(state.items) or ACCEPTED_OFFER_REASON
        )
        state.escalation_id = state.packet["escalation_id"]
    await escalation_module.resolve_escalation(
        state.packet,
        items=state.items,
        resolution=state.resolution,
        callback_time=state.callback_time,
    )
    state.pending_persist = False


def _already_resolved_message(state: EscalationState) -> str:
    detail = state.resolution or ""
    if state.callback_time:
        detail += f", callback at {state.callback_time}"
    return f"This handover is already resolved ({detail}) — anything else they raise goes to the same colleague."


SCHEDULE_CALLBACK_SCHEMA: dict[str, Any] = {
    "name": "schedule_human_callback",
    "description": (
        "Book a human colleague to call the customer back about their open "
        "issue — call this only once the customer has agreed to a callback, "
        "either because you offered one and they accepted it, or because a "
        "handover is already open and they'd rather be called back than "
        "transferred now. The FIRST call proposes the slot (from "
        "find_available_slots) and asks for confirmation — it does not book "
        "yet. Only call it a second time, with the exact same slot_time, "
        "after the customer has clearly confirmed in a LATER message. Never "
        "call this in an ordinary conversation with no handover open or "
        "offered."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "slot_time": {
                "type": "string",
                "description": "The exact slot, as returned by find_available_slots.",
            },
        },
        "required": ["slot_time"],
    },
}


async def schedule_human_callback(
    slot_time: str,
    state: EscalationState,
    gate: PendingActionGate,
    customer_id: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Propose-then-confirm booking of a human callback, then resolve the
    handover the moment it's actually booked. See module docstring for why
    persisting and notifying happen inline, in this same await, rather than
    deferred.

    Every branch returns a `scheduled: bool` key — including the
    pending-confirmation and slot-unavailable branches, which merely forward
    scheduling.book_appointment's own dict (keyed on `booked`, not
    `scheduled`) if not translated here.
    """
    if state.resolution is not None:
        return {"scheduled": False, "error": "already_resolved", "message": _already_resolved_message(state)}

    if not _ensure_open(state):
        return {
            "scheduled": False,
            "error": "no_handover",
            "message": "There is no handover to arrange — help them normally.",
        }

    booking = scheduling.book_appointment(
        slot_time=slot_time,
        reason="human callback",
        state=gate,
        customer_id=customer_id,
        key_prefix="callback",
    )

    if booking.get("status") == "pending_confirmation":
        return {"scheduled": False, "status": "pending_confirmation", "message": booking["message"]}

    if not booking["booked"]:
        # State stays open — a vanished slot is not a resolution, it's just
        # a booking attempt that failed; the customer still needs a callback.
        return {"scheduled": False, "error": booking["error"], "message": booking["message"]}

    state.record_resolution(RESOLUTION_CALLBACK, slot_time)
    try:
        await _persist_resolution(state, customer_id, messages)
    except Exception:  # noqa: BLE001 — a tool must never raise into the loop
        logger.exception("failed to persist a resolved callback handover")

    return {
        "scheduled": True,
        "appointment_id": booking["appointment_id"],
        "callback_time": slot_time,
        "spoken_time": _spoken_time(slot_time),
    }


RECORD_CALLBACK_DECLINED_SCHEMA: dict[str, Any] = {
    "name": "record_customer_will_reach_out",
    "description": (
        "Record that the customer — already offered a human callback, or "
        "already in an open handover — has declined a scheduled callback "
        "and will reach out themselves once they know their own "
        "availability. This resolves and closes out the handover. Never "
        "call this in an ordinary conversation with no handover open or "
        "offered — a customer mentioning they'll get back to you is not, on "
        "its own, reason to open one."
    ),
    "input_schema": {"type": "object", "properties": {}},
}


async def record_customer_will_reach_out(
    escalation: EscalationState,
    customer_id: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Same guard as schedule_human_callback, then record RESOLUTION_SELF
    and run the same awaited persist. "I'll call back when I know my
    schedule" is a legitimate ending to a handover, not a failure of it —
    and it books nothing, since holding a slot the customer never agreed to
    would be wrong.
    """
    if escalation.resolution is not None:
        return {"recorded": False, "error": "already_resolved", "message": _already_resolved_message(escalation)}

    if not _ensure_open(escalation):
        return {
            "recorded": False,
            "error": "no_handover",
            "message": "There is no handover to arrange — help them normally.",
        }

    escalation.record_resolution(RESOLUTION_SELF)
    try:
        await _persist_resolution(escalation, customer_id, messages)
    except Exception:  # noqa: BLE001 — a tool must never raise into the loop
        logger.exception("failed to persist a customer-will-reach-out resolution")

    return {"recorded": True}
