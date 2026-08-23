"""find_available_slots / book_appointment / cancel_appointment tools.

A mock calendar over the appointments table (Phase 0): fixed business
hours, no real calendar API. This is the plan's first feature needing
genuine multi-turn negotiation, not just a single lookup.

CLAUDE.md rule 6 ("never let a tool execute an irreversible action... —
booking/cancelling — without an explicit confirmation turn from the user
first") applies directly here. Rather than trust the model's own say-so (a
prompt instruction alone), book_appointment/cancel_appointment enforce
this at the code level: the first call for a given action only *proposes*
it; committing requires a second call in a LATER conversation turn — never
the same one — referencing the same pending proposal. This mirrors
PROJECT_PLAN.md's literal 3-tool list (no separate confirm_* tool) while
still making "confirmed within one breath" structurally impossible.

Double-booking and a slot vanishing before confirmation are handled by the
same mechanism: availability is re-checked fresh at BOTH the propose step
and the confirm step, so a slot someone else grabbed in between is caught
at confirm time even if it was free when first proposed.

SchedulingState is per-session, not global — Phase 9's telephony server
will handle multiple concurrent calls in one process, and global state
would leak across them. See transport/text_cli.py's build_dispatch_tool
for how a fresh instance is threaded in and its `turn` counter advanced.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from data.mock_db import get_connection

BUSINESS_START_HOUR = 9
BUSINESS_END_HOUR = 17  # last slot starts at 16:30
SLOT_MINUTES = 30
BUSINESS_DAYS_AHEAD = 5  # how many business days find_available_slots searches by default
MAX_SLOTS_RETURNED = 10


def _business_days(start: datetime, count: int):
    """Yield `count` business-day dates (Mon-Fri), starting the day after `start`."""
    current = start.date() + timedelta(days=1)
    yielded = 0
    while yielded < count:
        if current.weekday() < 5:  # Mon=0 .. Fri=4
            yield current
            yielded += 1
        current += timedelta(days=1)


def _all_slots_for_day(day: date) -> list[str]:
    slots = []
    hour, minute = BUSINESS_START_HOUR, 0
    while hour < BUSINESS_END_HOUR:
        # Naive on purpose: matches the appointments table's existing
        # scheduled_time convention (Phase 0's seed data), which stores
        # local business hours as plain ISO strings, not UTC. These need
        # to string-match exactly against _booked_slot_times()'s DB reads.
        slots.append(datetime(day.year, day.month, day.day, hour, minute).isoformat())  # noqa: DTZ001
        minute += SLOT_MINUTES
        if minute >= 60:
            minute -= 60
            hour += 1
    return slots


def _booked_slot_times() -> set[str]:
    with get_connection() as conn:
        rows = conn.execute("SELECT scheduled_time FROM appointments WHERE status = 'scheduled'").fetchall()
    return {row["scheduled_time"] for row in rows}


FIND_SLOTS_SCHEMA: dict[str, Any] = {
    "name": "find_available_slots",
    "description": (
        "List available appointment/callback slots. Business hours only "
        "(9 AM-5 PM, Monday-Friday, 30-minute slots). If the customer asks "
        "for a different week or date, call this again with start_date set "
        "accordingly — don't assume, re-search."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "start_date": {
                "type": "string",
                "description": (
                    "Search starting after this date (YYYY-MM-DD) — results "
                    "begin the next business day after it. Omit to search "
                    "starting tomorrow."
                ),
            },
        },
    },
}


def find_available_slots(start_date: str | None = None, now: datetime | None = None) -> dict[str, Any]:
    """List open slots over the next few business days. Never raises."""
    reference = datetime.fromisoformat(start_date) if start_date else (now or datetime.now())  # noqa: DTZ005 — naive on purpose, see _all_slots_for_day
    booked = _booked_slot_times()

    available: list[str] = []
    for day in _business_days(reference, BUSINESS_DAYS_AHEAD):
        for slot in _all_slots_for_day(day):
            if slot not in booked:
                available.append(slot)
                if len(available) >= MAX_SLOTS_RETURNED:
                    break
        if len(available) >= MAX_SLOTS_RETURNED:
            break

    if not available:
        return {"slots": [], "message": "No available slots found in that window."}
    return {"slots": available}


@dataclass
class SchedulingState:
    """Per-session state: which turn we're on, and any pending (proposed
    but not yet confirmed) booking or cancellation. One instance per
    session — see the module docstring.
    """

    turn: int = 0
    pending: dict[str, Any] | None = None


BOOK_APPOINTMENT_SCHEMA: dict[str, Any] = {
    "name": "book_appointment",
    "description": (
        "Book an appointment slot (from find_available_slots) for the "
        "customer. The FIRST call proposes the booking and asks for "
        "confirmation — it does not book yet. Only call it a second time, "
        "with the exact same slot_time and reason, after the customer has "
        "clearly confirmed in their own words in a later message. Never "
        "call it twice in the same reply."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "slot_time": {
                "type": "string",
                "description": "The exact slot, as returned by find_available_slots.",
            },
            "reason": {"type": "string", "description": "Why the customer is booking this appointment."},
        },
        "required": ["slot_time", "reason"],
    },
}


def book_appointment(slot_time: str, reason: str, state: SchedulingState, customer_id: str) -> dict[str, Any]:
    """Propose-then-confirm booking. See module docstring for the mechanism."""
    if slot_time in _booked_slot_times():
        state.pending = None
        return {
            "booked": False,
            "error": "slot_unavailable",
            "message": f"{slot_time} is no longer available — someone else has booked it.",
        }

    pending = state.pending
    is_confirmation = (
        pending is not None
        and pending.get("kind") == "book"
        and pending.get("slot_time") == slot_time
        and pending.get("reason") == reason
        and pending["proposed_turn"] < state.turn
    )

    if not is_confirmation:
        state.pending = {"kind": "book", "slot_time": slot_time, "reason": reason, "proposed_turn": state.turn}
        return {
            "booked": False,
            "status": "pending_confirmation",
            "message": f"{slot_time} is available for '{reason}'. Should I go ahead and book it?",
        }

    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO appointments (customer_id, scheduled_time, reason, status) VALUES (?, ?, ?, 'scheduled')",
            (customer_id, slot_time, reason),
        )
        appointment_id = cursor.lastrowid

    state.pending = None
    return {"booked": True, "appointment_id": appointment_id, "slot_time": slot_time, "reason": reason}


CANCEL_APPOINTMENT_SCHEMA: dict[str, Any] = {
    "name": "cancel_appointment",
    "description": (
        "Cancel one of the customer's own scheduled appointments. The "
        "FIRST call proposes the cancellation and asks for confirmation — "
        "it does not cancel yet. Only call it a second time after the "
        "customer has clearly confirmed in a later message. If the "
        "customer has more than one scheduled appointment, pass slot_time "
        "to say which one; otherwise it's optional."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "slot_time": {
                "type": "string",
                "description": "Which appointment to cancel, if the customer has more than one scheduled.",
            },
        },
    },
}


def _customers_scheduled_appointments(customer_id: str) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT appointment_id, scheduled_time, reason FROM appointments "
            "WHERE customer_id = ? AND status = 'scheduled' ORDER BY scheduled_time",
            (customer_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def cancel_appointment(state: SchedulingState, customer_id: str, slot_time: str | None = None) -> dict[str, Any]:
    """Propose-then-confirm cancellation, scoped to this customer's own appointments."""
    appointments = _customers_scheduled_appointments(customer_id)
    matches = [a for a in appointments if a["scheduled_time"] == slot_time] if slot_time else appointments

    if not matches:
        state.pending = None
        return {"cancelled": False, "error": "not_found", "message": "No matching scheduled appointment found."}
    if len(matches) > 1:
        state.pending = None
        return {
            "cancelled": False,
            "error": "ambiguous",
            "message": "This customer has more than one scheduled appointment — ask which one.",
            "appointments": matches,
        }

    target = matches[0]
    pending = state.pending
    is_confirmation = (
        pending is not None
        and pending.get("kind") == "cancel"
        and pending.get("appointment_id") == target["appointment_id"]
        and pending["proposed_turn"] < state.turn
    )

    if not is_confirmation:
        state.pending = {"kind": "cancel", "appointment_id": target["appointment_id"], "proposed_turn": state.turn}
        return {
            "cancelled": False,
            "status": "pending_confirmation",
            "message": f"Cancel the appointment at {target['scheduled_time']}? This can't be undone.",
            "appointment": target,
        }

    with get_connection() as conn:
        conn.execute("UPDATE appointments SET status = 'cancelled' WHERE appointment_id = ?", (target["appointment_id"],))

    state.pending = None
    return {"cancelled": True, "appointment_id": target["appointment_id"], "slot_time": target["scheduled_time"]}
