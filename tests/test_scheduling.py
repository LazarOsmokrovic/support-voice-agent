"""Phase 5 checkpoint (deterministic half): double-booking, a slot that
vanishes before confirmation, and cancellation, all tested directly against
the tool functions — no network, no LLM needed for any of this, since the
propose-then-confirm mechanism and slot availability are pure logic.

"Reschedule" — the plan's other checkpoint scenario — is inherently
conversational (it's the model choosing to book-then-cancel across a
multi-turn negotiation), so that one is a live scripted conversation in
tests/test_text_cli.py instead, alongside the project's other full-loop
checkpoints.
"""

from __future__ import annotations

from datetime import datetime

from agent.confirmation import PendingActionGate
from agent.tools.scheduling import (
    book_appointment,
    cancel_appointment,
    find_available_slots,
)
from data import mock_db


def _seed(tmp_path, monkeypatch):
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_scheduling.db")
    mock_db.reset_and_seed()


# --- find_available_slots ---


def test_find_available_slots_only_returns_business_hours_on_weekdays(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)

    result = find_available_slots(now=datetime(2026, 8, 24, 10, 0))  # a Monday  # noqa: DTZ001 — naive on purpose, matches scheduling.py

    assert result["slots"], "expected some available slots"
    for slot in result["slots"]:
        dt = datetime.fromisoformat(slot)
        assert dt.weekday() < 5, f"{slot} falls on a weekend"
        assert 9 <= dt.hour < 17, f"{slot} is outside business hours"
        assert dt.minute in (0, 30), f"{slot} isn't on the 30-minute grid"


def test_find_available_slots_excludes_already_booked_slots(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    now = datetime(2026, 8, 24, 10, 0)  # noqa: DTZ001 — naive on purpose, matches scheduling.py
    first_pass = find_available_slots(now=now)
    taken_slot = first_pass["slots"][0]

    with mock_db.get_connection() as conn:
        conn.execute(
            "INSERT INTO appointments (customer_id, scheduled_time, reason, status) "
            "VALUES (?, ?, ?, 'scheduled')",
            (mock_db.CUSTOMERS[0][0], taken_slot, "test booking"),
        )

    second_pass = find_available_slots(now=now)

    assert taken_slot not in second_pass["slots"]


def test_find_available_slots_respects_start_date_override(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)

    near = find_available_slots(now=datetime(2026, 8, 24, 10, 0))  # noqa: DTZ001 — naive on purpose, matches scheduling.py
    later = find_available_slots(start_date="2026-09-01")

    assert near["slots"] != later["slots"]
    assert all(slot > "2026-09-01" for slot in later["slots"])


# --- book_appointment: propose-then-confirm, double-booking, vanishing slots ---


def test_book_appointment_first_call_only_proposes(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)

    result = book_appointment("2026-08-25T09:00:00", "callback", state=state, customer_id="CUST-1001")

    assert result["booked"] is False
    assert result["status"] == "pending_confirmation"
    with mock_db.get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM appointments WHERE scheduled_time = ?", ("2026-08-25T09:00:00",)).fetchone()[0]
    assert count == 0


def test_book_appointment_confirms_in_a_later_turn(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)
    book_appointment("2026-08-25T09:00:00", "callback", state=state, customer_id="CUST-1001")

    state.turn = 2  # a later turn
    result = book_appointment("2026-08-25T09:00:00", "callback", state=state, customer_id="CUST-1001")

    assert result["booked"] is True
    assert "appointment_id" in result
    assert state.pending is None


def test_book_appointment_redacts_pii_in_reason_before_writing(tmp_path, monkeypatch):
    """Phase 10a: appointments.reason is model-authored free text derived
    from what the caller said — a storage boundary, so it's redacted before
    the write (guardrails/pii.py), same as summary.py's log_ticket."""
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)
    reason = "call me back on 555-123-4567"
    book_appointment("2026-08-25T09:00:00", reason, state=state, customer_id="CUST-1001")

    state.turn = 2
    result = book_appointment("2026-08-25T09:00:00", reason, state=state, customer_id="CUST-1001")

    assert result["booked"] is True
    with mock_db.get_connection() as conn:
        stored_reason = conn.execute(
            "SELECT reason FROM appointments WHERE scheduled_time = ?", ("2026-08-25T09:00:00",)
        ).fetchone()["reason"]
    assert "555-123-4567" not in stored_reason
    assert "[redacted-phone]" in stored_reason


def test_book_appointment_rejects_confirmation_attempted_in_the_same_turn(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)
    book_appointment("2026-08-25T09:00:00", "callback", state=state, customer_id="CUST-1001")

    result = book_appointment("2026-08-25T09:00:00", "callback", state=state, customer_id="CUST-1001")

    assert result["booked"] is False
    assert result["status"] == "pending_confirmation"


def test_book_appointment_rejects_double_booking(tmp_path, monkeypatch):
    """Phase 5 checkpoint: a double-booking attempt."""
    _seed(tmp_path, monkeypatch)
    state_a = PendingActionGate(turn=1)
    book_appointment("2026-08-25T09:00:00", "callback", state=state_a, customer_id="CUST-1001")
    state_a.turn = 2
    first = book_appointment("2026-08-25T09:00:00", "callback", state=state_a, customer_id="CUST-1001")
    assert first["booked"] is True

    state_b = PendingActionGate(turn=1)
    second = book_appointment("2026-08-25T09:00:00", "different reason", state=state_b, customer_id="CUST-1002")

    assert second["booked"] is False
    assert second["error"] == "slot_unavailable"


def test_book_appointment_rejects_slot_that_vanished_before_confirmation(tmp_path, monkeypatch):
    """Phase 5 checkpoint: a slot that no longer exists by the time of
    confirmation — proposed while free, but grabbed by someone else before
    the confirming call comes in.
    """
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)
    proposal = book_appointment("2026-08-25T09:00:00", "callback", state=state, customer_id="CUST-1001")
    assert proposal["status"] == "pending_confirmation"

    # A different customer grabs the same slot in between.
    other_state = PendingActionGate(turn=1)
    book_appointment("2026-08-25T09:00:00", "someone else's reason", state=other_state, customer_id="CUST-1002")
    other_state.turn = 2
    book_appointment("2026-08-25T09:00:00", "someone else's reason", state=other_state, customer_id="CUST-1002")

    state.turn = 2
    result = book_appointment("2026-08-25T09:00:00", "callback", state=state, customer_id="CUST-1001")

    assert result["booked"] is False
    assert result["error"] == "slot_unavailable"
    assert state.pending is None  # cleared, so the model can propose something else


def test_a_new_proposal_replaces_the_old_pending_one(tmp_path, monkeypatch):
    """Handles a mid-conversation correction ('actually, next week instead')."""
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)
    book_appointment("2026-08-25T09:00:00", "callback", state=state, customer_id="CUST-1001")

    state.turn = 2
    book_appointment("2026-08-26T10:00:00", "callback", state=state, customer_id="CUST-1001")  # changed their mind

    state.turn = 3
    # Confirming the OLD slot should not go through — it's no longer what's pending.
    stale_confirm = book_appointment("2026-08-25T09:00:00", "callback", state=state, customer_id="CUST-1001")
    assert stale_confirm["status"] == "pending_confirmation"  # treated as a fresh proposal, not a confirmation

    state.turn = 4
    real_confirm = book_appointment("2026-08-26T10:00:00", "callback", state=state, customer_id="CUST-1001")
    assert real_confirm["booked"] is False  # stale_confirm above overwrote the pending state again


# --- cancel_appointment: propose-then-confirm, ownership, ambiguity ---


def test_cancel_appointment_first_call_only_proposes(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)
    book_appointment("2026-08-25T09:00:00", "callback", state=state, customer_id="CUST-1001")
    state.turn = 2
    book_appointment("2026-08-25T09:00:00", "callback", state=state, customer_id="CUST-1001")

    state.turn = 3
    result = cancel_appointment(state=state, customer_id="CUST-1001")

    assert result["cancelled"] is False
    assert result["status"] == "pending_confirmation"
    with mock_db.get_connection() as conn:
        status = conn.execute(
            "SELECT status FROM appointments WHERE scheduled_time = ?", ("2026-08-25T09:00:00",)
        ).fetchone()["status"]
    assert status == "scheduled"


def test_cancel_appointment_confirms_in_a_later_turn(tmp_path, monkeypatch):
    """Phase 5 checkpoint: cancellation."""
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)
    book_appointment("2026-08-25T09:00:00", "callback", state=state, customer_id="CUST-1001")
    state.turn = 2
    book_appointment("2026-08-25T09:00:00", "callback", state=state, customer_id="CUST-1001")

    state.turn = 3
    cancel_appointment(state=state, customer_id="CUST-1001")
    state.turn = 4
    result = cancel_appointment(state=state, customer_id="CUST-1001")

    assert result["cancelled"] is True
    with mock_db.get_connection() as conn:
        status = conn.execute(
            "SELECT status FROM appointments WHERE scheduled_time = ?", ("2026-08-25T09:00:00",)
        ).fetchone()["status"]
    assert status == "cancelled"


def test_cancelling_frees_the_slot_for_rebooking(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)
    book_appointment("2026-08-25T09:00:00", "callback", state=state, customer_id="CUST-1001")
    state.turn = 2
    book_appointment("2026-08-25T09:00:00", "callback", state=state, customer_id="CUST-1001")
    state.turn = 3
    cancel_appointment(state=state, customer_id="CUST-1001")
    state.turn = 4
    cancel_appointment(state=state, customer_id="CUST-1001")

    other_state = PendingActionGate(turn=1)
    result = book_appointment("2026-08-25T09:00:00", "a different customer's callback", state=other_state, customer_id="CUST-1002")

    assert result["status"] == "pending_confirmation"  # slot is free again, not rejected as unavailable


def test_cancel_appointment_will_not_cancel_someone_elses_appointment(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)
    book_appointment("2026-08-25T09:00:00", "callback", state=state, customer_id="CUST-1001")
    state.turn = 2
    book_appointment("2026-08-25T09:00:00", "callback", state=state, customer_id="CUST-1001")

    other_state = PendingActionGate(turn=1)
    result = cancel_appointment(state=other_state, customer_id="CUST-1002")

    assert result["cancelled"] is False
    assert result["error"] == "not_found"


def test_cancel_appointment_reports_not_found_when_none_scheduled(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)

    result = cancel_appointment(state=state, customer_id="CUST-1001")

    assert result["cancelled"] is False
    assert result["error"] == "not_found"


def test_cancel_appointment_reports_ambiguous_with_multiple_scheduled(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)
    book_appointment("2026-08-25T09:00:00", "first", state=state, customer_id="CUST-1001")
    state.turn = 2
    book_appointment("2026-08-25T09:00:00", "first", state=state, customer_id="CUST-1001")
    state.turn = 3
    book_appointment("2026-08-26T10:00:00", "second", state=state, customer_id="CUST-1001")
    state.turn = 4
    book_appointment("2026-08-26T10:00:00", "second", state=state, customer_id="CUST-1001")

    state.turn = 5
    result = cancel_appointment(state=state, customer_id="CUST-1001")

    assert result["cancelled"] is False
    assert result["error"] == "ambiguous"
    assert len(result["appointments"]) == 2


def test_cancel_appointment_disambiguates_via_slot_time(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)
    book_appointment("2026-08-25T09:00:00", "first", state=state, customer_id="CUST-1001")
    state.turn = 2
    book_appointment("2026-08-25T09:00:00", "first", state=state, customer_id="CUST-1001")
    state.turn = 3
    book_appointment("2026-08-26T10:00:00", "second", state=state, customer_id="CUST-1001")
    state.turn = 4
    book_appointment("2026-08-26T10:00:00", "second", state=state, customer_id="CUST-1001")

    state.turn = 5
    cancel_appointment(state=state, customer_id="CUST-1001", slot_time="2026-08-26T10:00:00")
    state.turn = 6
    result = cancel_appointment(state=state, customer_id="CUST-1001", slot_time="2026-08-26T10:00:00")

    assert result["cancelled"] is True
    assert result["slot_time"] == "2026-08-26T10:00:00"
    with mock_db.get_connection() as conn:
        remaining = conn.execute(
            "SELECT COUNT(*) FROM appointments WHERE customer_id = ? AND status = 'scheduled'", ("CUST-1001",)
        ).fetchone()[0]
    assert remaining == 1  # the 08-25 one is still scheduled


def test_confirmation_survives_the_model_rephrasing_the_reason(tmp_path, monkeypatch):
    """The booking loop bug, as a permanent regression test.

    `reason` is model-authored prose, and a model legitimately rephrases prose
    between turns — "callback about my Kindle" becoming "call back regarding
    the Kindle order". The gate used to key on ("book", slot_time, reason), so
    a customer's "yes" arrived under a different key and check() read the
    confirmation as a brand-new proposal. The agent asked the same question
    again, and again, for as long as the customer kept agreeing.

    Keying on the slot alone fixes it: the slot is what identifies a booking,
    the reason merely describes it. issue_refund and cancel_appointment never
    had this bug because they key on stable identifiers rather than sentences.
    """
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)
    slot = find_available_slots()["slots"][0]

    proposal = book_appointment(slot, "callback about my Kindle", state=state, customer_id="CUST-1001")
    assert proposal["booked"] is False
    assert proposal["status"] == "pending_confirmation"

    # The customer says yes; the model re-words the reason on its way back.
    state.turn = 2  # a later turn
    confirmed = book_appointment(
        slot, "call back regarding the Kindle order", state=state, customer_id="CUST-1001"
    )

    assert confirmed["booked"] is True, "a rephrased reason must not restart the confirmation loop"


def test_a_different_slot_still_requires_its_own_confirmation(tmp_path, monkeypatch):
    """The other half of the same tension. Loosening the key must not make the
    gate accept a booking the customer never agreed to: changing the SLOT is a
    genuinely different action and has to be proposed on its own."""
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)
    slots = find_available_slots()["slots"]

    book_appointment(slots[0], "callback", state=state, customer_id="CUST-1001")
    state.turn = 2  # a later turn
    other = book_appointment(slots[1], "callback", state=state, customer_id="CUST-1001")

    assert other["booked"] is False
    assert other["status"] == "pending_confirmation"
