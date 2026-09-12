"""Phase 12, Task 3 checkpoint: the two async resolution tools that close
out a handover — booking a human callback, or recording that the customer
will reach out themselves.

Every test here repoints mock_db.DB_PATH to a tmp path (autouse fixture
below) rather than touching the developer's real database, and every test
that would otherwise reach a real network call
(agent.tools.escalation._infer_handoff_fields, .notify_escalation) stubs it
— same conventions as tests/test_escalation.py's Task 2 section.
"""

from __future__ import annotations

import copy
from unittest.mock import AsyncMock

import pytest

from agent.confirmation import PendingActionGate
from agent.tools import escalation, handoff, scheduling
from agent.tools.escalation import (
    RESOLUTION_CALLBACK,
    RESOLUTION_SELF,
    STATUS_NONE,
    STATUS_OPEN,
    STATUS_RESOLVED,
    EscalationState,
)
from data import mock_db


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    """Without this, these tests insert appointments/escalations rows into
    the developer's real data/mock_data.db."""
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_handoff.db")
    mock_db.reset_and_seed()


CUSTOMER_ID = mock_db.CUSTOMERS[0][0]  # never hard-coded


def _free_slot() -> str:
    return scheduling.find_available_slots()["slots"][0]


def _open_state(reason: str = "explicit request for a human") -> EscalationState:
    """An EscalationState with a real, already-open handover: a persisted
    escalations row plus a packet — exactly what a MANDATORY/SUGGESTED
    trigger would have produced via escalation.open_escalation() before
    either of Task 3's tools ever runs.

    Built directly against log_escalation (sync) rather than the full async
    open_escalation() + its LLM inference call, so tests that only care about
    the resolution tools' own behavior don't all have to stub the inference
    too. Tests that actually need to exercise the "packet doesn't exist yet"
    path (the accepted-offer tests below) use a bare EscalationState()
    instead, and DO stub the inference.
    """
    fields = escalation.HandoffFields(
        customer_intent="wants a callback",
        conversation_summary="asked to be called back about their order",
        verified_account_info="verified by order ID",
        actions_taken="looked up the order",
        sentiment="neutral",
    )
    escalation_id = escalation.log_escalation(CUSTOMER_ID, reason, fields)
    state = EscalationState()
    state.open(reason)
    state.escalation_id = escalation_id
    state.packet = {
        "escalation_id": escalation_id,
        "reason": reason,
        **fields.model_dump(),
        "items": [reason],
        "resolution": None,
        "callback_time": None,
    }
    return state


async def _stub_infer_fields(customer_id, messages, client=None):
    """Keeps the accepted-offer tests offline — see test_escalation.py's
    identical helper for why every field must be a real string."""
    return escalation.HandoffFields(
        customer_intent="wants a callback",
        conversation_summary="accepted an offered callback",
        verified_account_info="verified by order ID",
        actions_taken="offered a callback",
        sentiment="neutral",
    )


def _recording_notifier(sink: list[dict]):
    async def _notify(packet, **kwargs):
        sink.append(dict(packet))
        return True

    return _notify


# --- schedule_human_callback ---


@pytest.mark.asyncio
async def test_scheduling_a_callback_proposes_before_it_books(monkeypatch):
    """CLAUDE.md rule 6: booking is irreversible, so the first call proposes
    and only a later confirmed turn commits."""
    monkeypatch.setattr(escalation, "notify_escalation", AsyncMock(return_value=True))
    state = _open_state()
    gate = PendingActionGate(turn=1)
    slot = _free_slot()

    result = await handoff.schedule_human_callback(
        slot_time=slot, state=state, gate=gate, customer_id=CUSTOMER_ID, messages=[]
    )

    assert result["scheduled"] is False
    assert result["status"] == "pending_confirmation"
    assert state.status == STATUS_OPEN, "a proposal resolves nothing"
    with mock_db.get_connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM appointments WHERE scheduled_time = ?", (slot,)
        ).fetchone()[0]
    assert count == 0


@pytest.mark.asyncio
async def test_a_confirmed_callback_books_persists_and_notifies_in_one_await(monkeypatch):
    """The whole point of the tools being async. When this returns, the slot
    is held, the escalations row says `callback`, and the human has been
    told — so a turn that dies immediately afterwards cannot lose any of it.
    """
    sent: list[dict] = []
    monkeypatch.setattr(escalation, "notify_escalation", _recording_notifier(sent))
    state = _open_state()
    gate = PendingActionGate(turn=1)
    slot = _free_slot()

    await handoff.schedule_human_callback(
        slot_time=slot, state=state, gate=gate, customer_id=CUSTOMER_ID, messages=[]
    )
    gate.turn = 2  # a later, confirming turn
    result = await handoff.schedule_human_callback(
        slot_time=slot, state=state, gate=gate, customer_id=CUSTOMER_ID, messages=[]
    )

    assert result["scheduled"] is True
    assert result["callback_time"] == slot
    assert "spoken_time" in result and result["spoken_time"]
    assert state.resolution == RESOLUTION_CALLBACK
    assert state.callback_time == slot
    assert state.status == STATUS_RESOLVED
    assert state.pending_persist is False, "nothing left deferred"

    assert len(sent) == 1, f"exactly one notification, got {len(sent)}"
    assert sent[0]["callback_time"] == slot

    with mock_db.get_connection() as conn:
        appt = conn.execute(
            "SELECT scheduled_time FROM appointments WHERE scheduled_time = ?", (slot,)
        ).fetchone()
        escalation_row = conn.execute(
            "SELECT resolution FROM escalations WHERE escalation_id = ?", (state.escalation_id,)
        ).fetchone()
    assert appt is not None and appt["scheduled_time"] == slot
    assert escalation_row["resolution"] == RESOLUTION_CALLBACK


@pytest.mark.asyncio
async def test_a_callback_does_not_share_a_confirmation_key_with_an_appointment(monkeypatch):
    """agent/tools/scheduling.py keys the gate on ("book", slot_time) and
    nothing else. Without a distinct prefix: the customer proposes an
    ordinary appointment at slot X on turn 0 and never confirms it; a
    handover opens; the model proposes a CALLBACK at slot X on a later turn;
    the gate sees a matching pending proposal from an earlier turn and
    commits — booking an action the customer was never asked to confirm. A
    rule 6 violation.
    """
    monkeypatch.setattr(escalation, "notify_escalation", AsyncMock(return_value=True))
    gate = PendingActionGate()
    slot = _free_slot()
    scheduling.book_appointment(
        slot_time=slot, reason="a haircut", state=gate, customer_id=CUSTOMER_ID
    )  # proposal, turn 0
    gate.turn += 1

    # Prove the vulnerability this key_prefix exists to close, on a snapshot
    # of the gate — NOT the real one below, since gate.check() consumes the
    # pending proposal as a side effect and this probe must not interfere
    # with the actual assertion that follows.
    probe = copy.deepcopy(gate)
    would_have_committed_under_the_old_scheme = probe.check(key=("book", slot))
    assert would_have_committed_under_the_old_scheme is True, (
        "sanity check: without a distinct prefix this scenario really would "
        "have silently committed an unconfirmed appointment as a callback"
    )

    result = await handoff.schedule_human_callback(
        slot_time=slot, state=_open_state(), gate=gate, customer_id=CUSTOMER_ID, messages=[]
    )

    assert result["scheduled"] is False, "a callback must need its own confirmation"
    assert result["status"] == "pending_confirmation"
    with mock_db.get_connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM appointments WHERE scheduled_time = ?", (slot,)
        ).fetchone()[0]
    assert count == 0, "must not have booked anything from an unconfirmed proposal"


@pytest.mark.asyncio
async def test_an_unavailable_slot_leaves_the_handover_open(monkeypatch):
    monkeypatch.setattr(escalation, "notify_escalation", AsyncMock(return_value=True))
    state = _open_state()
    gate = PendingActionGate(turn=1)
    slot = _free_slot()
    # Someone else grabs the slot first.
    other_gate = PendingActionGate(turn=1)
    scheduling.book_appointment(slot_time=slot, reason="someone else", state=other_gate, customer_id="CUST-1002")
    other_gate.turn = 2
    scheduling.book_appointment(slot_time=slot, reason="someone else", state=other_gate, customer_id="CUST-1002")

    result = await handoff.schedule_human_callback(
        slot_time=slot, state=state, gate=gate, customer_id=CUSTOMER_ID, messages=[]
    )

    assert result["scheduled"] is False
    assert result["error"] == "slot_unavailable"
    assert state.status == STATUS_OPEN, "a failed booking attempt is not a resolution"
    assert state.resolution is None


@pytest.mark.asyncio
async def test_resolving_twice_does_not_book_a_second_callback(monkeypatch):
    """One customer, one callback, by design."""
    sent: list[dict] = []
    monkeypatch.setattr(escalation, "notify_escalation", _recording_notifier(sent))
    state = _open_state()
    gate = PendingActionGate(turn=1)
    slot = _free_slot()
    await handoff.schedule_human_callback(
        slot_time=slot, state=state, gate=gate, customer_id=CUSTOMER_ID, messages=[]
    )
    gate.turn = 2
    first = await handoff.schedule_human_callback(
        slot_time=slot, state=state, gate=gate, customer_id=CUSTOMER_ID, messages=[]
    )
    assert first["scheduled"] is True

    other_slot = scheduling.find_available_slots()["slots"][1]
    gate.turn = 3
    second = await handoff.schedule_human_callback(
        slot_time=other_slot, state=state, gate=gate, customer_id=CUSTOMER_ID, messages=[]
    )

    assert second["scheduled"] is False
    assert second["error"] == "already_resolved"
    assert len(sent) == 1, "the second attempt must not have notified again"
    with mock_db.get_connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM appointments WHERE customer_id = ? AND status = 'scheduled'",
            (CUSTOMER_ID,),
        ).fetchone()[0]
    assert count == 1, "exactly the one original booking — the second attempt must not have added another"


# --- record_customer_will_reach_out ---


@pytest.mark.asyncio
async def test_a_declined_callback_is_a_real_resolution_and_books_nothing(monkeypatch):
    """"I'll call back when I know my schedule" is a legitimate ending, not a
    failure — and holding a slot they never agreed to is wrong."""
    sent: list[dict] = []
    monkeypatch.setattr(escalation, "notify_escalation", _recording_notifier(sent))
    state = _open_state()
    with mock_db.get_connection() as conn:
        before = conn.execute(
            "SELECT COUNT(*) FROM appointments WHERE customer_id = ?", (CUSTOMER_ID,)
        ).fetchone()[0]

    result = await handoff.record_customer_will_reach_out(escalation=state, customer_id=CUSTOMER_ID, messages=[])

    assert result["recorded"] is True
    assert state.resolution == RESOLUTION_SELF
    assert state.status == STATUS_RESOLVED
    assert state.pending_persist is False
    assert len(sent) == 1
    with mock_db.get_connection() as conn:
        after = conn.execute(
            "SELECT COUNT(*) FROM appointments WHERE customer_id = ?", (CUSTOMER_ID,)
        ).fetchone()[0]
    assert after == before, "declining a callback must not book one"


@pytest.mark.asyncio
async def test_the_tools_refuse_to_manufacture_a_handover_from_nothing():
    """Both tools are registered for EVERY session. A customer saying "I'll
    get back to you when I know my schedule" in an entirely ordinary call is
    a plausible prompt for the model to call record_customer_will_reach_out
    — which would write an escalations row, fire a Slack notification, and
    flip end_reason. Only a real offer or an open handover authorises
    opening one."""
    state = EscalationState()  # nothing open, nothing offered

    result = await handoff.record_customer_will_reach_out(escalation=state, customer_id=CUSTOMER_ID, messages=[])

    assert result["recorded"] is False
    assert result["error"] == "no_handover"
    assert state.status == STATUS_NONE


@pytest.mark.asyncio
async def test_accepting_an_offer_opens_the_handover_the_offer_implied(monkeypatch):
    """A suggested trigger only offers — nothing is open until the customer
    says yes, so the resolution tool has to open it. The offer is what makes
    this legitimate; see the test above for the case with no offer."""
    monkeypatch.setattr(escalation, "notify_escalation", AsyncMock(return_value=True))
    monkeypatch.setattr(escalation, "_infer_handoff_fields", _stub_infer_fields)
    state = EscalationState()
    state.offered.add("repeated failed lookups")

    result = await handoff.record_customer_will_reach_out(escalation=state, customer_id=CUSTOMER_ID, messages=[])

    assert result["recorded"] is True
    assert state.status == STATUS_RESOLVED and state.items
    assert state.escalation_id is not None
    with mock_db.get_connection() as conn:
        row = conn.execute(
            "SELECT resolution FROM escalations WHERE escalation_id = ?", (state.escalation_id,)
        ).fetchone()
    assert row["resolution"] == RESOLUTION_SELF


@pytest.mark.asyncio
async def test_already_resolved_is_refused_on_the_second_call(monkeypatch):
    """Calling either tool a second time after a resolution is already
    recorded must not silently overwrite it — that would let a customer's
    later, unrelated "I'll call you" turn quietly reclassify a completed
    callback as a self-resolution."""
    monkeypatch.setattr(escalation, "notify_escalation", AsyncMock(return_value=True))
    state = _open_state()
    await handoff.record_customer_will_reach_out(escalation=state, customer_id=CUSTOMER_ID, messages=[])

    result = await handoff.record_customer_will_reach_out(escalation=state, customer_id=CUSTOMER_ID, messages=[])

    assert result["recorded"] is False
    assert result["error"] == "already_resolved"
