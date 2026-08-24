"""Phase 6 checkpoint: the returns & refunds decision path, tested directly
against issue_refund — no network needed for any of this, since every step
(ownership, window-by-condition, amount, threshold) is deterministic. The
one live call it makes internally (search_policy) uses the local embedding
backend, same as Phase 3's tests — no API key required.
"""

from __future__ import annotations

from datetime import datetime

from agent.confirmation import PendingActionGate
from agent.tools.refunds import HIGH_VALUE_REFUND_THRESHOLD, issue_refund
from data import mock_db


def _seed(tmp_path, monkeypatch):
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_refunds.db")
    mock_db.reset_and_seed()


# Seeded delivered orders (see data/mock_db.py):
# 112-3487561-2938471  CUST-1001  Echo Dot            $34.99   delivered 2026-08-13
# 116-1029384-7563829  CUST-1003  Nike Air Zoom shoes $129.95  delivered 2026-08-08
# 119-5647382-9182736  CUST-1005  Sony WH-1000XM5     $349.99  delivered 2026-08-02
LOW_VALUE_ORDER = "112-3487561-2938471"
LOW_VALUE_CUSTOMER = "CUST-1001"
HIGH_VALUE_ORDER = "119-5647382-9182736"
HIGH_VALUE_CUSTOMER = "CUST-1005"

RECENT_NOW = datetime(2026, 8, 15, 10, 0)  # noqa: DTZ001 — naive on purpose, matches refunds.py; a couple days after 08-13, within any window


def test_rejects_invalid_order_id_format(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)

    result = issue_refund(
        "not-an-order", "unopened_or_unwanted", "changed my mind", state=state, customer_id=LOW_VALUE_CUSTOMER
    )

    assert result["issued"] is False
    assert result["error"] == "invalid_order_id"


def test_rejects_unknown_order(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)

    result = issue_refund(
        "999-9999999-9999999", "unopened_or_unwanted", "n/a", state=state, customer_id=LOW_VALUE_CUSTOMER
    )

    assert result["issued"] is False
    assert result["error"] == "not_found"


def test_rejects_order_belonging_to_another_customer(tmp_path, monkeypatch):
    """Ownership check — same pattern as Phase 5's cancel_appointment, more
    important here since this moves money."""
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)

    result = issue_refund(
        LOW_VALUE_ORDER, "unopened_or_unwanted", "not mine", state=state, customer_id="CUST-1002"
    )

    assert result["issued"] is False
    assert result["error"] == "not_your_order"


def test_rejects_order_not_yet_delivered(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)
    # 115-... is "Processing" per the seed data.
    result = issue_refund(
        "115-4857392-8374651", "unopened_or_unwanted", "too soon", state=state, customer_id="CUST-1002"
    )

    assert result["issued"] is False
    assert result["error"] == "not_delivered"


def test_opened_software_condition_is_never_eligible_regardless_of_window(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)

    result = issue_refund(
        LOW_VALUE_ORDER,
        "opened_software_or_digital",
        "opened it",
        state=state,
        customer_id=LOW_VALUE_CUSTOMER,
        now=RECENT_NOW,
    )

    assert result["issued"] is False
    assert result["error"] == "not_eligible"


def test_rejects_standard_return_outside_the_30_day_window(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)
    far_future = datetime(2026, 9, 20, 10, 0)  # noqa: DTZ001 — naive on purpose, matches refunds.py; well over 30 days after 2026-08-13

    result = issue_refund(
        LOW_VALUE_ORDER, "unopened_or_unwanted", "too late", state=state, customer_id=LOW_VALUE_CUSTOMER, now=far_future
    )

    assert result["issued"] is False
    assert result["error"] == "outside_window"


def test_rejects_damaged_claim_outside_the_14_day_window(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)
    three_weeks_later = datetime(2026, 9, 3, 10, 0)  # noqa: DTZ001 — naive on purpose, matches refunds.py; >14 days after 2026-08-13, but <30

    result = issue_refund(
        LOW_VALUE_ORDER,
        "damaged_or_defective",
        "arrived cracked",
        state=state,
        customer_id=LOW_VALUE_CUSTOMER,
        now=three_weeks_later,
    )

    assert result["issued"] is False
    assert result["error"] == "outside_window"


def test_damaged_claim_within_the_14_day_window_is_eligible(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)

    result = issue_refund(
        LOW_VALUE_ORDER,
        "damaged_or_defective",
        "arrived cracked",
        state=state,
        customer_id=LOW_VALUE_CUSTOMER,
        now=RECENT_NOW,
    )

    assert result["issued"] is False
    assert result["status"] == "pending_confirmation"
    assert result["amount"] == 34.99


def test_first_call_only_proposes_and_does_not_write_anything(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)

    result = issue_refund(
        LOW_VALUE_ORDER, "unopened_or_unwanted", "changed my mind", state=state, customer_id=LOW_VALUE_CUSTOMER, now=RECENT_NOW
    )

    assert result["issued"] is False
    assert result["status"] == "pending_confirmation"
    with mock_db.get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM refunds").fetchone()[0] == 0
        status = conn.execute("SELECT status FROM orders WHERE order_id = ?", (LOW_VALUE_ORDER,)).fetchone()["status"]
    assert status == "Delivered"


def test_rejects_confirmation_attempted_in_the_same_turn(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)
    issue_refund(LOW_VALUE_ORDER, "unopened_or_unwanted", "changed my mind", state=state, customer_id=LOW_VALUE_CUSTOMER, now=RECENT_NOW)

    result = issue_refund(
        LOW_VALUE_ORDER, "unopened_or_unwanted", "changed my mind", state=state, customer_id=LOW_VALUE_CUSTOMER, now=RECENT_NOW
    )

    assert result["issued"] is False
    assert result["status"] == "pending_confirmation"


def test_confirms_in_a_later_turn_and_writes_refund_and_updates_order(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)
    issue_refund(LOW_VALUE_ORDER, "unopened_or_unwanted", "changed my mind", state=state, customer_id=LOW_VALUE_CUSTOMER, now=RECENT_NOW)

    state.turn = 2
    result = issue_refund(
        LOW_VALUE_ORDER, "unopened_or_unwanted", "changed my mind", state=state, customer_id=LOW_VALUE_CUSTOMER, now=RECENT_NOW
    )

    assert result["issued"] is True
    assert result["amount"] == 34.99
    with mock_db.get_connection() as conn:
        row = conn.execute("SELECT * FROM refunds WHERE order_id = ?", (LOW_VALUE_ORDER,)).fetchone()
        order_status = conn.execute(
            "SELECT status FROM orders WHERE order_id = ?", (LOW_VALUE_ORDER,)
        ).fetchone()["status"]
    assert row["amount"] == 34.99
    assert row["customer_id"] == LOW_VALUE_CUSTOMER
    assert order_status == "Refunded"


def test_rejects_double_refund_of_an_already_refunded_order(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)
    issue_refund(LOW_VALUE_ORDER, "unopened_or_unwanted", "changed my mind", state=state, customer_id=LOW_VALUE_CUSTOMER, now=RECENT_NOW)
    state.turn = 2
    issue_refund(LOW_VALUE_ORDER, "unopened_or_unwanted", "changed my mind", state=state, customer_id=LOW_VALUE_CUSTOMER, now=RECENT_NOW)

    state.turn = 3
    result = issue_refund(
        LOW_VALUE_ORDER, "unopened_or_unwanted", "changed my mind again", state=state, customer_id=LOW_VALUE_CUSTOMER, now=RECENT_NOW
    )

    assert result["issued"] is False
    assert result["error"] == "already_refunded"


def test_high_value_refund_escalates_immediately_with_no_confirmation_step(tmp_path, monkeypatch):
    """Phase 6 checkpoint: auto-escalate above the $ threshold — and per the
    plan's literal ordering, this happens INSTEAD of asking for
    confirmation, not after it."""
    _seed(tmp_path, monkeypatch)
    state = PendingActionGate(turn=1)

    result = issue_refund(
        HIGH_VALUE_ORDER,
        "unopened_or_unwanted",
        "changed my mind",
        state=state,
        customer_id=HIGH_VALUE_CUSTOMER,
        now=datetime(2026, 8, 5, 10, 0),  # noqa: DTZ001 — naive on purpose, matches refunds.py
    )

    assert result["amount"] > HIGH_VALUE_REFUND_THRESHOLD
    assert result["issued"] is False
    assert result["escalate"] is True
    assert "escalation_reason" in result
    assert state.pending is None  # no confirmation dance was ever started
    with mock_db.get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM refunds").fetchone()[0] == 0
