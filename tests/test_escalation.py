"""Phase 4 checkpoint: escalation fires neither too eagerly nor too late.

EscalationTracker's rules are deterministic and tested directly here, with
no network, against fabricated classifications — this is the precise,
scriptable half of the checkpoint. classify_turn's actual judgment quality
(does it correctly read a real message's intent/sentiment) is checked
separately with live calls, gated on a real ANTHROPIC_API_KEY. The
full-pipeline scripted-conversation checkpoint (real Claude calls across
several turns of escalating frustration) lives in tests/test_text_cli.py,
alongside the project's other full-loop tests.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.tools import escalation
from agent.tools.escalation import (
    EscalationTracker,
    HandoffFields,
    TurnClassification,
    create_handoff_packet,
    log_escalation,
)
from data import mock_db


def _classification(
    intent: str = "chitchat", sentiment: str = "neutral", policy_restricted: bool = False
) -> TurnClassification:
    return TurnClassification(intent=intent, sentiment=sentiment, policy_restricted=policy_restricted)


# --- EscalationTracker: deterministic triggers, no network involved ---


def test_explicit_human_request_escalates_immediately():
    tracker = EscalationTracker()
    reason = tracker.record_turn(_classification(intent="request_human"), [])
    assert reason == "explicit request for a human"


def test_policy_restricted_topic_escalates_immediately():
    tracker = EscalationTracker()
    reason = tracker.record_turn(_classification(policy_restricted=True), [])
    assert reason == "policy-restricted topic"


def test_a_tool_signaling_escalate_fires_immediately():
    """Phase 6: issue_refund (or any future tool) can trigger escalation
    directly by returning escalate: true — the tracker doesn't need to
    know anything refund-specific to honor it."""
    tracker = EscalationTracker()
    tool_calls = [
        {
            "name": "issue_refund",
            "input": {},
            "output": {"escalate": True, "escalation_reason": "high-value refund ($349.99) requires specialist approval"},
        }
    ]

    reason = tracker.record_turn(_classification(), tool_calls)

    assert reason == "high-value refund ($349.99) requires specialist approval"


def test_a_tool_escalate_flag_takes_priority_over_a_calm_classification():
    tracker = EscalationTracker()
    tool_calls = [{"name": "issue_refund", "input": {}, "output": {"escalate": True}}]

    reason = tracker.record_turn(_classification(sentiment="positive"), tool_calls)

    assert reason == "a high-value action requires human approval"  # default reason, none was provided


def test_single_negative_turn_does_not_escalate():
    tracker = EscalationTracker()
    reason = tracker.record_turn(_classification(sentiment="negative"), [])
    assert reason is None


def test_two_consecutive_negative_turns_escalates():
    tracker = EscalationTracker()
    assert tracker.record_turn(_classification(sentiment="negative"), []) is None
    reason = tracker.record_turn(_classification(sentiment="negative"), [])
    assert reason == "sustained negative sentiment across multiple turns"


def test_a_calm_turn_resets_the_negative_streak():
    tracker = EscalationTracker()
    assert tracker.record_turn(_classification(sentiment="negative"), []) is None
    assert tracker.record_turn(_classification(sentiment="neutral"), []) is None
    reason = tracker.record_turn(_classification(sentiment="negative"), [])
    assert reason is None  # streak was reset; this is only the 1st negative again


def test_single_failed_lookup_does_not_escalate():
    tracker = EscalationTracker()
    tool_calls = [{"name": "get_order_status", "input": {}, "output": {"found": False}}]
    reason = tracker.record_turn(_classification(), tool_calls)
    assert reason is None


def test_two_consecutive_failed_lookups_escalates():
    tracker = EscalationTracker()
    failed = [{"name": "get_order_status", "input": {}, "output": {"found": False}}]
    assert tracker.record_turn(_classification(), failed) is None
    reason = tracker.record_turn(_classification(), failed)
    assert reason == "repeated failed lookups"


def test_a_successful_lookup_resets_the_failure_streak():
    tracker = EscalationTracker()
    failed = [{"name": "get_order_status", "input": {}, "output": {"found": False}}]
    succeeded = [{"name": "get_order_status", "input": {}, "output": {"found": True}}]
    assert tracker.record_turn(_classification(), failed) is None
    assert tracker.record_turn(_classification(), succeeded) is None
    reason = tracker.record_turn(_classification(), failed)
    assert reason is None  # streak was reset by the successful lookup


def test_turns_without_lookups_do_not_affect_the_failure_streak():
    tracker = EscalationTracker()
    failed = [{"name": "get_order_status", "input": {}, "output": {"found": False}}]
    assert tracker.record_turn(_classification(), failed) is None
    assert tracker.record_turn(_classification(), []) is None  # chitchat turn, no lookup at all
    reason = tracker.record_turn(_classification(), failed)
    assert reason == "repeated failed lookups"  # streak was NOT reset by the chitchat turn


def test_single_ungrounded_reply_does_not_escalate():
    tracker = EscalationTracker()
    assert tracker.record_turn(_classification(), [], ungrounded=True) is None


def test_two_consecutive_ungrounded_replies_escalates():
    tracker = EscalationTracker()
    assert tracker.record_turn(_classification(), [], ungrounded=True) is None
    assert tracker.record_turn(_classification(), [], ungrounded=True) == "repeated ungrounded replies"


def test_a_grounded_reply_resets_the_ungrounded_streak():
    tracker = EscalationTracker()
    assert tracker.record_turn(_classification(), [], ungrounded=True) is None
    assert tracker.record_turn(_classification(), [], ungrounded=False) is None
    assert tracker.record_turn(_classification(), [], ungrounded=True) is None


def test_ungrounded_defaults_to_false_for_existing_callers():
    """Every pre-Phase-10a caller passes two arguments; that must still mean
    'this reply was fine'."""
    tracker = EscalationTracker()
    assert tracker.record_turn(_classification(), []) is None
    assert tracker.record_turn(_classification(), []) is None


# --- log_escalation / create_handoff_packet ---


def test_log_escalation_writes_row_to_escalations_table(tmp_path, monkeypatch):
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_escalation.db")
    mock_db.reset_and_seed()
    customer_id = mock_db.CUSTOMERS[0][0]
    fields = HandoffFields(
        customer_intent="Wanted a refund",
        conversation_summary="Customer asked about a refund for a late order.",
        verified_account_info=f"Customer ID {customer_id}",
        actions_taken="None yet",
        sentiment="negative",
    )

    escalation_id = log_escalation(
        customer_id, "explicit request for a human", fields, created_at="2026-08-24T00:00:00+00:00"
    )

    with mock_db.get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM escalations WHERE escalation_id = ?", (escalation_id,)
        ).fetchone()

    assert row["customer_id"] == customer_id
    assert row["reason"] == "explicit request for a human"
    assert row["customer_intent"] == fields.customer_intent
    assert row["sentiment"] == "negative"


@pytest.mark.asyncio
async def test_create_handoff_packet_infers_fields_and_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_handoff.db")
    mock_db.reset_and_seed()
    customer_id = mock_db.CUSTOMERS[0][0]

    fake_fields = HandoffFields(
        customer_intent="Wanted a refund",
        conversation_summary="Asked about a refund.",
        verified_account_info=f"Customer ID {customer_id}",
        actions_taken="None yet",
        sentiment="negative",
    )
    fake_response = MagicMock()
    fake_response.parsed_output = fake_fields
    fake_client = MagicMock()
    fake_client.messages.parse = AsyncMock(return_value=fake_response)

    packet = await create_handoff_packet(
        customer_id,
        [{"role": "user", "content": "I want a refund"}],
        "explicit request for a human",
        client=fake_client,
    )

    assert packet["reason"] == "explicit request for a human"
    assert packet["customer_intent"] == "Wanted a refund"
    assert "escalation_id" in packet
    with mock_db.get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM escalations WHERE escalation_id = ?", (packet["escalation_id"],)
        ).fetchone()
    assert row is not None


@pytest.mark.asyncio
async def test_create_handoff_packet_marks_notified_true_on_successful_delivery(tmp_path, monkeypatch):
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_handoff_notify.db")
    mock_db.reset_and_seed()
    customer_id = mock_db.CUSTOMERS[0][0]

    fake_fields = HandoffFields(
        customer_intent="Wanted a refund",
        conversation_summary="Asked about a refund.",
        verified_account_info=f"Customer ID {customer_id}",
        actions_taken="None yet",
        sentiment="negative",
    )
    fake_response = MagicMock()
    fake_response.parsed_output = fake_fields
    fake_client = MagicMock()
    fake_client.messages.parse = AsyncMock(return_value=fake_response)
    monkeypatch.setattr(escalation, "notify_escalation", AsyncMock(return_value=True))

    packet = await create_handoff_packet(
        customer_id,
        [{"role": "user", "content": "I want a refund"}],
        "explicit request for a human",
        client=fake_client,
    )

    with mock_db.get_connection() as conn:
        row = conn.execute(
            "SELECT notified, notified_at FROM escalations WHERE escalation_id = ?", (packet["escalation_id"],)
        ).fetchone()
    assert row["notified"] == 1
    assert row["notified_at"] is not None


@pytest.mark.asyncio
async def test_create_handoff_packet_still_returns_and_logs_when_notify_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_handoff_notify_fail.db")
    mock_db.reset_and_seed()
    customer_id = mock_db.CUSTOMERS[0][0]

    fake_fields = HandoffFields(
        customer_intent="Wanted a refund",
        conversation_summary="Asked about a refund.",
        verified_account_info=f"Customer ID {customer_id}",
        actions_taken="None yet",
        sentiment="negative",
    )
    fake_response = MagicMock()
    fake_response.parsed_output = fake_fields
    fake_client = MagicMock()
    fake_client.messages.parse = AsyncMock(return_value=fake_response)
    monkeypatch.setattr(
        escalation, "notify_escalation", AsyncMock(side_effect=RuntimeError("webhook host unreachable"))
    )

    packet = await create_handoff_packet(
        customer_id,
        [{"role": "user", "content": "I want a refund"}],
        "explicit request for a human",
        client=fake_client,
    )

    assert packet["reason"] == "explicit request for a human"
    with mock_db.get_connection() as conn:
        row = conn.execute(
            "SELECT notified, notified_at FROM escalations WHERE escalation_id = ?", (packet["escalation_id"],)
        ).fetchone()
    assert row["notified"] == 0
    assert row["notified_at"] is not None  # mark_notified still ran, just with delivered=False


@pytest.mark.asyncio
async def test_create_handoff_packet_still_returns_and_logs_when_mark_notified_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_handoff_marknotify_fail.db")
    mock_db.reset_and_seed()
    customer_id = mock_db.CUSTOMERS[0][0]

    fake_fields = HandoffFields(
        customer_intent="Wanted a refund",
        conversation_summary="Asked about a refund.",
        verified_account_info=f"Customer ID {customer_id}",
        actions_taken="None yet",
        sentiment="negative",
    )
    fake_response = MagicMock()
    fake_response.parsed_output = fake_fields
    fake_client = MagicMock()
    fake_client.messages.parse = AsyncMock(return_value=fake_response)
    monkeypatch.setattr(escalation, "notify_escalation", AsyncMock(return_value=True))
    monkeypatch.setattr(
        escalation, "mark_notified", MagicMock(side_effect=RuntimeError("database is locked"))
    )

    packet = await create_handoff_packet(
        customer_id,
        [{"role": "user", "content": "I want a refund"}],
        "explicit request for a human",
        client=fake_client,
    )

    assert packet["reason"] == "explicit request for a human"
    assert "escalation_id" in packet
    # The escalation row from log_escalation must survive a failing
    # mark_notified intact — that's the whole point: an already-persisted
    # escalation must never be discarded because the *follow-up* status
    # update failed.
    with mock_db.get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM escalations WHERE escalation_id = ?", (packet["escalation_id"],)
        ).fetchone()
    assert row is not None
    assert row["customer_id"] == customer_id


@pytest.mark.asyncio
async def test_create_handoff_packet_redacts_pii_in_packet_and_db(tmp_path, monkeypatch):
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_handoff_redaction.db")
    mock_db.reset_and_seed()
    customer_id = mock_db.CUSTOMERS[0][0]
    order_id = mock_db.ORDERS[0][0]

    fake_fields = HandoffFields(
        customer_intent=f"Refund for order {order_id}, contact jane.doe@example.com",
        conversation_summary="Customer called from 555-123-4567 about a refund.",
        verified_account_info=f"Customer ID {customer_id}",
        actions_taken="Looked up the order.",
        sentiment="negative",
    )
    fake_response = MagicMock()
    fake_response.parsed_output = fake_fields
    fake_client = MagicMock()
    fake_client.messages.parse = AsyncMock(return_value=fake_response)
    monkeypatch.setattr(escalation, "notify_escalation", AsyncMock(return_value=True))

    packet = await create_handoff_packet(
        customer_id, [{"role": "user", "content": "refund please"}], "explicit request for a human", client=fake_client
    )

    # PII gone from both the returned packet and the persisted row...
    assert "jane.doe@example.com" not in packet["customer_intent"]
    assert "555-123-4567" not in packet["conversation_summary"]
    # ...but the order ID, which is not PII and is the most useful thing a
    # human taking this handoff can be given, survives intact.
    assert order_id in packet["customer_intent"]

    with mock_db.get_connection() as conn:
        row = conn.execute(
            "SELECT customer_intent, conversation_summary FROM escalations WHERE escalation_id = ?",
            (packet["escalation_id"],),
        ).fetchone()
    assert "jane.doe@example.com" not in row["customer_intent"]
    assert "555-123-4567" not in row["conversation_summary"]
    assert order_id in row["customer_intent"]


# The three live classify_turn checks that used to sit here moved to
# eval/scenarios.py in Phase 10c: triage_explicit_human_request,
# triage_sustained_frustration and triage_calm_conversation_never_escalates
# each exercise the same classification PLUS the tracker PLUS the handoff,
# so keeping these was the duplicate harness that phase exists to remove.
# Everything above stays: EscalationTracker's triggers, log_escalation and
# create_handoff_packet's redaction are deterministic and model-free, which
# makes them faster and more precise than any scenario could be.
