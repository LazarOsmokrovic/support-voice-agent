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
from data.mock_db import get_connection


def _classification(
    intent: str = "chitchat", sentiment: str = "neutral", policy_restricted: bool = False
) -> TurnClassification:
    return TurnClassification(intent=intent, sentiment=sentiment, policy_restricted=policy_restricted)


# --- EscalationTracker: deterministic triggers, no network involved ---


def test_explicit_human_request_escalates_immediately():
    tracker = EscalationTracker()
    signal = tracker.record_turn(_classification(intent="request_human"), [])
    assert signal is not None and signal.reason == "explicit request for a human"


def test_policy_restricted_topic_escalates_immediately():
    tracker = EscalationTracker()
    signal = tracker.record_turn(_classification(policy_restricted=True), [])
    assert signal is not None and signal.reason == "policy-restricted topic"


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

    signal = tracker.record_turn(_classification(), tool_calls)

    assert signal is not None and signal.reason == "high-value refund ($349.99) requires specialist approval"


def test_a_tool_escalate_flag_takes_priority_over_a_calm_classification():
    tracker = EscalationTracker()
    tool_calls = [{"name": "issue_refund", "input": {}, "output": {"escalate": True}}]

    signal = tracker.record_turn(_classification(sentiment="positive"), tool_calls)

    assert signal is not None and signal.reason == "a high-value action requires human approval"  # default reason, none was provided


def test_single_negative_turn_does_not_escalate():
    tracker = EscalationTracker()
    reason = tracker.record_turn(_classification(sentiment="negative"), [])
    assert reason is None


def test_two_consecutive_negative_turns_escalates():
    tracker = EscalationTracker()
    assert tracker.record_turn(_classification(sentiment="negative"), []) is None
    signal = tracker.record_turn(_classification(sentiment="negative"), [])
    assert signal is not None and signal.reason == "sustained negative sentiment across multiple turns"


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
    signal = tracker.record_turn(_classification(), failed)
    assert signal is not None and signal.reason == "repeated failed lookups"


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
    signal = tracker.record_turn(_classification(), failed)
    assert signal is not None and signal.reason == "repeated failed lookups"  # streak was NOT reset by the chitchat turn


def test_single_ungrounded_reply_does_not_escalate():
    tracker = EscalationTracker()
    assert tracker.record_turn(_classification(), [], ungrounded=True) is None


def test_two_consecutive_ungrounded_replies_escalates():
    tracker = EscalationTracker()
    assert tracker.record_turn(_classification(), [], ungrounded=True) is None
    signal = tracker.record_turn(_classification(), [], ungrounded=True)
    assert signal is not None and signal.reason == "repeated ungrounded replies"


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


def test_the_classifier_is_told_that_cancelling_is_not_a_complaint():
    """From a live call: the customer said "I changed my mind, I don't want it
    anymore, I want to cancel" and then "I can't accept it, so I want to
    cancel". Both turns were scored negative, two in a row tripped
    NEGATIVE_SENTIMENT_ESCALATION_THRESHOLD, and the call ended on an
    escalation — while the agent was answering both turns correctly.

    The bug was in what the prompt asked for. It asked for the customer's
    TONE, but the answer is used as "this customer is frustrated with US and
    needs a human". Those are different questions: wanting to undo a purchase
    is an ordinary transactional request, and someone can ask to cancel an
    order perfectly cheerfully.

    Asserted on the prompt text rather than on a classification, because
    classify_turn needs an API call and the whole suite runs offline. The real
    verification is a live call that cancels an order without escalating.
    """
    from agent.prompts import CLASSIFICATION_PROMPT

    prompt = CLASSIFICATION_PROMPT.lower()
    assert "cancel" in prompt, "the prompt must address cancellation explicitly"
    assert "not negative" in prompt or "is not negative" in prompt, (
        "the prompt must say outright that wanting to cancel is not negative sentiment"
    )
    assert "service" in prompt, (
        "sentiment must be scoped to how they feel about the SERVICE, not raw tone"
    )


def test_the_prompt_tells_the_agent_what_to_do_when_no_id_arrives():
    """Live: the agent asked for an order number, the caller said "give me a
    moment to check, please", and the agent replied as though a lookup had
    started.

    agent/prompts.py's thinking_phrase already suppresses the spoken filler
    for that turn, but the filler is only the first half — the MODEL's own
    reply has to handle it too, and nothing in the prompt told it how. Both
    halves are needed: one stops the wrong thing being said before the model
    answers, the other stops the model answering wrongly.
    """
    from agent.prompts import SYSTEM_PROMPT

    prompt = SYSTEM_PROMPT.lower()
    assert "take your time" in prompt, "the stalling case needs a concrete reply to give"
    assert "still looking for it" in prompt or "looking for it" in prompt
    assert "do not call a tool" in prompt, (
        "a caller who has not given a number yet must not trigger a lookup"
    )


# --- Phase 12: EscalationSignal and EscalationState ---


def test_a_mandatory_trigger_is_marked_mandatory():
    """An explicit request for a human is the customer's decision, not the
    agent's inference, so it opens a handover without asking permission."""
    tracker = escalation.EscalationTracker()
    signal = tracker.record_turn(
        escalation.TurnClassification(
            intent="request_human", sentiment="neutral", policy_restricted=False
        ),
        tool_calls=[],
    )
    assert signal is not None and signal.mandatory is True
    assert signal.reason == "explicit request for a human"


def test_inferred_triggers_are_only_suggestions():
    """Both of these hung up on customers who were fine: one had mis-dictated
    a digit and corrected themselves, the other was calmly cancelling an
    order. They are the AGENT's inference that it is failing — sometimes
    true, sometimes not. Inferences get offered, not imposed."""
    tracker = escalation.EscalationTracker()
    neutral = escalation.TurnClassification(
        intent="order_status", sentiment="neutral", policy_restricted=False
    )
    failed = [{"name": "get_order_status", "output": {"found": False}}]
    for _ in range(escalation.FAILED_LOOKUP_ESCALATION_THRESHOLD):
        signal = tracker.record_turn(neutral, failed)
    assert signal.reason == "repeated failed lookups"
    assert signal.mandatory is False
    assert signal.reason in escalation.SUGGESTED_REASONS

    tracker2 = escalation.EscalationTracker()
    upset = escalation.TurnClassification(
        intent="complaint", sentiment="negative", policy_restricted=False
    )
    for _ in range(escalation.NEGATIVE_SENTIMENT_ESCALATION_THRESHOLD):
        signal2 = tracker2.record_turn(upset, [])
    assert signal2.mandatory is False


def test_the_sets_are_disjoint_and_are_not_the_mechanism():
    """MANDATORY_REASONS / SUGGESTED_REASONS are for tests, eval scoring and
    readers — never for deriving `mandatory` (D-12)."""
    assert escalation.MANDATORY_REASONS & escalation.SUGGESTED_REASONS == set()


def test_a_tool_signalled_escalation_is_mandatory_even_though_no_set_contains_it():
    """agent/tools/refunds.py:185 builds its reason with an f-string carrying
    the amount, so the string is different for every refund and can never be a
    member of a static set. An implementation deriving mandatory as
    `reason in MANDATORY_REASONS` therefore downgrades "a specialist must
    approve this $200 refund" to "would you like a human?" — and breaks
    eval/scenarios.py:314.

    A tool asking for a specialist is a rule, not an inference. Asserting on a
    dynamic reason is the only way this test can tell the two implementations
    apart."""
    tracker = escalation.EscalationTracker()
    calm = escalation.TurnClassification(
        intent="refund_or_return", sentiment="neutral", policy_restricted=False
    )
    signalled = [
        {
            "name": "issue_refund",
            "output": {
                "escalate": True,
                "escalation_reason": "high-value refund ($249.99) requires specialist approval",
            },
        }
    ]

    signal = tracker.record_turn(calm, signalled)

    assert signal is not None
    assert signal.mandatory is True, "a specialist requirement is a rule, not an offer"
    assert signal.reason not in escalation.MANDATORY_REASONS, (
        "and it is mandatory despite no set containing it — proving the set is not the mechanism"
    )


def test_resetting_a_streak_gives_the_customer_a_clean_run():
    tracker = escalation.EscalationTracker()
    neutral = escalation.TurnClassification(
        intent="order_status", sentiment="neutral", policy_restricted=False
    )
    failed = [{"name": "get_order_status", "output": {"found": False}}]
    for _ in range(escalation.FAILED_LOOKUP_ESCALATION_THRESHOLD):
        tracker.record_turn(neutral, failed)

    tracker.reset_streak("repeated failed lookups")

    assert tracker.consecutive_failed_lookups == 0
    assert tracker.record_turn(neutral, failed) is None


def test_a_turn_with_no_tool_calls_does_not_advance_the_lookup_counter():
    """Turn 10 of the live call escalated while the agent was merely asked to
    repeat a number back — no lookup happened at all. The bug was that the
    threshold check sat OUTSIDE the `if outcomes:` block, so it re-fired on
    every subsequent turn regardless of whether a lookup was even attempted."""
    tracker = escalation.EscalationTracker()
    neutral = escalation.TurnClassification(
        intent="order_status", sentiment="neutral", policy_restricted=False
    )
    failed = [{"name": "get_order_status", "output": {"found": False}}]

    # Drive the streak to the threshold — this call SHOULD return a signal.
    for _ in range(escalation.FAILED_LOOKUP_ESCALATION_THRESHOLD):
        signal = tracker.record_turn(neutral, failed)
    assert signal is not None and signal.reason == "repeated failed lookups"

    before = tracker.consecutive_failed_lookups
    # A turn with NO tool calls at all — nothing was looked up.
    signal_again = tracker.record_turn(neutral, [])

    assert tracker.consecutive_failed_lookups == before, "the counter must not move"
    assert signal_again is None, (
        "an empty-outcomes turn must not re-fire the escalation signal — "
        "this is the actual bug: the old code re-checked the threshold every "
        "turn regardless of whether a lookup was attempted"
    )


def test_the_state_machine_opens_amends_and_resolves():
    state = escalation.EscalationState()
    assert state.status == escalation.STATUS_NONE and state.is_open is False

    state.open("explicit request for a human")
    assert state.status == escalation.STATUS_OPEN
    assert state.items == ["explicit request for a human"]

    state.amend("policy-restricted topic")
    assert state.status == escalation.STATUS_OPEN, "an amendment never reopens"
    assert state.items == ["explicit request for a human", "policy-restricted topic"]

    state.record_resolution(escalation.RESOLUTION_CALLBACK, callback_time="2026-09-14T09:00:00")
    assert state.status == escalation.STATUS_RESOLVED and state.is_open is False


def test_amending_a_resolved_handover_keeps_it_resolved():
    state = escalation.EscalationState()
    state.open("explicit request for a human")
    state.record_resolution(escalation.RESOLUTION_SELF)
    state.amend("high-value refund requires approval")
    assert state.status == escalation.STATUS_RESOLVED
    assert len(state.items) == 2


def test_a_repeated_trigger_does_not_duplicate_an_item():
    """One tripped trigger firing every turn produced escalation rows 7, 8
    and 9 for one problem in a live call."""
    state = escalation.EscalationState()
    state.open("explicit request for a human")
    state.amend("explicit request for a human")
    assert state.items == ["explicit request for a human"]


def test_the_refusal_budget_is_spent_once_per_turn():
    """agent/core.py:91 allows 8 tool iterations, so a model that reads the
    refusal and simply retries can call end_conversation three times inside
    ONE agent.send(). A call-counted budget is exhausted with no customer
    utterance in between, and the call ends on the trigger turn with the
    handover open — which is the bug this phase exists to fix."""
    state = escalation.EscalationState()
    state.open("explicit request for a human")

    assert state.consume_refusal(turn=4) is True
    assert state.consume_refusal(turn=4) is True, "a repeat within one turn still refuses"
    assert state.refusals == 1, "but it must not spend budget"


# --- Task 2: open_escalation / resolve_escalation split ---
#
# Every test below repoints the database via the autouse fixture rather than
# writing to the developer's real data/mock_data.db — the same live-call
# evidence Task 1 Step 1 backs up before this phase starts.


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    """Without this, these tests insert escalations rows into the developer's
    real data/mock_data.db — the same live-call evidence Task 1 Step 1 goes out
    of its way to back up before this phase starts."""
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "phase12.db")
    mock_db.reset_and_seed()


CUSTOMER_ID = mock_db.CUSTOMERS[0][0]  # never hard-coded (Global Constraints)


def _recording_notifier(sink: list[dict]):
    async def _notify(packet, **kwargs):
        sink.append(dict(packet))  # copy: resolve_escalation mutates in place
        return True

    return _notify


async def _stub_infer_fields(customer_id, messages, client=None):
    """Keeps these offline; the inference is covered separately."""
    # Real strings on every field. HandoffFields declares verified_account_info
    # and actions_taken as `str`, not `str | None`
    # (agent/tools/escalation.py:209-214), so None raises a pydantic
    # ValidationError at helper-call time. Do NOT widen the model to make this
    # helper convenient — that changes what every escalation row and every
    # outbound webhook packet carries, for a test's sake.
    return escalation.HandoffFields(
        customer_intent="wants a person",
        conversation_summary="asked for a human",
        verified_account_info="verified by order ID",
        actions_taken="looked up the order",
        sentiment="neutral",
    )


@pytest.mark.asyncio
async def test_opening_a_handover_writes_a_row_but_tells_nobody(monkeypatch):
    """Until an outcome is known there is nothing useful to tell a human. "A
    customer needs help, we don't know what about or when to ring" is a
    message they have to chase. The row is still written immediately, so a
    dropped call leaves a record."""
    sent = []
    monkeypatch.setattr(escalation, "notify_escalation", _recording_notifier(sent))
    monkeypatch.setattr(escalation, "_infer_handoff_fields", _stub_infer_fields)

    packet = await escalation.open_escalation(
        CUSTOMER_ID, [{"role": "user", "content": "human please"}],
        "explicit request for a human",
    )

    assert packet["escalation_id"] is not None
    assert sent == []
    with get_connection() as conn:
        row = conn.execute(
            "SELECT resolution FROM escalations WHERE escalation_id = ?",
            (packet["escalation_id"],),
        ).fetchone()
    assert row["resolution"] is None


@pytest.mark.asyncio
async def test_resolving_notifies_once_and_puts_the_time_where_a_human_reads_it(monkeypatch):
    """"Notifies twice" is the failure a reader cannot see, so this counts.

    And the agreed time is folded into `reason` deliberately:
    transport/telephony.py's render_whisper reads escalation_id, reason,
    customer_intent, verified_account_info, actions_taken and sentiment — it
    never reads `items` or `callback_time`. Writing the time only to those
    fields would mean the human hearing the whisper never learns it."""
    sent = []
    monkeypatch.setattr(escalation, "notify_escalation", _recording_notifier(sent))
    monkeypatch.setattr(escalation, "_infer_handoff_fields", _stub_infer_fields)
    packet = await escalation.open_escalation(
        CUSTOMER_ID, [{"role": "user", "content": "human please"}],
        "explicit request for a human",
    )
    slot = "2026-09-14T09:00:00"

    delivered = await escalation.resolve_escalation(
        packet,
        items=["explicit request for a human", "refund eligibility question"],
        resolution=escalation.RESOLUTION_CALLBACK,
        callback_time=slot,
    )

    assert delivered is True
    assert len(sent) == 1, f"exactly one notification per handover, got {len(sent)}"
    assert sent[0]["resolution"] == escalation.RESOLUTION_CALLBACK
    assert sent[0]["callback_time"] == slot
    assert sent[0]["items"] == ["explicit request for a human", "refund eligibility question"]
    assert "refund eligibility question" in sent[0]["reason"], (
        "every item must reach `reason`, the only field the whisper renders"
    )
    assert slot in sent[0]["reason"], (
        "the agreed callback time must reach `reason` too — render_whisper never reads callback_time"
    )
    assert packet["resolution"] == escalation.RESOLUTION_CALLBACK, (
        "the caller's packet is what the transport whispers — it must be updated in place"
    )


@pytest.mark.asyncio
async def test_a_notification_failure_still_leaves_the_handover_resolved(monkeypatch):
    """Delivery has never been allowed to affect persistence (Phase 11)."""
    async def _explode(packet, **kwargs):
        raise RuntimeError("webhook down")

    monkeypatch.setattr(escalation, "notify_escalation", _explode)
    monkeypatch.setattr(escalation, "_infer_handoff_fields", _stub_infer_fields)
    packet = await escalation.open_escalation(
        CUSTOMER_ID, [{"role": "user", "content": "human please"}],
        "explicit request for a human",
    )

    delivered = await escalation.resolve_escalation(
        packet, items=["explicit request for a human"],
        resolution=escalation.RESOLUTION_SELF,
    )

    assert delivered is False
    with get_connection() as conn:
        row = conn.execute(
            "SELECT resolution, resolved_at FROM escalations WHERE escalation_id = ?",
            (packet["escalation_id"],),
        ).fetchone()
    assert row["resolution"] == escalation.RESOLUTION_SELF
    assert row["resolved_at"] is not None


@pytest.mark.asyncio
async def test_create_handoff_packet_still_opens_and_resolves_together(monkeypatch):
    """transport/pipecat_processors.py:227 (DTMF zero) calls this and must not
    change — CLAUDE.md rule 5. Pressing zero IS the resolution: the caller is
    being put through right now, so there is no process to work through."""
    sent = []
    monkeypatch.setattr(escalation, "notify_escalation", _recording_notifier(sent))
    monkeypatch.setattr(escalation, "_infer_handoff_fields", _stub_infer_fields)

    packet = await escalation.create_handoff_packet(
        CUSTOMER_ID, [{"role": "user", "content": "0"}], "caller pressed 0 for a human"
    )

    assert packet["escalation_id"] is not None
    assert "callback_time" in packet, "the existing contract includes a callback time"
    assert packet["resolution"] == escalation.RESOLUTION_TRANSFER
    assert len(sent) == 1
    assert packet["reason"] == "caller pressed 0 for a human", (
        "an immediate transfer has no future outcome to fold in — the existing "
        "create_handoff_packet contract (test_create_handoff_packet_infers_fields_and_logs) "
        "asserts `reason` passes through unchanged"
    )
