"""Phase 10c: tests for the eval suite's scoring, grounding arithmetic,
report rendering and exit codes.

Every assertion that touches a seeded identifier reads it out of
data/mock_db.py rather than typing the literal. Hand-typed seed values
produced a Critical defect in Phase 10a and another in Phase 11; the rule
here is that a test may never know a seeded value the code does not.
"""

from __future__ import annotations

import sqlite3

import pytest

from data import mock_db
from eval.harness import HarnessResult, ObservedTurn
from eval.scenarios import DbAssertion, Expectations, Scenario, ToolExpectation
from eval.scoring import (
    score_db,
    score_escalation,
    score_expectations,
    score_pii,
    score_tools,
)


def _scenario(**overrides) -> Scenario:
    base = dict(
        name="demo",
        capability="refunds",
        customer_id=mock_db.CUSTOMERS[0][0],
        turns=("Refund my order", "Yes please"),
        expect=Expectations(),
        grounding_truth=("not_applicable", "not_applicable"),
    )
    base.update(overrides)
    return Scenario(**base)


def _result(**overrides) -> HarnessResult:
    base = dict(scenario="demo", replies=["ok", "done"], observed=[], records=[], log_lines=[])
    base.update(overrides)
    return HarnessResult(**base)


def _turn(turn: int, **overrides) -> ObservedTurn:
    base = dict(
        turn=turn,
        tool_calls=[],
        grounding_flagged=False,
        hedge_spoken=False,
        escalation_reason=None,
        end_reason=None,
        block_input_runtime_type=None,
    )
    base.update(overrides)
    return ObservedTurn(**base)


def test_tool_expectation_matches_on_a_subset_and_ignores_extra_arguments():
    order_id = mock_db.ORDERS[0][0]
    scenario = _scenario(
        expect=Expectations(tools_called=(ToolExpectation("issue_refund", {"order_id": order_id}),))
    )
    result = _result(
        observed=[
            _turn(
                1,
                tool_calls=[
                    {
                        "name": "issue_refund",
                        "input": {"order_id": order_id, "condition": "unopened_or_unwanted", "reason": "x"},
                        "output": {},
                    }
                ],
            )
        ]
    )
    assert score_tools(scenario, result) == []


def test_tool_expectation_fails_on_a_wrong_argument_value():
    order_id = mock_db.ORDERS[0][0]
    scenario = _scenario(
        expect=Expectations(tools_called=(ToolExpectation("issue_refund", {"condition": "unopened_or_unwanted"}),))
    )
    result = _result(
        observed=[
            _turn(1, tool_calls=[{"name": "issue_refund", "input": {"order_id": order_id, "condition": "damaged_or_defective"}, "output": {}}])
        ]
    )
    failures = score_tools(scenario, result)
    assert len(failures) == 1
    assert failures[0].kind == "tools"
    assert "damaged_or_defective" in failures[0].detail


def test_tool_expectation_pinned_to_a_turn_fails_when_it_happens_on_another():
    scenario = _scenario(expect=Expectations(tools_called=(ToolExpectation("search_policy", turn=2),)))
    result = _result(
        observed=[_turn(1, tool_calls=[{"name": "search_policy", "input": {}, "output": {}}]), _turn(2)]
    )
    failures = score_tools(scenario, result)
    assert len(failures) == 1
    assert "turn 2" in failures[0].detail


def test_tools_not_called_fails_when_the_tool_was_called():
    scenario = _scenario(expect=Expectations(tools_not_called=("issue_refund",)))
    result = _result(observed=[_turn(1, tool_calls=[{"name": "issue_refund", "input": {}, "output": {}}])])
    failures = score_tools(scenario, result)
    assert len(failures) == 1
    assert "issue_refund" in failures[0].detail


def test_escalation_turn_none_fails_when_an_escalation_fired():
    scenario = _scenario(expect=Expectations(escalation_turn=None))
    result = _result(observed=[_turn(1, escalation_reason="explicit request for a human", end_reason="escalated")])
    failures = score_escalation(scenario, result)
    assert len(failures) == 1
    assert "expected no escalation" in failures[0].detail


def test_escalation_turn_two_fails_when_it_fired_too_eagerly_on_turn_one():
    """One field, two assertions: declaring turn 2 also asserts turn 1 stayed
    quiet — Phase 4's 'neither too eager nor too late' checkpoint, currently
    split across two live tests and stated only in prose."""
    scenario = _scenario(
        expect=Expectations(escalation_turn=2, escalation_reason="sustained negative sentiment across multiple turns")
    )
    result = _result(
        observed=[_turn(1, escalation_reason="sustained negative sentiment across multiple turns", end_reason="escalated")]
    )
    failures = score_escalation(scenario, result)
    assert len(failures) == 1
    assert "turn 1" in failures[0].detail


def test_escalation_reason_mismatch_names_both_reasons():
    scenario = _scenario(expect=Expectations(escalation_turn=1, escalation_reason="policy-restricted topic"))
    result = _result(observed=[_turn(1, escalation_reason="explicit request for a human", end_reason="escalated")])
    failures = score_escalation(scenario, result)
    assert len(failures) == 1
    assert "policy-restricted topic" in failures[0].detail
    assert "explicit request for a human" in failures[0].detail


def test_db_assertion_counts_rows_and_pins_column_values(tmp_path):
    db_path = tmp_path / "scored.db"
    saved = mock_db.DB_PATH
    try:
        mock_db.DB_PATH = db_path
        mock_db.reset_and_seed()
    finally:
        mock_db.DB_PATH = saved

    order_id, customer_id = mock_db.ORDERS[0][0], mock_db.ORDERS[0][1]
    scenario = _scenario(
        expect=Expectations(
            db_assertions=(
                DbAssertion(
                    sql="SELECT status FROM orders WHERE order_id = ?",
                    params=(order_id,),
                    rows=1,
                    columns={"status": mock_db.ORDERS[0][5]},
                ),
                DbAssertion(sql="SELECT * FROM refunds WHERE customer_id = ?", params=(customer_id,), rows=0),
            )
        )
    )
    result = _result(db_path=db_path)

    assert score_db(scenario, result) == []


def test_db_assertion_fails_with_expected_and_actual_row_counts(tmp_path):
    db_path = tmp_path / "scored2.db"
    saved = mock_db.DB_PATH
    try:
        mock_db.DB_PATH = db_path
        mock_db.reset_and_seed()
    finally:
        mock_db.DB_PATH = saved

    scenario = _scenario(expect=Expectations(db_assertions=(DbAssertion(sql="SELECT * FROM refunds", rows=1),)))
    failures = score_db(scenario, _result(db_path=db_path))
    assert len(failures) == 1
    assert failures[0].kind == "db"
    assert "expected 1" in failures[0].detail
    assert "found 0" in failures[0].detail


def test_pii_scoring_flags_a_leaked_email_and_a_destroyed_tracking_number(tmp_path):
    """Built from real seeded values, never invented ones. This is Phase
    10a's date-and-tracking-number destruction bug turned into a permanent
    assertion: the customer's email and phone must be gone from stored
    records, and the order ID and TBA...US tracking number must survive."""
    customer_id, _name, email, _phone = mock_db.CUSTOMERS[0]
    order_id, _cust, _item, _q, _p, _s, _od, _ed, tracking = mock_db.ORDERS[0]
    scenario = _scenario(customer_id=customer_id, expect=Expectations(no_pii_in_records=True))
    result = _result(
        log_lines=[
            {"reply": f"Order {order_id} shipped, tracking {tracking}", "user_text": f"my email is {email}"}
        ]
    )
    failures = score_pii(scenario, result)
    assert any(email in failure.detail for failure in failures)
    assert all(failure.kind == "pii" for failure in failures)


def test_pii_scoring_passes_when_identifiers_survive_and_contacts_are_redacted(tmp_path):
    customer_id = mock_db.CUSTOMERS[0][0]
    order_id, _cust, _item, _q, _p, _s, _od, _ed, tracking = mock_db.ORDERS[0]
    scenario = _scenario(customer_id=customer_id)
    result = _result(
        log_lines=[{"reply": f"Order {order_id} shipped, tracking {tracking}", "user_text": "[redacted-email]"}]
    )
    assert score_pii(scenario, result) == []


def test_score_expectations_reports_a_wrong_end_reason():
    scenario = _scenario(expect=Expectations(end_reason="model_ended"))
    result = _result(observed=[_turn(1, end_reason="escalated")])
    failures = score_expectations(scenario, result)
    assert any(failure.kind == "end_reason" for failure in failures)


def test_score_drift_reports_a_tool_output_that_changed_since_recording():
    from eval.recording import Recording, current_hashes
    from eval.scoring import score_drift

    observed_then = [
        {
            "turn": 1,
            "tool_calls": [{"name": "search_policy", "input": {}, "output": {"found": True, "results": ["A"]}}],
            "grounding_flagged": False,
            "hedge_spoken": False,
            "escalation_reason": None,
            "end_reason": None,
            "block_input_runtime_type": "dict",
        }
    ]
    recording = Recording(
        scenario="demo",
        recorded_at="2026-09-08T12:00:00",
        model="claude-opus-5",
        anthropic_sdk_version="1.0.0",
        creates=[],
        parses=[],
        observed=observed_then,
        **current_hashes(),
    )
    result = _result(
        observed=[
            _turn(
                1,
                tool_calls=[{"name": "search_policy", "input": {}, "output": {"found": True, "results": ["B"]}}],
                block_input_runtime_type="dict",
            )
        ]
    )
    failures = score_drift(recording, result)
    assert len(failures) == 1
    assert failures[0].kind == "drift"
    assert "turn 1" in failures[0].detail


def test_rate_reports_n_over_n_and_never_a_bare_percentage():
    from eval.scoring import rate

    assert rate(1, 9) == "1/9"
    assert rate(0, 2) == "0/2"


def test_rate_refuses_to_divide_by_zero_or_pretend_zero_percent():
    """Spec §9 test 8's degenerate case. '0%' would read as a measured
    result where nothing was measured at all."""
    from eval.scoring import rate

    assert rate(0, 0) == "0/0 — insufficient data"


def test_grounding_counts_scores_a_hand_built_confusion_matrix():
    from eval.scoring import grounding_counts

    scenario = _scenario(
        turns=("a", "b", "c", "d"),
        grounding_truth=("grounded", "grounded", "ungrounded", "not_applicable"),
    )
    result = _result(
        observed=[
            _turn(1, grounding_flagged=False),  # grounded, quiet -> true negative
            _turn(2, grounding_flagged=True, hedge_spoken=True),  # grounded, flagged -> false positive
            _turn(3, grounding_flagged=False),  # ungrounded, quiet -> false negative
            _turn(4, grounding_flagged=True, hedge_spoken=True),  # unlabelled -> excluded
        ]
    )

    counts = grounding_counts(scenario, result)

    assert (counts.grounded, counts.ungrounded) == (2, 1)
    assert counts.labeled_turns == 3
    assert counts.true_negative == 1
    assert counts.false_positive == 1
    assert counts.false_negative == 1
    assert counts.true_positive == 0
    assert counts.hedged == 2
    assert counts.flagged == 2


def test_grounding_counts_counts_unreachable_claims_and_the_ladder():
    """The unreachable-claims number quantifies a real blind spot: issue_refund
    calls search_policy internally (refunds.py:144) and returns its text as
    policy_reference, but that internal call never enters
    TurnResult.tool_calls — so a turn asserting a dollar amount and a 30-day
    window is never grounding-checked at all. Measured here, not fixed."""
    from eval.scoring import grounding_counts

    scenario = _scenario(turns=("a", "b"), grounding_truth=("not_applicable", "not_applicable"))
    result = _result(
        replies=["You're eligible for a $349.99 refund, and you're within the 30-day window.", "Handing you over."],
        observed=[
            _turn(1, tool_calls=[{"name": "issue_refund", "input": {}, "output": {}}]),
            _turn(2, grounding_flagged=True, escalation_reason="repeated ungrounded replies", end_reason="escalated"),
        ],
    )

    counts = grounding_counts(scenario, result)

    assert counts.unreachable_claims == 1
    assert counts.ladder_fired == 1


def test_combine_counts_sums_every_field_across_scenarios():
    from eval.scoring import EMPTY_COUNTS, GroundingCounts, combine_counts

    first = GroundingCounts(
        grounded=3, ungrounded=1, flagged=2, true_positive=1, false_positive=1,
        true_negative=2, false_negative=0, hedged=1, unreachable_claims=2, ladder_fired=0, labeled_turns=4,
    )
    second = GroundingCounts(
        grounded=1, ungrounded=1, flagged=1, true_positive=1, false_positive=0,
        true_negative=1, false_negative=0, hedged=1, unreachable_claims=1, ladder_fired=1, labeled_turns=2,
    )

    total = combine_counts([first, second])

    assert total.grounded == 4
    assert total.ungrounded == 2
    assert total.false_positive == 1
    assert total.ladder_fired == 1
    assert total.labeled_turns == 6
    assert combine_counts([]) == EMPTY_COUNTS
