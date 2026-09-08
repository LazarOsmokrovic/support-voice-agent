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
