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


def test_pii_scoring_flags_a_boundary_mangled_tracking_number(tmp_path):
    """Phase 10a's actual damage shape: a phone-like pattern eats the digit
    run inside an alphanumeric tracking number, leaving only the non-digit
    boundary letters intact — TBA123456789US becomes TBA[redacted-phone]US.
    Only 3 characters survive on each side of the marker, so a heuristic
    demanding a long surviving prefix (the code this replaces) missed this
    shape entirely; that's the Critical the review caught."""
    customer_id, _name, _email, _phone = mock_db.CUSTOMERS[0]
    order_id, _cust, _item, _q, _p, _s, _od, _ed, tracking = mock_db.ORDERS[0]
    mangled_tracking = f"{tracking[:3]}[redacted-phone]{tracking[-2:]}"
    scenario = _scenario(customer_id=customer_id)
    result = _result(
        log_lines=[{"reply": f"Order {order_id} shipped, tracking {mangled_tracking}", "user_text": "hi"}]
    )
    failures = score_pii(scenario, result)
    assert any(tracking in failure.detail for failure in failures)
    assert all(failure.kind == "pii" for failure in failures)


def test_pii_scoring_flags_a_fully_wiped_order_id(tmp_path):
    """The other real damage shape: a card-like pattern's replacement
    swallows an identifier whole, leaving zero surviving characters —
    112-3487561-2938471 becomes [redacted-number]. No surviving-prefix
    heuristic could ever catch this one, however short the prefix
    requirement; the check has to work from full-string absence, not
    remnant characters."""
    customer_id, _name, _email, _phone = mock_db.CUSTOMERS[0]
    order_id, _cust, _item, _q, _p, _s, _od, _ed, tracking = mock_db.ORDERS[0]
    scenario = _scenario(customer_id=customer_id)
    result = _result(
        log_lines=[{"reply": f"Order [redacted-number] shipped, tracking {tracking}", "user_text": "hi"}]
    )
    failures = score_pii(scenario, result)
    assert any(order_id in failure.detail for failure in failures)
    assert all(failure.kind == "pii" for failure in failures)


def test_pii_scoring_flags_a_destroyed_order_id_with_no_tracking_sibling(tmp_path):
    """A third seed shape, and the round-2 review finding: several orders
    (e.g. 'Processing' and 'Cancelled' ones) have no tracking_number at all,
    so order_id has no sibling identifier to anchor against when IT is the
    one destroyed. `item` fills that role — it is free text this project's
    redactor never touches, so its survival intact confirms this record
    concerns this specific order even with order_id gone."""
    order_id, customer_id, item, *_rest = next(order for order in mock_db.ORDERS if order[8] is None)
    scenario = _scenario(customer_id=customer_id)
    result = _result(
        log_lines=[
            {"reply": f"Your {item} order [redacted-number] is currently being processed", "user_text": "hi"}
        ]
    )
    failures = score_pii(scenario, result)
    assert any(order_id in failure.detail for failure in failures)
    assert all(failure.kind == "pii" for failure in failures)


def test_pii_scoring_does_not_flag_an_unpaired_order_that_is_simply_not_mentioned(tmp_path):
    """The other half of the same tension: an unrelated redaction marker
    present somewhere in a record must not be mistaken for THIS order's
    lone identifier having been destroyed, when the record never mentions
    this order (not even its item) at all."""
    order_id, customer_id, _item, *_rest = next(order for order in mock_db.ORDERS if order[8] is None)
    scenario = _scenario(customer_id=customer_id)
    result = _result(
        log_lines=[{"reply": "Thanks for calling, have a great day.", "user_text": "[redacted-email]"}]
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


def _report(*scenarios):
    from eval.report import EvalReport

    return EvalReport(scenarios=list(scenarios))


def _scenario_report(name, outcome, capability="refunds", turns=2, details=None):
    from eval.report import ScenarioReport

    return ScenarioReport(name=name, capability=capability, turns=turns, outcome=outcome, details=details or [])


def test_exit_code_is_zero_only_when_everything_passed():
    from eval.report import exit_code

    assert exit_code(_report(_scenario_report("a", "PASS"), _scenario_report("b", "PASS"))) == 0


def test_exit_code_one_for_a_behavioural_failure_and_two_for_stale_fixtures():
    """Three codes because CI should go red on a regression and go red
    DIFFERENTLY on 'your fixtures need refreshing' — the fixes differ, and
    conflating them trains people to ignore the signal."""
    from eval.report import exit_code

    assert exit_code(_report(_scenario_report("a", "FAIL"), _scenario_report("b", "STALE"))) == 1
    assert exit_code(_report(_scenario_report("a", "PASS"), _scenario_report("b", "STALE"))) == 2
    assert exit_code(_report(_scenario_report("a", "MISSING"))) == 2
    assert exit_code(_report(_scenario_report("a", "DRIFT"))) == 2
    assert exit_code(_report(_scenario_report("a", "ERROR"))) == 2


def test_strict_collapses_the_fixture_code_into_the_failure_code():
    from eval.report import exit_code

    report = _report(_scenario_report("a", "PASS"), _scenario_report("b", "STALE"))
    assert exit_code(report, strict=True) == 1
    assert exit_code(_report(_scenario_report("a", "PASS")), strict=True) == 0


def test_render_shows_every_outcome_the_capability_tally_and_the_grounding_block():
    from eval.report import render
    from eval.scoring import GroundingCounts

    report = _report(
        _scenario_report("order_status_delivered", "PASS", capability="order_status", turns=3),
        _scenario_report("refund_low_value_propose_then_confirm", "FAIL", details=["db: expected 1 row, found 0"]),
        _scenario_report("scheduling_book_then_reschedule", "STALE", capability="scheduling", turns=6),
    )
    report.grounding = GroundingCounts(
        grounded=9, ungrounded=2, flagged=3, true_positive=2, false_positive=1,
        true_negative=8, false_negative=0, hedged=2, unreachable_claims=4, ladder_fired=1, labeled_turns=11,
    )
    report.pii_leaks = 0
    report.stored_records = {"turn_log": 68, "tickets": 4, "escalations": 4}

    text = render(report)

    assert "PASS   order_status_delivered" in text
    assert "FAIL   refund_low_value_propose_then_confirm" in text
    assert "db: expected 1 row, found 0" in text
    assert "1 passed · 1 failed · 1 stale" in text
    assert "capability coverage" in text
    assert "false positives       1/9" in text
    assert "unreachable claims      4" in text
    assert "pii: 0 leaks across 76 stored records" in text


def test_render_never_crashes_on_an_empty_or_all_not_applicable_report():
    """Spec §9 test 11. A report that dies on the degenerate case is a report
    nobody can trust on the interesting one."""
    from eval.report import render
    from eval.scoring import EMPTY_COUNTS

    empty = _report()
    empty.grounding = EMPTY_COUNTS
    text = render(empty)
    assert "0 passed" in text
    assert "0/0 — insufficient data" in text

    errored = _report(_scenario_report("boom", "ERROR", details=["RuntimeError: exploded"]))
    errored.grounding = EMPTY_COUNTS
    assert "ERROR  boom" in render(errored)


def test_to_json_emits_the_whole_report_as_one_serialisable_object():
    import json

    from eval.report import to_json
    from eval.scoring import EMPTY_COUNTS

    report = _report(_scenario_report("a", "PASS"))
    report.grounding = EMPTY_COUNTS
    payload = to_json(report)

    assert payload["scenarios"][0]["outcome"] == "PASS"
    assert payload["summary"]["passed"] == 1
    assert payload["grounding"]["false_positive"] == 0
    json.dumps(payload)  # must be serialisable, not just dict-shaped
