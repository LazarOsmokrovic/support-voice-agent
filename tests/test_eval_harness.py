"""Phase 10c: tests for the eval suite's own machinery — scenario integrity,
the recording format, the replay seam, and the harness.

These live in tests/ (not eval/) on purpose: they inherit tests/conftest.py's
autouse fixture, so no test here can fire a real webhook or write to the
repo's own logs/turns.jsonl. Deliberately NOT tested: that any particular
scenario passes. That is the eval's job — asserting it here would recreate
the duplicate harness this phase exists to remove.
"""

from __future__ import annotations

from eval.scenarios import (
    CAPABILITIES,
    DbAssertion,
    Expectations,
    Scenario,
    ToolExpectation,
    scenario_by_name,
)


def _minimal_scenario(**overrides) -> Scenario:
    base = dict(
        name="demo",
        capability="order_status",
        customer_id="CUST-1001",
        turns=("Where is my order?",),
        expect=Expectations(),
        grounding_truth=("not_applicable",),
    )
    base.update(overrides)
    return Scenario(**base)


def test_capabilities_are_exactly_the_six_the_project_promises():
    assert CAPABILITIES == (
        "order_status",
        "refunds",
        "policy_qa",
        "triage",
        "scheduling",
        "summary",
    )


def test_scenario_is_frozen_so_a_run_cannot_mutate_the_contract():
    import dataclasses

    import pytest

    scenario = _minimal_scenario()
    with pytest.raises(dataclasses.FrozenInstanceError):
        scenario.name = "something else"  # type: ignore[misc]


def test_scenario_defaults_leave_the_clock_unshifted_and_the_session_open():
    scenario = _minimal_scenario()
    assert scenario.clock_offset_days == 0
    assert scenario.close_session is False
    assert scenario.notes == ""


def test_expectations_default_to_asserting_nothing_except_no_pii():
    expect = Expectations()
    assert expect.tools_called == ()
    assert expect.tools_not_called == ()
    assert expect.escalation_turn is None
    assert expect.escalation_reason is None
    assert expect.end_reason is None
    assert expect.db_assertions == ()
    assert expect.no_pii_in_records is True


def test_tool_expectation_and_db_assertion_carry_their_optional_halves():
    tool = ToolExpectation(name="issue_refund", args_subset={"condition": "unopened_or_unwanted"}, turn=2)
    assert (tool.name, tool.args_subset, tool.turn) == (
        "issue_refund",
        {"condition": "unopened_or_unwanted"},
        2,
    )
    assertion = DbAssertion(sql="SELECT * FROM refunds WHERE order_id = ?", params=("x",), rows=1)
    assert assertion.columns is None


def test_scenario_by_name_returns_none_for_an_unknown_name():
    assert scenario_by_name("no_such_scenario_exists") is None
