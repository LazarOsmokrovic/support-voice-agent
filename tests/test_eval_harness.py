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


def test_current_hashes_covers_every_hash_field_and_is_stable():
    from eval.recording import HASH_FIELDS, current_hashes

    first = current_hashes()
    second = current_hashes()
    assert set(first) == set(HASH_FIELDS)
    assert first == second
    assert all(len(value) == 64 for value in first.values())


def test_seed_hash_changes_when_the_seed_changes():
    from data import mock_db
    from eval.recording import current_hashes

    before = current_hashes()["seed_sha256"]
    original = mock_db.ORDERS
    try:
        mock_db.ORDERS = [*original, ("999-9999999-9999999", "CUST-1001", "x", 1, 1.0, "Delivered", "2026-01-01", "2026-01-02", None)]
        after = current_hashes()["seed_sha256"]
    finally:
        mock_db.ORDERS = original
    assert before != after


def _recording(**overrides):
    from eval.recording import Recording, current_hashes

    base = dict(
        scenario="demo",
        recorded_at="2026-09-08T12:00:00",
        model="claude-opus-5",
        anthropic_sdk_version="1.0.0",
        creates=[],
        parses=[],
        observed=[],
        **current_hashes(),
    )
    base.update(overrides)
    return Recording(**base)


def test_stale_fields_is_empty_for_a_recording_taken_right_now():
    from eval.recording import stale_fields

    assert stale_fields(_recording()) == []


def test_stale_fields_names_the_changed_hash_and_both_values():
    from eval.recording import current_hashes, stale_fields

    changed = _recording(system_prompt_sha256="0" * 64)
    result = stale_fields(changed)
    assert len(result) == 1
    name, recorded, current = result[0]
    assert name == "system_prompt_sha256"
    assert recorded == "0" * 64
    assert current == current_hashes()["system_prompt_sha256"]


def test_save_then_load_round_trips_a_recording(tmp_path, monkeypatch):
    from eval import recording as recording_module
    from eval.recording import load_recording, save_recording

    monkeypatch.setattr(recording_module, "RECORDINGS_DIR", tmp_path)
    original = _recording(creates=[{"id": "msg_1"}], parses=[{"output_format": "TurnClassification"}])

    path = save_recording(original)

    assert path.name == "demo.json"
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert '\n  "scenario": "demo"' in path.read_text(encoding="utf-8")
    assert load_recording("demo") == original


def test_load_recording_returns_none_when_there_is_no_file(tmp_path, monkeypatch):
    from eval import recording as recording_module
    from eval.recording import load_recording

    monkeypatch.setattr(recording_module, "RECORDINGS_DIR", tmp_path)
    assert load_recording("never_recorded") is None


def test_frozen_now_applies_the_scenarios_clock_offset():
    from datetime import datetime

    from eval.recording import frozen_now

    scenario = _minimal_scenario(clock_offset_days=45)
    assert frozen_now(_recording(), scenario) == datetime(2026, 10, 23, 12, 0, 0)
    assert frozen_now(_recording(), _minimal_scenario()) == datetime(2026, 9, 8, 12, 0, 0)
