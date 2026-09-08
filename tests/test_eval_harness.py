"""Phase 10c: tests for the eval suite's own machinery — scenario integrity,
the recording format, the replay seam, and the harness.

These live in tests/ (not eval/) on purpose: they inherit tests/conftest.py's
autouse fixture, so no test here can fire a real webhook or write to the
repo's own logs/turns.jsonl. Deliberately NOT tested: that any particular
scenario passes. That is the eval's job — asserting it here would recreate
the duplicate harness this phase exists to remove.
"""

from __future__ import annotations

import pytest

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


def _message_payload(**overrides) -> dict:
    base = {
        "id": "msg_eval_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "content": [{"type": "text", "text": "Happy to help!"}],
    }
    base.update(overrides)
    return base


def _tool_use_payload() -> dict:
    from data.mock_db import ORDERS

    return _message_payload(
        stop_reason="tool_use",
        content=[
            {"type": "text", "text": "Let me check that."},
            {
                "type": "tool_use",
                "id": "toolu_eval_1",
                "name": "get_order_status",
                "input": {"order_id": ORDERS[0][0]},
            },
        ],
    )


def test_rebuild_message_produces_a_real_sdk_message_not_a_mock():
    from anthropic.types import Message

    from eval.replay import rebuild_message

    message = rebuild_message(_message_payload())
    assert isinstance(message, Message)
    assert message.stop_reason == "end_turn"
    assert message.content[0].type == "text"
    assert message.content[0].text == "Happy to help!"


def test_rebuild_message_gives_a_tool_use_block_the_attributes_agent_send_consumes():
    """agent/core.py:125-130 reads block.type, block.name, block.id and
    block.input, and puts block.input straight into TurnResult.tool_calls.
    A dict-LIKE input would pass isinstance checks nowhere and corrupt every
    downstream args_subset comparison, so this pins the runtime type."""
    from data.mock_db import ORDERS

    from eval.replay import rebuild_message

    message = rebuild_message(_tool_use_payload())
    block = [b for b in message.content if b.type == "tool_use"][0]
    assert block.name == "get_order_status"
    assert block.id == "toolu_eval_1"
    assert type(block.input) is dict
    assert block.input == {"order_id": ORDERS[0][0]}


@pytest.mark.asyncio
async def test_fake_client_pops_creates_in_recorded_order():
    from eval.replay import FakeAnthropicClient

    client = FakeAnthropicClient("demo", [_tool_use_payload(), _message_payload()], [])

    first = await client.messages.create(model="claude-opus-5", max_tokens=1024, messages=[])
    second = await client.messages.create(model="claude-opus-5", max_tokens=1024, messages=[])

    assert first.stop_reason == "tool_use"
    assert second.stop_reason == "end_turn"
    assert client.creates_consumed == 2
    assert client.creates_remaining == 0


@pytest.mark.asyncio
async def test_fake_client_raises_recording_exhausted_naming_scenario_and_index():
    from eval.replay import FakeAnthropicClient, RecordingExhausted

    client = FakeAnthropicClient("refund_high_value_escalates", [_message_payload()], [])
    await client.messages.create(model="m", max_tokens=1, messages=[])

    with pytest.raises(RecordingExhausted) as excinfo:
        await client.messages.create(model="m", max_tokens=1, messages=[])
    assert "refund_high_value_escalates" in str(excinfo.value)
    assert "create #2" in str(excinfo.value)
    assert "1 recorded" in str(excinfo.value)


@pytest.mark.asyncio
async def test_fake_client_parse_dispatches_on_output_format_and_returns_parsed_output():
    from agent.tools.escalation import TurnClassification

    from eval.replay import FakeAnthropicClient

    client = FakeAnthropicClient(
        "demo",
        [],
        [
            {
                "output_format": "TurnClassification",
                "parsed_output": {"intent": "chitchat", "sentiment": "neutral", "policy_restricted": False},
            }
        ],
    )

    response = await client.messages.parse(
        model="m", max_tokens=256, messages=[], output_format=TurnClassification
    )

    assert isinstance(response.parsed_output, TurnClassification)
    assert response.parsed_output.intent == "chitchat"
    assert client.parses_consumed == 1


@pytest.mark.asyncio
async def test_fake_client_parse_raises_mismatch_when_the_call_order_diverges():
    from agent.tools.summary import SessionSummary

    from eval.replay import FakeAnthropicClient, RecordingMismatch

    client = FakeAnthropicClient(
        "demo",
        [],
        [
            {
                "output_format": "TurnClassification",
                "parsed_output": {"intent": "chitchat", "sentiment": "neutral", "policy_restricted": False},
            }
        ],
    )

    with pytest.raises(RecordingMismatch) as excinfo:
        await client.messages.parse(model="m", max_tokens=1, messages=[], output_format=SessionSummary)
    assert "SessionSummary" in str(excinfo.value)
    assert "TurnClassification" in str(excinfo.value)
