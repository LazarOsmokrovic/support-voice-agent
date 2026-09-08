"""Phase 10c: tests for the eval suite's own machinery — scenario integrity,
the recording format, the replay seam, and the harness.

These live in tests/ (not eval/) on purpose: they inherit tests/conftest.py's
autouse fixture, so no test here can fire a real webhook or write to the
repo's own logs/turns.jsonl. Deliberately NOT tested: that any particular
scenario passes. That is the eval's job — asserting it here would recreate
the duplicate harness this phase exists to remove.
"""

from __future__ import annotations

from datetime import datetime

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


def test_the_documented_model_construction_sites_are_still_the_only_ones():
    """Spec §3 enumerates four anthropic.AsyncAnthropic() construction sites.
    An enumeration in a comment rots; this makes it a maintained fact. If a
    fifth site appears, this fails and the enumeration gets updated — the
    seam itself still covers it, because it patches the constructor."""
    import re
    from pathlib import Path

    from eval.replay import MODEL_CONSTRUCTION_SITES

    root = Path(__file__).resolve().parent.parent
    found = set()
    for path in sorted((root / "agent").rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if re.search(r"anthropic\.AsyncAnthropic\(", line):
                found.add((path.relative_to(root).as_posix(), number))
    assert found == set(MODEL_CONSTRUCTION_SITES)


def test_the_documented_decision_affecting_clock_sites_are_still_the_only_ones():
    """Only a bare datetime.now() can change a decision (the refund window,
    which slots exist). datetime.now(timezone.utc) writes stored strings and
    is deliberately left real, so it is excluded here."""
    import re
    from pathlib import Path

    from eval.replay import FROZEN_CLOCK_SITES

    root = Path(__file__).resolve().parent.parent
    found = set()
    for path in sorted((root / "agent").rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if re.search(r"datetime\.now\(\s*\)", line):
                found.add((path.relative_to(root).as_posix(), number))
    assert found == set(FROZEN_CLOCK_SITES)


def test_frozen_datetime_freezes_naive_now_but_leaves_aware_now_real():
    from datetime import datetime, timezone

    from eval.replay import frozen_datetime_class

    frozen = datetime(2026, 9, 8, 12, 0, 0)
    cls = frozen_datetime_class(frozen)

    assert cls.now() == frozen
    assert cls.now().tzinfo is None
    aware = cls.now(timezone.utc)
    assert aware.tzinfo is timezone.utc
    assert abs((aware.replace(tzinfo=None) - datetime.utcnow()).total_seconds()) < 5


def test_frozen_datetime_still_parses_iso_strings_the_tools_depend_on():
    from datetime import datetime

    from eval.replay import frozen_datetime_class

    cls = frozen_datetime_class(datetime(2026, 9, 8, 12, 0, 0))
    delivered = cls.fromisoformat("2026-08-31")
    assert (cls.now() - delivered).days == 8


def test_scenario_patch_intercepts_every_constructor_and_blocks_the_sync_client(tmp_path):
    import anthropic

    from eval.replay import FakeAnthropicClient, scenario_patch

    fake = FakeAnthropicClient("demo", [], [])
    with scenario_patch(fake, datetime(2026, 9, 8, 12, 0, 0), tmp_path / "turns.jsonl"):
        assert anthropic.AsyncAnthropic() is fake
        assert anthropic.AsyncAnthropic(api_key="garbage") is fake
        with pytest.raises(RuntimeError, match="synchronous"):
            anthropic.Anthropic()
    assert anthropic.AsyncAnthropic is not fake


def test_scenario_patch_strips_webhook_env_and_points_the_turn_log_at_its_own_file(tmp_path, monkeypatch):
    import os

    from eval.replay import FakeAnthropicClient, scenario_patch

    monkeypatch.setenv("ESCALATION_WEBHOOK_URL", "https://real.example.com/hook")
    monkeypatch.setenv("ESCALATION_WEBHOOK_SECRET", "s3cret")
    log_path = tmp_path / "turns.jsonl"

    with scenario_patch(FakeAnthropicClient("demo", [], []), datetime(2026, 9, 8), log_path):
        assert "ESCALATION_WEBHOOK_URL" not in os.environ
        assert "ESCALATION_WEBHOOK_SECRET" not in os.environ
        assert os.environ["TURN_LOG_PATH"] == str(log_path)

    assert os.environ["ESCALATION_WEBHOOK_URL"] == "https://real.example.com/hook"


def test_the_frozen_clock_reaches_issue_refund_and_decides_the_window(tmp_path, monkeypatch):
    """The regression test for the defect that motivated this phase.
    tests/test_text_cli.py's high-value refund test silently degraded into a
    window check when the calendar moved past the seeded delivery date.
    Frozen inside the window it is eligible; frozen outside it is not — and
    neither answer depends on what today happens to be."""
    from datetime import timedelta

    from agent.confirmation import PendingActionGate
    from agent.tools.refunds import issue_refund
    from data import mock_db
    from eval.replay import FakeAnthropicClient, scenario_patch

    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "eval_clock.db")
    mock_db.reset_and_seed()
    order_id, _customer, _item, _qty, _price, _status, _ordered, delivered, _tracking = mock_db.ORDERS[0]
    delivered_on = datetime.fromisoformat(delivered)
    log_path = tmp_path / "turns.jsonl"

    inside = delivered_on + timedelta(days=5)
    with scenario_patch(FakeAnthropicClient("demo", [], []), inside, log_path):
        result = issue_refund(
            order_id=order_id,
            condition="unopened_or_unwanted",
            reason="changed my mind",
            state=PendingActionGate(),
            customer_id=mock_db.ORDERS[0][1],
        )
    assert result.get("error") != "outside_window"
    assert result["status"] == "pending_confirmation"

    outside = delivered_on + timedelta(days=45)
    with scenario_patch(FakeAnthropicClient("demo", [], []), outside, log_path):
        result = issue_refund(
            order_id=order_id,
            condition="unopened_or_unwanted",
            reason="changed my mind",
            state=PendingActionGate(),
            customer_id=mock_db.ORDERS[0][1],
        )
    assert result["error"] == "outside_window"
