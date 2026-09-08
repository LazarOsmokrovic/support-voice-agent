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


def _calm_parse_entry() -> dict:
    return {
        "output_format": "TurnClassification",
        "parsed_output": {"intent": "chitchat", "sentiment": "neutral", "policy_restricted": False},
    }


@pytest.mark.asyncio
async def test_run_scenario_drives_every_turn_and_returns_one_observed_row_each(tmp_path):
    from eval.harness import run_scenario
    from eval.replay import FakeAnthropicClient

    scenario = _minimal_scenario(
        name="two_turn_demo",
        turns=("Hello there", "Thanks, bye"),
        grounding_truth=("not_applicable", "not_applicable"),
    )
    client = FakeAnthropicClient(
        "two_turn_demo",
        [_message_payload(content=[{"type": "text", "text": "Hi!"}]), _message_payload(content=[{"type": "text", "text": "Bye!"}])],
        [_calm_parse_entry(), _calm_parse_entry()],
    )

    result = await run_scenario(scenario, client, datetime(2026, 9, 8, 12, 0, 0), tmp_path)

    assert result.error is None
    assert result.replies == ["Hi!", "Bye!"]
    assert [row.turn for row in result.observed] == [1, 2]
    assert result.observed[0].end_reason is None


@pytest.mark.asyncio
async def test_run_scenario_holds_the_seam_with_a_garbage_api_key_present(tmp_path, monkeypatch):
    """Spec §9 test 6: end to end under the patch with a garbage key. If the
    seam leaked, the real SDK would be constructed and the call would fail
    with an auth error rather than returning the recorded reply."""
    from eval.harness import run_scenario
    from eval.replay import FakeAnthropicClient

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key-at-all")
    scenario = _minimal_scenario(name="seam_demo", turns=("Hello",), grounding_truth=("not_applicable",))
    client = FakeAnthropicClient("seam_demo", [_message_payload()], [_calm_parse_entry()])

    result = await run_scenario(scenario, client, datetime(2026, 9, 8, 12, 0, 0), tmp_path)

    assert result.error is None
    assert result.replies == ["Happy to help!"]
    assert client.creates_remaining == 0
    assert client.parses_remaining == 0


@pytest.mark.asyncio
async def test_run_scenario_writes_one_real_log_line_per_captured_record(tmp_path):
    """The pass-through spy's whole point. log_turn never raises, so a
    missing record would otherwise be indistinguishable from a disabled log.
    Comparing the two counts turns that never-raise policy from a blind spot
    into a checked invariant — and the file's real bytes are what PII
    scoring reads, because they went through the actual redacting
    serialiser."""
    from eval.harness import run_scenario
    from eval.replay import FakeAnthropicClient

    scenario = _minimal_scenario(name="log_demo", turns=("One", "Two"), grounding_truth=("not_applicable",) * 2)
    client = FakeAnthropicClient(
        "log_demo",
        [_message_payload(), _message_payload()],
        [_calm_parse_entry(), _calm_parse_entry()],
    )

    result = await run_scenario(scenario, client, datetime(2026, 9, 8, 12, 0, 0), tmp_path)

    assert len(result.records) == 2
    assert len(result.log_lines) == len(result.records)
    assert result.log_lines[0]["transport"] == "eval"


@pytest.mark.asyncio
async def test_run_scenario_seeds_a_fresh_database_and_leaves_the_real_one_alone(tmp_path):
    from data import mock_db
    from eval.harness import run_scenario
    from eval.replay import FakeAnthropicClient

    real_db_path = mock_db.DB_PATH
    scenario = _minimal_scenario(name="db_demo", turns=("Hi",), grounding_truth=("not_applicable",))
    client = FakeAnthropicClient("db_demo", [_message_payload()], [_calm_parse_entry()])

    result = await run_scenario(scenario, client, datetime(2026, 9, 8, 12, 0, 0), tmp_path)

    assert mock_db.DB_PATH == real_db_path
    assert result.db_path.parent == tmp_path
    assert result.db_path != real_db_path
    import sqlite3

    conn = sqlite3.connect(result.db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == len(mock_db.ORDERS)
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_run_scenario_reports_an_exhausted_recording_as_an_error_rather_than_raising(tmp_path):
    from eval.harness import run_scenario
    from eval.replay import FakeAnthropicClient

    scenario = _minimal_scenario(name="short_demo", turns=("One", "Two"), grounding_truth=("not_applicable",) * 2)
    client = FakeAnthropicClient("short_demo", [_message_payload()], [_calm_parse_entry()])

    result = await run_scenario(scenario, client, datetime(2026, 9, 8, 12, 0, 0), tmp_path)

    assert result.error is not None
    assert "short_demo" in result.error
    assert "create #2" in result.error
    assert result.replies == ["Happy to help!"]


@pytest.mark.asyncio
async def test_recording_client_captures_creates_and_parses_while_delegating(tmp_path):
    """Built and tested against a fake inner client on purpose — nothing in
    this plan ever calls the real API."""
    from agent.tools.escalation import TurnClassification

    from eval.record import RecordingAnthropicClient
    from eval.replay import FakeAnthropicClient

    inner = FakeAnthropicClient("inner", [_message_payload()], [_calm_parse_entry()])
    wrapper = RecordingAnthropicClient(inner)

    message = await wrapper.messages.create(model="m", max_tokens=1, messages=[])
    parsed = await wrapper.messages.parse(
        model="m", max_tokens=1, messages=[], output_format=TurnClassification
    )

    assert message.stop_reason == "end_turn"
    assert parsed.parsed_output.intent == "chitchat"
    assert len(wrapper.creates) == 1
    assert wrapper.creates[0]["content"][0]["text"] == "Happy to help!"
    assert wrapper.parses == [
        {
            "output_format": "TurnClassification",
            "parsed_output": {"intent": "chitchat", "sentiment": "neutral", "policy_restricted": False},
        }
    ]


@pytest.mark.asyncio
async def test_record_scenario_builds_a_recording_with_current_hashes(tmp_path, monkeypatch):
    from eval import recording as recording_module
    from eval.record import record_scenario
    from eval.recording import current_hashes
    from eval.replay import FakeAnthropicClient

    monkeypatch.setattr(recording_module, "RECORDINGS_DIR", tmp_path / "recordings")
    scenario = _minimal_scenario(name="record_demo", turns=("Hi",), grounding_truth=("not_applicable",))

    def factory():
        return FakeAnthropicClient("record_demo", [_message_payload()], [_calm_parse_entry()])

    recording, result = await record_scenario(scenario, tmp_path / "work", factory)

    assert result.error is None
    assert recording.scenario == "record_demo"
    assert len(recording.creates) == 1
    assert len(recording.parses) == 1
    assert len(recording.observed) == 1
    for name, value in current_hashes().items():
        assert getattr(recording, name) == value
    datetime.fromisoformat(recording.recorded_at)  # parses, i.e. is a usable frozen clock


def test_grounding_worksheet_emits_a_paste_ready_block_with_one_label_per_turn():
    from eval.harness import HarnessResult, ObservedTurn
    from eval.record import grounding_worksheet

    scenario = _minimal_scenario(name="ws_demo", turns=("a", "b"), grounding_truth=())
    result = HarnessResult(
        scenario="ws_demo",
        replies=["You have 30 days.", "Anything else?"],
        observed=[
            ObservedTurn(1, [{"name": "search_policy", "input": {}, "output": {"found": True, "results": []}}], True, True, None, None, "dict"),
            ObservedTurn(2, [], False, False, None, "model_ended", None),
        ],
    )

    text = grounding_worksheet(scenario, result)

    assert "grounding_truth=(" in text
    assert text.count('"not_applicable"') == 2
    assert "grounding_flagged=True" in text
    assert "You have 30 days." in text


def test_record_main_exits_clearly_when_there_is_no_api_key(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from eval.record import main

    assert main(["--all"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().out


def test_every_scenario_references_a_customer_that_exists_in_the_seed():
    from data.mock_db import CUSTOMERS
    from eval.scenarios import SCENARIOS

    known = {row[0] for row in CUSTOMERS}
    for scenario in SCENARIOS:
        assert scenario.customer_id in known, f"{scenario.name} references unknown customer {scenario.customer_id}"


def test_every_order_id_mentioned_by_a_scenario_exists_in_the_seed():
    """The project's most-repeated defect: a hand-typed order ID that no
    longer matches the seed produces a scenario which tests nothing and
    fails plausibly. Any 3-7-7 shaped ID in a turn or a db_assertion param
    must resolve, unless the scenario deliberately uses an unknown one."""
    import re

    from data.mock_db import ORDERS
    from eval.scenarios import SCENARIOS

    known = {row[0] for row in ORDERS}
    pattern = re.compile(r"\b\d{3}-\d{7}-\d{7}\b")
    deliberately_unknown = {"order_status_invalid_id_then_correct", "triage_repeated_failed_lookups"}
    for scenario in SCENARIOS:
        haystack = " ".join(scenario.turns) + " " + " ".join(
            str(param) for assertion in scenario.expect.db_assertions for param in assertion.params
        )
        for found in pattern.findall(haystack):
            if scenario.name in deliberately_unknown and found not in known:
                continue
            assert found in known, f"{scenario.name} references unknown order {found}"


def test_every_escalation_reason_is_a_literal_the_code_can_actually_produce():
    from agent.tools import escalation as escalation_module
    from eval.scenarios import SCENARIOS

    fixed = {
        "explicit request for a human",
        "policy-restricted topic",
        "sustained negative sentiment across multiple turns",
        "repeated failed lookups",
        "repeated ungrounded replies",
    }
    source = (
        __import__("pathlib").Path(escalation_module.__file__).read_text(encoding="utf-8")
    )
    for literal in fixed:
        assert literal in source, f"{literal!r} is no longer produced by agent/tools/escalation.py"
    for scenario in SCENARIOS:
        reason = scenario.expect.escalation_reason
        if reason is None:
            continue
        assert reason in fixed or reason.startswith("high-value refund ("), scenario.name


def test_every_scenario_declares_one_grounding_label_per_turn_and_a_known_capability():
    from eval.scenarios import CAPABILITIES, SCENARIOS

    for scenario in SCENARIOS:
        assert scenario.capability in CAPABILITIES, scenario.name
        assert len(scenario.grounding_truth) == len(scenario.turns), scenario.name
        assert all(
            label in ("grounded", "ungrounded", "not_applicable") for label in scenario.grounding_truth
        ), scenario.name


def test_scenario_names_are_unique_and_usable_as_recording_filenames():
    import re

    from eval.scenarios import SCENARIOS, scenario_by_name

    names = [scenario.name for scenario in SCENARIOS]
    assert len(names) == len(set(names))
    for name in names:
        assert re.fullmatch(r"[a-z0-9_]+", name), name
        assert scenario_by_name(name) is not None


def test_the_roster_is_exactly_twenty_and_covers_all_six_capabilities():
    """PROJECT_PLAN.md promises 10-20 scenarios across all six features.
    That contract is asserted here rather than assumed."""
    from collections import Counter

    from eval.scenarios import CAPABILITIES, SCENARIOS

    assert len(SCENARIOS) == 20
    assert 10 <= len(SCENARIOS) <= 20
    tally = Counter(scenario.capability for scenario in SCENARIOS)
    assert set(tally) == set(CAPABILITIES)
    assert tally["order_status"] == 3
    assert tally["refunds"] == 4
    assert tally["policy_qa"] == 4
    assert tally["triage"] == 6
    assert tally["scheduling"] == 2
    assert tally["summary"] == 1


def test_every_escalation_trigger_the_agent_can_take_is_exercised_by_some_scenario():
    """Before this phase, live coverage reached two of five escalation
    triggers (agent/tools/escalation.py:161-186). These scenarios close the
    other three, so every escalation path is exercised for the first time."""
    from eval.scenarios import SCENARIOS

    reasons = {scenario.expect.escalation_reason for scenario in SCENARIOS}
    assert "explicit request for a human" in reasons
    assert "policy-restricted topic" in reasons
    assert "sustained negative sentiment across multiple turns" in reasons
    assert "repeated failed lookups" in reasons
    assert "repeated ungrounded replies" in reasons
    assert any(r and r.startswith("high-value refund (") for r in reasons)
