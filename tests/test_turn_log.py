"""Phase 10b: observability/turn_log.py — one structured JSON record per turn.

Every test isolates the log file with tmp_path and TURN_LOG_PATH, the same way
the DB tests isolate mock_db.DB_PATH.
"""

from __future__ import annotations

import json
import threading

import pytest

from data.mock_db import ORDERS
from observability.turn_log import TurnRecord, log_turn


def _record(**overrides) -> TurnRecord:
    base = dict(
        session_id="sess-abc",
        customer_id="CUST-1001",
        transport="text_cli",
        turn=3,
        user_text="Where is my order?",
        reply="It ships tomorrow.",
        original_reply=None,
        grounding_flagged=False,
        hedge_spoken=False,
        tool_calls=[],
        llm_latency_seconds=1.8404,
        warnings=[],
        escalated=False,
        escalation_reason=None,
        escalation_id=None,
        ended=False,
        end_reason=None,
    )
    base.update(overrides)
    return TurnRecord(**base)


def _read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_log_turn_writes_one_parseable_json_line(tmp_path, monkeypatch):
    path = tmp_path / "turns.jsonl"
    monkeypatch.setenv("TURN_LOG_PATH", str(path))

    log_turn(_record())

    lines = _read_lines(path)
    assert len(lines) == 1
    assert lines[0]["session_id"] == "sess-abc"
    assert lines[0]["turn"] == 3
    assert lines[0]["transport"] == "text_cli"


def test_record_carries_every_schema_field(tmp_path, monkeypatch):
    path = tmp_path / "turns.jsonl"
    monkeypatch.setenv("TURN_LOG_PATH", str(path))

    log_turn(_record())

    record = _read_lines(path)[0]
    for field in (
        "ts", "session_id", "customer_id", "transport", "turn", "user_text", "reply",
        "original_reply", "grounding_flagged", "hedge_spoken", "tool_calls", "llm_latency_ms",
        "warnings", "escalated", "escalation_reason", "escalation_id", "ended", "end_reason",
    ):
        assert field in record, field


def test_latency_is_reported_as_integer_milliseconds(tmp_path, monkeypatch):
    path = tmp_path / "turns.jsonl"
    monkeypatch.setenv("TURN_LOG_PATH", str(path))

    log_turn(_record(llm_latency_seconds=1.8404))

    assert _read_lines(path)[0]["llm_latency_ms"] == 1840


def test_pii_is_redacted_in_text_and_tool_output(tmp_path, monkeypatch):
    path = tmp_path / "turns.jsonl"
    monkeypatch.setenv("TURN_LOG_PATH", str(path))

    log_turn(_record(
        user_text="my email is jane@example.com",
        reply="I'll call you on 555-123-4567.",
        tool_calls=[{"name": "get_order_status", "input": {}, "output": {"note": "card 4111 1111 1111 1111"}}],
    ))

    record = _read_lines(path)[0]
    assert "jane@example.com" not in record["user_text"]
    assert "555-123-4567" not in record["reply"]
    assert "4111 1111 1111 1111" not in json.dumps(record["tool_calls"])


def test_real_order_identifiers_survive_into_the_log(tmp_path, monkeypatch):
    """A log whose identifiers are masked is useless for debugging — this is
    the check Phase 11 and Phase 10a both learned to make."""
    path = tmp_path / "turns.jsonl"
    monkeypatch.setenv("TURN_LOG_PATH", str(path))
    order_id, delivery, tracking = ORDERS[0][0], ORDERS[0][7], ORDERS[0][8]

    log_turn(_record(
        user_text=f"where is {order_id}",
        tool_calls=[{"name": "get_order_status", "input": {"order_id": order_id},
                     "output": {"estimated_delivery": delivery, "tracking_number": tracking}}],
    ))

    blob = json.dumps(_read_lines(path)[0])
    assert order_id in blob
    assert delivery in blob
    assert tracking in blob


def test_disabled_by_empty_path_writes_nothing(tmp_path, monkeypatch):
    path = tmp_path / "turns.jsonl"
    monkeypatch.setenv("TURN_LOG_PATH", "")

    log_turn(_record())

    assert not path.exists()


def test_records_append_rather_than_overwrite(tmp_path, monkeypatch):
    path = tmp_path / "turns.jsonl"
    monkeypatch.setenv("TURN_LOG_PATH", str(path))

    log_turn(_record(turn=1))
    log_turn(_record(turn=2))

    assert [line["turn"] for line in _read_lines(path)] == [1, 2]


def test_creates_a_missing_parent_directory(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "dir" / "turns.jsonl"
    monkeypatch.setenv("TURN_LOG_PATH", str(path))

    log_turn(_record())

    assert path.exists()


def test_an_unwritable_path_does_not_raise(tmp_path, monkeypatch, caplog):
    """A telemetry failure must never break a call."""
    monkeypatch.setenv("TURN_LOG_PATH", str(tmp_path))  # a directory, not a file

    log_turn(_record())  # unguarded: an escaping exception fails this test

    assert "turn log" in caplog.text.lower()


def test_an_unserializable_value_does_not_raise(tmp_path, monkeypatch):
    path = tmp_path / "turns.jsonl"
    monkeypatch.setenv("TURN_LOG_PATH", str(path))

    log_turn(_record(tool_calls=[{"name": "x", "input": {}, "output": object()}]))


def test_pii_is_redacted_in_original_reply(tmp_path, monkeypatch):
    """original_reply carries the suppressed sentence a hedge replaced — it
    needs the same redaction as reply/user_text, not a free pass."""
    path = tmp_path / "turns.jsonl"
    monkeypatch.setenv("TURN_LOG_PATH", str(path))

    log_turn(_record(original_reply="I'll call you back on 555-123-4567."))

    record = _read_lines(path)[0]
    assert "555-123-4567" not in record["original_reply"]
    assert record["original_reply"] is not None


def test_none_original_reply_stays_none(tmp_path, monkeypatch):
    path = tmp_path / "turns.jsonl"
    monkeypatch.setenv("TURN_LOG_PATH", str(path))

    log_turn(_record(original_reply=None))

    assert _read_lines(path)[0]["original_reply"] is None


def test_pii_is_redacted_in_warnings(tmp_path, monkeypatch):
    """warnings carries interpolated exception text — the field most likely
    to accumulate unpredictable content, so redaction must not be
    forgettable at the call site."""
    path = tmp_path / "turns.jsonl"
    monkeypatch.setenv("TURN_LOG_PATH", str(path))

    log_turn(_record(warnings=["contact the customer at jane@example.com about this"]))

    record = _read_lines(path)[0]
    assert "jane@example.com" not in json.dumps(record["warnings"])


def test_pii_is_redacted_in_escalation_reason(tmp_path, monkeypatch):
    path = tmp_path / "turns.jsonl"
    monkeypatch.setenv("TURN_LOG_PATH", str(path))

    log_turn(_record(escalation_reason="caller's callback number is 555-123-4567"))

    record = _read_lines(path)[0]
    assert "555-123-4567" not in record["escalation_reason"]


def test_none_escalation_reason_stays_none(tmp_path, monkeypatch):
    path = tmp_path / "turns.jsonl"
    monkeypatch.setenv("TURN_LOG_PATH", str(path))

    log_turn(_record(escalation_reason=None))

    assert _read_lines(path)[0]["escalation_reason"] is None


def test_a_genuinely_unpicklable_value_does_not_drop_the_record(tmp_path, monkeypatch):
    """The old `object()`-based unserializable-value test deep-copies fine
    under asdict(), so it never exercised this: a value like a
    threading.Lock raises inside copy.deepcopy itself ('cannot pickle'),
    which would previously lose the whole record. vars() instead of
    asdict() must sidestep the deep copy entirely."""
    path = tmp_path / "turns.jsonl"
    monkeypatch.setenv("TURN_LOG_PATH", str(path))

    log_turn(_record(tool_calls=[{"name": "x", "input": {}, "output": {"lock": threading.Lock()}}]))

    assert len(_read_lines(path)) == 1


def test_the_default_json_fallback_is_also_redacted(tmp_path, monkeypatch):
    """default=str would write an object's raw __str__ into the log
    verbatim; the fallback must run it through redact_text too."""

    class _LeakyRepr:
        def __str__(self):
            return "contact jane@example.com"

    path = tmp_path / "turns.jsonl"
    monkeypatch.setenv("TURN_LOG_PATH", str(path))

    log_turn(_record(tool_calls=[{"name": "x", "input": {}, "output": _LeakyRepr()}]))

    blob = json.dumps(_read_lines(path)[0])
    assert "jane@example.com" not in blob


def test_escalation_id_field_round_trips(tmp_path, monkeypatch):
    path = tmp_path / "turns.jsonl"
    monkeypatch.setenv("TURN_LOG_PATH", str(path))

    log_turn(_record(escalation_id=42))

    assert _read_lines(path)[0]["escalation_id"] == 42
