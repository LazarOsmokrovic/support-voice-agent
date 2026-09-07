# Phase 10b — Structured Per-Turn Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit one structured JSON record per turn — transcript, tool calls, latency, guardrail findings and escalation events — so a conversation can be reconstructed after the fact and so Phase 10c can measure what the grounding detector actually does.

**Architecture:** One new module, `observability/turn_log.py`, with a typed `TurnRecord` and a single writer, emitted once from `agent/session.py::run_turn`. Redaction happens inside the writer so a caller cannot forget it. No logging framework, no decorator.

**Tech Stack:** Python 3.12+, stdlib only (`json`, `dataclasses`, `threading`, `uuid`, `logging`, `pathlib`). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-07-phase-10b-turn-observability-design.md`

## Global Constraints

- **`agent/core.py` must NOT change.** Its last edit was Phase 4; it has since survived local voice, Pipecat, Twilio, notifications and the 10a guardrails untouched. That is the project's evidence its I/O decoupling held.
- **Transports change by exactly one argument each** — a `transport=` label on their existing `create_session(...)` call. That is configuration, not business logic. No other transport edit is in scope.
- **Logging must never break a call.** `log_turn` never raises; `run_turn` wraps it anyway (the precedent `create_handoff_packet` sets for `notify_escalation`). Never catch `BaseException` — `asyncio.CancelledError` must keep propagating or Pipecat barge-in breaks.
- **Redaction is not optional and not the caller's job.** It happens inside `log_turn`.
- **`TurnOutcome`'s shape does not change.** No new fields.
- The project uses a mock SQLite store of fictional data by deliberate decision; this phase adds no database concerns at all.
- Tests run offline. Run with `python -m pytest` from the repo root — bare `pytest` fails with `ModuleNotFoundError: No module named 'transport'` (pre-existing project quirk, not to be fixed).
- Baseline before this work: **174 passed, 13 failed.** The 13 are long-standing live tests gated on stale API keys — unrelated; do not fix them and do not describe the suite as fully green.

---

### Task 1: `redact_structure` in `guardrails/pii.py`

**Files:**
- Modify: `guardrails/pii.py`
- Test: `tests/test_pii.py`

**Interfaces:**
- Consumes: the module's existing `redact_text`.
- Produces: `redact_structure(value: Any) -> Any` — consumed by Task 2's writer.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_pii.py`:

```python
def test_redact_structure_redacts_strings_nested_in_dicts_and_lists():
    value = {
        "message": "Email jane@example.com",
        "items": ["call 555-123-4567", {"note": "card 4111 1111 1111 1111"}],
    }
    result = redact_structure(value)
    assert "[redacted-email]" in result["message"]
    assert "[redacted-phone]" in result["items"][0]
    assert "[redacted-number]" in result["items"][1]["note"]


def test_redact_structure_leaves_non_strings_untouched():
    value = {"found": True, "price": 34.99, "quantity": 1, "tracking": None}
    assert redact_structure(value) == value


def test_redact_structure_leaves_dict_keys_untouched():
    value = {"jane@example.com": "ordinary text"}
    assert list(redact_structure(value).keys()) == ["jane@example.com"]


def test_redact_structure_does_not_mutate_its_input():
    value = {"message": "Email jane@example.com"}
    redact_structure(value)
    assert value["message"] == "Email jane@example.com"


def test_redact_structure_preserves_real_seeded_identifiers():
    order_id, delivery, tracking = ORDERS[0][0], ORDERS[0][7], ORDERS[0][8]
    value = {"order_id": order_id, "estimated_delivery": delivery, "tracking_number": tracking}
    assert redact_structure(value) == value


def test_redact_structure_handles_a_bare_string_and_a_bare_scalar():
    assert redact_structure("Email jane@example.com") == "Email [redacted-email]"
    assert redact_structure(42) == 42
```

`ORDERS` is already imported at the top of this file from Task 1 of Phase 10a; confirm and add `redact_structure` to the existing `guardrails.pii` import line.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pii.py -v`
Expected: FAIL with `ImportError: cannot import name 'redact_structure'`

- [ ] **Step 3: Implement**

Add to `guardrails/pii.py`, after `redact_fields`:

```python
def redact_structure(value: Any) -> Any:
    """Recursively redact every string inside a nested structure.

    `redact_fields` handles a flat mapping's named fields, which is the right
    shape for a handoff packet. Tool outputs are arbitrary nested dicts and
    lists (policy chunks, order rows, refund results), so anything logged from
    them needs this instead — see observability/turn_log.py.

    Dict KEYS are deliberately left alone: they are field names chosen by this
    codebase, never customer-supplied, and redacting them would make a log
    record unreadable. Non-string scalars pass through untouched.

    Returns a copy; never mutates the input, because callers hand us live tool
    output that is still in use elsewhere on the turn.
    """
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {key: redact_structure(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_structure(item) for item in value]
    return value
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pii.py -v`
Expected: all pass, including every pre-existing test in the file unmodified.

- [ ] **Step 5: Commit**

```bash
git add guardrails/pii.py tests/test_pii.py
git commit -m "Phase 10b: add redact_structure for nested tool output

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: `observability/turn_log.py`

**Files:**
- Create: `observability/__init__.py` (empty), `observability/turn_log.py`
- Test: `tests/test_turn_log.py` (new)

**Interfaces:**
- Consumes: `redact_text`, `redact_structure` (Task 1).
- Produces: `TurnRecord` (dataclass) and `log_turn(record: TurnRecord) -> None` — consumed by Task 3.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_turn_log.py`:

```python
"""Phase 10b: observability/turn_log.py — one structured JSON record per turn.

Every test isolates the log file with tmp_path and TURN_LOG_PATH, the same way
the DB tests isolate mock_db.DB_PATH.
"""

from __future__ import annotations

import json

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
        hedged=False,
        tool_calls=[],
        llm_latency_seconds=1.8404,
        warnings=[],
        escalated=False,
        escalation_reason=None,
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
        "hedged", "tool_calls", "llm_latency_ms", "warnings", "escalated",
        "escalation_reason", "ended", "end_reason",
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_turn_log.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'observability'`

- [ ] **Step 3: Create the package**

Create an empty `observability/__init__.py`, then `observability/turn_log.py`:

```python
"""One structured JSON record per conversation turn — Phase 10b.

PROJECT_PLAN.md asks for "structured per-turn logs (transcript, tool calls,
latency, escalation events)". Before this, that signal was computed in
run_turn and then discarded at the transport boundary: four ad-hoc loggers
emitting prose, and transports printing latency with print(). Nothing was
machine-readable and nothing survived the process.

This mostly exists to serve Phase 10c. The eval suite cannot measure what the
grounding detector actually does — its false-positive rate is unknown, and
10a's UNGROUNDED_REPLY_ESCALATION_THRESHOLD is admittedly a guess. The
`hedged` and `warnings` fields are the instrument that turns that argument
into a measurement.

Deliberately ON by default (logs/turns.jsonl), unlike every other optional
integration in this project. Observability that is off by default observes
nothing, and the turns worth having a record of are exactly the ones nobody
anticipated. TURN_LOG_PATH="" disables it.

Redaction happens HERE rather than at the call site, so a caller cannot
forget it — every string in the transcript and in tool output passes through
guardrails/pii.py first.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from guardrails.pii import redact_structure, redact_text

logger = logging.getLogger("observability.turn_log")

DEFAULT_TURN_LOG_PATH = "logs/turns.jsonl"

# The telephony transport handles multiple simultaneous calls in one process,
# and a record carrying tool output can exceed the size at which a POSIX
# append is atomic — without this, two concurrent calls can interleave
# half-lines and corrupt the file.
_write_lock = threading.Lock()


@dataclass
class TurnRecord:
    """One turn's worth of telemetry. The dataclass IS the schema — it is the
    thing worth documenting, and it belongs in code rather than prose.
    """

    session_id: str
    customer_id: str
    transport: str
    turn: int
    user_text: str
    reply: str
    hedged: bool
    tool_calls: list[dict[str, Any]]
    llm_latency_seconds: float
    warnings: list[str]
    escalated: bool
    escalation_reason: str | None
    ended: bool
    end_reason: str | None


def _log_path() -> Path | None:
    """None means disabled. TURN_LOG_PATH="" is an explicit off switch."""
    raw = os.getenv("TURN_LOG_PATH", DEFAULT_TURN_LOG_PATH)
    return Path(raw) if raw else None


def _serialize(record: TurnRecord) -> str:
    fields = asdict(record)
    latency = fields.pop("llm_latency_seconds")
    return json.dumps(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            **fields,
            "user_text": redact_text(record.user_text),
            "reply": redact_text(record.reply),
            "tool_calls": redact_structure(record.tool_calls),
            "llm_latency_ms": round(latency * 1000),
        },
        default=str,
    )


def log_turn(record: TurnRecord) -> None:
    """Append one redacted JSON line for this turn. Never raises.

    A telemetry failure — an unwritable path, a full disk, a value json can't
    serialize — must never break a live call, so everything is caught, logged
    once, and swallowed. Same discipline guardrails/validators.py follows.
    """
    path = _log_path()
    if path is None:
        return

    try:
        line = _serialize(record)
        with _write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception as exc:  # noqa: BLE001 — telemetry must never break a call
        logger.warning("turn log write failed, dropping this record: %s", exc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_turn_log.py -v`
Expected: all 11 PASS

- [ ] **Step 5: Commit**

```bash
git add observability/ tests/test_turn_log.py
git commit -m "Phase 10b: add structured per-turn JSONL logging

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Session identity and the emit point

**Files:**
- Modify: `agent/session.py` (`Session`, `create_session`, `run_turn`)
- Test: `tests/test_session.py`

**Interfaces:**
- Consumes: `TurnRecord`, `log_turn` (Task 2).
- Produces: `Session.session_id`, `Session.transport`, and `create_session(customer_id, client=None, transport="unknown")` — consumed by Task 4's transports.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_session.py`:

```python
def test_create_session_assigns_a_unique_session_id_and_transport():
    first = create_session("CUST-1001", transport="text_cli")
    second = create_session("CUST-1001", transport="text_cli")

    assert first.session_id and second.session_id
    assert first.session_id != second.session_id
    assert first.transport == "text_cli"


def test_create_session_defaults_the_transport_label():
    assert create_session("CUST-1001").transport == "unknown"


@pytest.mark.asyncio
async def test_run_turn_emits_exactly_one_turn_record(tmp_path, monkeypatch):
    path = tmp_path / "turns.jsonl"
    monkeypatch.setenv("TURN_LOG_PATH", str(path))
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=_text_response("Happy to help!"))
    monkeypatch.setattr(escalation, "classify_turn", AsyncMock(return_value=_calm_classification()))
    session = create_session("CUST-1001", client=fake_client, transport="text_cli")

    await run_turn(session, "Hi there")

    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0]["session_id"] == session.session_id
    assert lines[0]["transport"] == "text_cli"
    assert lines[0]["hedged"] is False
    assert lines[0]["escalated"] is False
    assert lines[0]["end_reason"] is None


@pytest.mark.asyncio
async def test_a_turn_log_failure_becomes_a_warning_and_does_not_break_the_turn(monkeypatch):
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=_text_response("Happy to help!"))
    monkeypatch.setattr(escalation, "classify_turn", AsyncMock(return_value=_calm_classification()))
    monkeypatch.setattr(session_module, "log_turn", MagicMock(side_effect=RuntimeError("disk on fire")))
    session = create_session("CUST-1001", client=fake_client)

    outcome = await run_turn(session, "Hi there")

    assert outcome.reply == "Happy to help!"
    assert any("disk on fire" in warning for warning in outcome.warnings)
```

Add to the imports at the top of `tests/test_session.py`:

```python
import json

from agent import session as session_module
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_session.py -v -k "session_id or transport or turn_record or turn_log_failure"`
Expected: FAIL — `create_session()` takes no `transport` argument, and `Session` has no `session_id`.

- [ ] **Step 3: Add session identity**

In `agent/session.py`, add to the imports:

```python
import uuid

from observability.turn_log import TurnRecord, log_turn
```

Add two fields to the `Session` dataclass, after `customer_id`:

```python
    session_id: str
    transport: str
```

and update `create_session`:

```python
def create_session(customer_id: str, client: Any | None = None, transport: str = "unknown") -> Session:
    """`client` is only for tests — real callers never pass it, and Agent
    creates its own anthropic.AsyncAnthropic() by default, same as every
    other place in this project that accepts an injectable client.

    `transport` labels which I/O layer is driving this conversation, purely
    so per-turn records (observability/turn_log.py) say which channel a turn
    came from. run_turn cannot infer it, and once telephony and CLI turns
    share one log file it is the difference between a readable record and an
    ambiguous one. It is a label, not behavior — nothing branches on it.
    """
    dispatch_tool, handlers, gates = build_dispatch_tool(customer_id)
    agent = Agent(system=SYSTEM_PROMPT, tools=TOOLS, tool_executor=dispatch_tool, client=client)
    return Session(
        customer_id=customer_id,
        session_id=uuid.uuid4().hex,
        transport=transport,
        agent=agent,
        tracker=escalation.EscalationTracker(),
        gates=gates,
        handlers=handlers,
    )
```

- [ ] **Step 4: Restructure `run_turn` to a single exit and emit once**

`run_turn` currently has three `return TurnOutcome(...)` sites (`agent/session.py:275`, `:285`, `:289`). Replace each `return TurnOutcome(...)` with `outcome = TurnOutcome(...)` and restructure the tail into one exit, so the emit point cannot be duplicated three ways or missed by a future fourth branch:

```python
    if reason:
        try:
            packet = await escalation.create_handoff_packet(session.customer_id, session.agent.messages, reason)
            notice = f"I'm connecting you with a human agent — {reason}. (handoff #{packet['escalation_id']})"
        except Exception as exc:  # noqa: BLE001 — exit path must never crash on this
            notice = None
            warnings.append(f"Escalation triggered ({reason}) but the handoff packet couldn't be logged: {exc}")
        outcome = TurnOutcome(
            reply=reply,
            ended=True,
            end_reason="escalated",
            notice=notice,
            llm_latency_seconds=llm_latency,
            warnings=warnings,
        )
    elif should_end_session(result.tool_calls):
        outcome = TurnOutcome(
            reply=reply, ended=True, end_reason="model_ended", llm_latency_seconds=llm_latency, warnings=warnings
        )
    else:
        outcome = TurnOutcome(reply=reply, llm_latency_seconds=llm_latency, warnings=warnings)

    # One emit point, at the single exit. log_turn already guarantees it never
    # raises; this wrapper is belt-and-suspenders on top of that, the same
    # precedent create_handoff_packet sets for notify_escalation — telemetry
    # must never be the thing that breaks a live call.
    try:
        log_turn(
            TurnRecord(
                session_id=session.session_id,
                customer_id=session.customer_id,
                transport=session.transport,
                turn=session.gates.refunds.turn,
                user_text=user_text,
                reply=outcome.reply,
                hedged=bool(findings),
                tool_calls=result.tool_calls,
                llm_latency_seconds=llm_latency,
                warnings=outcome.warnings,
                escalated=outcome.end_reason == "escalated",
                escalation_reason=reason,
                ended=outcome.ended,
                end_reason=outcome.end_reason,
            )
        )
    except Exception as exc:  # noqa: BLE001 — telemetry must never break a turn
        outcome.warnings.append(f"Could not write the turn log this turn: {exc}")

    return outcome
```

Note `user_text` is the caller's **raw** text — `log_turn` redacts it, and the record should reflect what the caller actually said rather than the sanitized form the model saw. Note also that `turn` reads the session's own gate counter, which `advance_turn()` already incremented at the top of `run_turn`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_session.py tests/test_text_cli.py -v`
Expected: all pass, including every pre-existing test in both files unmodified. If a pre-existing test needed editing, stop and report — that would mean behavior changed beyond intent.

- [ ] **Step 6: Commit**

```bash
git add agent/session.py tests/test_session.py
git commit -m "Phase 10b: give sessions an id and emit one record per turn

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Transport labels and gitignore

**Files:**
- Modify: `transport/text_cli.py:24`, `transport/voice_local.py:133`, `transport/pipeline.py:35`, `transport/telephony.py:130` — one argument each
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `create_session(..., transport=...)` (Task 3).
- Produces: nothing.

- [ ] **Step 1: Label each transport**

One argument per call site, nothing else in these files changes:

| File | Change |
|---|---|
| `transport/text_cli.py:24` | `create_session(customer_id, transport="text_cli")` |
| `transport/voice_local.py:133` | `create_session(customer_id, transport="voice_local")` |
| `transport/pipeline.py:35` | `create_session(customer_id, transport="pipeline")` |
| `transport/telephony.py:130` | `create_session(DEFAULT_CUSTOMER_ID, transport="telephony")` |

- [ ] **Step 2: Ignore the log directory**

Add to `.gitignore`, next to the other generated-data entries:

```
logs/
```

Turn records contain redacted customer conversations. They are local operational data and must never be committed.

- [ ] **Step 3: Verify the labels reach the log**

Run: `python -m pytest -q`
Expected: no regressions against the 174-passed baseline.

Then confirm by hand that a real turn is labelled — from the repo root, with the venv active:

```bash
printf 'CUST-1001\nquit\n' | python -m transport.text_cli >/dev/null 2>&1
tail -1 logs/turns.jsonl
```

Expected: no record at all (quitting without a turn produces none). Then run one that does take a turn, if an API key is available; otherwise note in the report that the label was verified by unit test only.

- [ ] **Step 4: Commit**

```bash
git add transport/text_cli.py transport/voice_local.py transport/pipeline.py transport/telephony.py .gitignore
git commit -m "Phase 10b: label each transport in its session, ignore logs/

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: Docs

**Files:**
- Modify: `PROJECT_PLAN.md` (a `### Phase 10b` subsection under the existing Phase 10 decomposition), `PROGRESS.md` (the 10b row), `README.md` (a Phase 10b narrative section)

**Interfaces:** none — documentation only.

- [ ] **Step 1: `PROJECT_PLAN.md`**

Add a `### Phase 10b` subsection alongside 10a's, covering: the JSONL record and its schema; why redaction happens inside the writer rather than at the call site; the default-on decision and why it breaks this project's optional-by-default convention; the STT/TTS latency exclusion and why; and a bolded **Checkpoint:** line (automated tests, plus the manual read of a real `logs/turns.jsonl`).

- [ ] **Step 2: `PROGRESS.md`**

Update the 10b row from `Not started` to `Done` with the date and a one-line note. Keep it terse — one row, no narrative.

- [ ] **Step 3: `README.md`**

A Phase 10b section matching the existing per-phase style (`### <module>` subsections, `### Tests`, `### Checkpoint result`). Cover: the record schema; that `hedged` and `warnings` are what 10c will measure the grounding detector with; `redact_structure` and why `redact_fields` was the wrong shape; the session id and transport label; the single-exit restructure of `run_turn` and why one emit point matters; the concurrency lock and the telephony reason for it; the default-on break from convention; and — honestly — whether the manual checkpoint was run.

Verify all test counts by running the suite; do not copy numbers from this plan.

- [ ] **Step 4: Full suite**

Run: `python -m pytest -q`
Expected: unchanged from Task 4.

- [ ] **Step 5: Commit**

```bash
git add PROJECT_PLAN.md PROGRESS.md README.md
git commit -m "Phase 10b: docs — structured per-turn observability

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** every component in the spec maps to a task — `redact_structure` → Task 1, the writer and schema → Task 2, session identity and the emit point → Task 3, transport labels and gitignore → Task 4, docs → Task 5. The spec's explicit exclusions (STT/TTS latency, rotation, rewriting existing loggers) appear in no task, by design.

**Placeholder scan:** no TBDs. Every code step carries the literal code; every test step the literal test.

**Type consistency:** `redact_structure(Any) -> Any` is defined in Task 1 and consumed in Task 2's `_serialize`. `TurnRecord` and `log_turn(TurnRecord) -> None` are defined in Task 2 and constructed in Task 3 with exactly the fourteen fields the dataclass declares. `create_session(customer_id, client=None, transport="unknown")` is defined in Task 3 and called with the keyword in Task 4; the default keeps every existing caller and test valid.

**Ordering risk checked:** `transport` is added as the **third** parameter of `create_session`, after `client`. Existing calls pass `customer_id` positionally and `client` by keyword or not at all (`transport/*.py` ×4, and the tests), so no positional call breaks — Task 3's Step 5 test run over `test_session.py` and `test_text_cli.py` is what confirms it.

**One thing deliberately not centralized:** `run_turn` reads `session.gates.refunds.turn` for the turn number, reaching into a confirmation gate for a counter. That is pre-existing (10a's hedge index does the same) and a real smell, but extracting a session-level turn counter is a refactor this phase does not need. Noted so a reviewer flags it as known rather than new.
