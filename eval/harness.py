"""Drives ONE scenario through the real agent — Phase 10c.

Shared VERBATIM by eval/record.py (live) and eval/run_eval.py (replay). That
sharing is the load-bearing structural decision of this design: it makes
"replay drives the same code paths recording did" true by construction
rather than by discipline. Anything either caller needs differently is
passed in (which client, which clock), never branched on here.

Approach B from the spec — fake the model, run everything else for real:
dispatch_tool, all seven tools, real SQLite on a fresh seeded temp database,
real Chroma, all three guardrails, EscalationTracker, log_turn. Faking the
tools too would be faster and perfectly hermetic, and would stop exercising
the code the scenarios exist to protect: PendingActionGate (CLAUDE.md rule
6, this project's most safety-critical invariant) lives INSIDE
issue_refund/book_appointment, and issue_refund's window check could be
deleted entirely and a transcript-replay suite would stay green.

The turn-log spy is a PASS-THROUGH, not a replacement. Monkeypatching
log_turn outright would replace the code under test, so a serialisation or
redaction bug would never be caught and PII would be unmeasurable. Appending
the record AND calling the real writer scores behaviour from the in-process
objects and PII from the file's real bytes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agent import session as session_module
from agent.session import close_session, create_session, run_turn
from agent.tools import policy_rag
from data import mock_db
from eval.replay import scenario_patch
from eval.scenarios import Scenario
from observability.turn_log import TurnRecord, log_turn


@dataclass(frozen=True)
class ObservedTurn:
    """What one turn actually produced, at record time and again at replay.

    Recording stores these; replay recomputes and diffs them. A mismatch is
    DRIFT — either the tools regressed (caught) or the environment shifted
    (also worth knowing). This is the only mechanism that verifies replay
    drives the same code paths, so `block_input_runtime_type` is carried
    deliberately: it is the tripwire for the dict-LIKE block.input bug the
    SDK-native reconstruction in eval/replay.py exists to prevent.
    """

    turn: int
    tool_calls: list[dict[str, Any]]
    grounding_flagged: bool
    hedge_spoken: bool
    escalation_reason: str | None
    end_reason: str | None
    block_input_runtime_type: str | None


@dataclass
class HarnessResult:
    scenario: str
    replies: list[str] = field(default_factory=list)
    observed: list[ObservedTurn] = field(default_factory=list)
    records: list[TurnRecord] = field(default_factory=list)
    log_lines: list[dict[str, Any]] = field(default_factory=list)
    db_path: Path = Path()
    close_error: str | None = None
    ticket_id: int | None = None
    # A scenario raising unexpectedly is caught here and reported, so one
    # broken scenario never zeroes the whole report.
    error: str | None = None


def observed_as_dicts(observed: list[ObservedTurn]) -> list[dict[str, Any]]:
    return [asdict(row) for row in observed]


def ensure_policies_ingested() -> int:
    """Build the Chroma collection if it is empty, and say how many chunks.

    data/chroma_db/ is gitignored (.gitignore:7), so a fresh clone and any CI
    has no vector store at all. The local MiniLM backend is free and keyless,
    so "runs with no API key" still holds — but there is a first-run ONNX
    download, and an ONNX/chromadb version bump changes embeddings, hence
    top-k, hence the policy_reference text, hence possibly the grounding
    numbers. This is the weakest link in the suite's determinism. The DRIFT
    outcome exists precisely so it surfaces loudly rather than silently
    shifting the headline metric.
    """
    collection = policy_rag._get_collection()
    if collection.count() == 0:
        return policy_rag.ingest_policies(collection=collection)
    return collection.count()


def _block_input_runtime_type(tool_calls: list[dict[str, Any]]) -> str | None:
    for call in tool_calls:
        return type(call.get("input")).__name__
    return None


async def run_scenario(
    scenario: Scenario,
    client: Any,
    frozen: datetime,
    workdir: Path,
    transport: str = "eval",
) -> HarnessResult:
    """Run one scenario end to end and return everything worth scoring."""
    workdir.mkdir(parents=True, exist_ok=True)
    db_path = workdir / f"{scenario.name}.db"
    log_path = workdir / f"{scenario.name}.turns.jsonl"
    result = HarnessResult(scenario=scenario.name, db_path=db_path)

    ensure_policies_ingested()

    saved_db_path = mock_db.DB_PATH
    saved_log_turn = session_module.log_turn
    captured: list[TurnRecord] = []

    def _spy(record: TurnRecord) -> None:
        # Pass-through: capture the exact object AND exercise the real
        # serialiser/redactor, because those are what PII scoring reads.
        captured.append(record)
        log_turn(record)

    try:
        mock_db.DB_PATH = db_path
        mock_db.reset_and_seed()
        session_module.log_turn = _spy

        with scenario_patch(client, frozen, log_path):
            session = create_session(scenario.customer_id, transport=transport)
            try:
                for index, user_text in enumerate(scenario.turns, start=1):
                    outcome = await run_turn(session, user_text)
                    result.replies.append(outcome.reply)
                    record = captured[-1]
                    result.observed.append(
                        ObservedTurn(
                            turn=index,
                            tool_calls=record.tool_calls,
                            grounding_flagged=record.grounding_flagged,
                            hedge_spoken=record.hedge_spoken,
                            escalation_reason=record.escalation_reason,
                            end_reason=record.end_reason,
                            block_input_runtime_type=_block_input_runtime_type(record.tool_calls),
                        )
                    )
                    if outcome.ended:
                        break
                if scenario.close_session:
                    close_result = await close_session(session)
                    result.close_error = close_result.error
                    result.ticket_id = close_result.ticket_id
            except Exception as exc:  # noqa: BLE001 — one broken scenario must not zero the report
                result.error = f"{type(exc).__name__}: {exc}"
    finally:
        session_module.log_turn = saved_log_turn
        mock_db.DB_PATH = saved_db_path

    result.records = captured
    if log_path.exists():
        result.log_lines = [
            json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    return result
