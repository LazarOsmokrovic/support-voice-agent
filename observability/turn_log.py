"""One structured JSON record per conversation turn — Phase 10b.

PROJECT_PLAN.md asks for "structured per-turn logs (transcript, tool calls,
latency, escalation events)". Before this, that signal was computed in
run_turn and then discarded at the transport boundary: four ad-hoc loggers
emitting prose, and transports printing latency with print(). Nothing was
machine-readable and nothing survived the process.

This mostly exists to serve Phase 10c. The eval suite cannot measure what the
grounding detector actually does — its false-positive rate is unknown, and
10a's UNGROUNDED_REPLY_ESCALATION_THRESHOLD is admittedly a guess. The
`grounding_flagged`, `hedge_spoken`, and `warnings` fields are the instrument
that turns that argument into a measurement.

`grounding_flagged` and `hedge_spoken` used to be one field, `hedged`. They
were split because detection and action are genuinely different events: a
turn that proposed a refund/booking confirmation is still flagged by the
detector (`grounding_flagged=True`) but the hedge is deliberately NOT
substituted for it (`hedge_spoken=False`), because swapping out the real
reply would silently drop the confirmation the caller needs to act on. A
single `hedged` field would have conflated "the detector fired" with "we
said the canned hedge instead", which would have made 10c overcount how
often the detector actually changes what gets spoken.

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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from guardrails.pii import redact_structure, redact_text

logger = logging.getLogger("observability.turn_log")

DEFAULT_TURN_LOG_PATH = "logs/turns.jsonl"

# log_turn() contains no `await`, so within one event loop a single call to it
# already runs start-to-finish without yielding — it cannot itself be
# interleaved with another call to log_turn(). This lock is insurance for the
# case that guarantee doesn't cover: multiple OS threads or worker processes
# (e.g. telephony handling several concurrent calls) appending to the same
# file, where a record carrying tool output can exceed the size at which a
# POSIX append is atomic.
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
    original_reply: str | None
    grounding_flagged: bool
    hedge_spoken: bool
    tool_calls: list[dict[str, Any]]
    llm_latency_seconds: float
    warnings: list[str]
    escalated: bool
    escalation_reason: str | None
    escalation_id: int | None
    ended: bool
    end_reason: str | None
    # Phase 12 Task 5 (F-8). Last field, and the only one with a default:
    # every preceding field is non-default, so it cannot go anywhere else.
    # transport/pipecat_processors.py:249 (a file this plan forbids editing)
    # constructs a TurnRecord on the DTMF path and on the turn-failure path
    # without ever knowing this field exists — the default is what keeps
    # those two construction sites from raising TypeError on every call.
    # A reason that only OFFERED (a suggested trigger the customer hasn't
    # accepted) — never a reason that opened or amended a handover, which
    # belongs in `escalation_reason` instead. See agent/session.py's run_turn
    # for why the two must not be conflated: eval/scoring.py's
    # score_escalation treats any row with an escalation_reason as "this
    # scenario escalated", and folding offers into that field would make
    # every never-escalates scenario that merely offered look escalated.
    escalation_offered: str | None = None


def _log_path() -> Path | None:
    """None means disabled. TURN_LOG_PATH="" is an explicit off switch."""
    raw = os.getenv("TURN_LOG_PATH", DEFAULT_TURN_LOG_PATH)
    return Path(raw) if raw else None


def _serialize(record: TurnRecord) -> str:
    # vars() (a shallow copy of the instance's own __dict__), not asdict():
    # asdict() deep-copies every field, and copy.deepcopy chokes on values
    # like a threading.Lock ("cannot pickle") that a tool output could
    # legitimately be carrying — losing the whole record where a shallow copy
    # plus the redacting `default=` below would have coped.
    fields = dict(vars(record))
    latency = fields.pop("llm_latency_seconds")
    return json.dumps(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            **fields,
            "user_text": redact_text(record.user_text),
            "reply": redact_text(record.reply),
            "original_reply": redact_text(record.original_reply) if record.original_reply is not None else None,
            "tool_calls": redact_structure(record.tool_calls),
            "warnings": redact_structure(record.warnings),
            "escalation_reason": redact_text(record.escalation_reason)
            if record.escalation_reason is not None
            else None,
            "llm_latency_ms": round(latency * 1000),
        },
        # Anything json can't serialize natively falls back to str(obj) — and
        # an object's own __str__ can just as easily contain PII as any other
        # string reaching this file, so redact that fallback too rather than
        # writing it verbatim.
        default=lambda o: redact_text(str(o)),
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
            try:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except FileNotFoundError:
                # The common case is an already-existing parent directory, so
                # mkdir only runs on the (rare) miss instead of every call —
                # one retry; if that also fails it falls into the same
                # catch-log-swallow path as any other failure below.
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
    except Exception as exc:  # noqa: BLE001 — telemetry must never break a call
        logger.warning("turn log write failed, dropping this record: %s", exc)
