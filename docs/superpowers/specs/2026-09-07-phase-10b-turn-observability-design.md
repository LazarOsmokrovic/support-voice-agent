# Phase 10b — Structured per-turn observability

## Context

`PROJECT_PLAN.md`'s Phase 10 lists "**Observability:** structured per-turn logs (transcript,
tool calls, latency, escalation events)". Phase 10 was decomposed into 10a–10e; 10a (model
boundary guardrails) is done and merged. This is 10b.

What exists today is not observability:

- Four ad-hoc loggers (`agent.core`, `agent.tools.escalation`, `agent.tools.notifications`,
  `guardrails.validators`) emitting prose strings at WARNING/INFO.
- Transports using bare `print()` for latency and reply previews
  (`transport/pipecat_processors.py:169-170`, `transport/voice_local.py`).
- Real per-turn signal — `llm_latency_seconds`, `warnings`, `end_reason`, and 10a's new
  `hedged` behavior — computed in `run_turn` and then **discarded at the transport
  boundary**. Voice transports measure STT and TTS latency, print it, and throw it away.

Nothing is queryable, nothing is machine-readable, and nothing survives the process.

**10b exists mostly to serve 10c.** The eval suite cannot measure what the grounding
detector actually does — its false-positive rate is currently unknown, and 10a's
`UNGROUNDED_REPLY_ESCALATION_THRESHOLD = 2` is admittedly a guess whose own code comment
argues 3 might be better. A per-turn record carrying `hedged` and `warnings` is the
instrument that turns that argument into a measurement, the same way eight hand-labelled
questions fixed Phase 3's RAG threshold instead of intuition.

Decisions taken during brainstorming, all confirmed by the project owner:

- **Sink: a JSONL file.** One JSON object per turn. Machine-readable, greppable, trivially
  consumed by 10c, no schema migration, and decoupled from the mock SQLite store.
- **Tool detail: names + inputs + redacted outputs.** Full debuggability — you can
  reconstruct why the agent said what it said — with outputs passing through
  `guardrails/pii.py` first. This is precisely the dependency that made 10a a prerequisite.
- **Redacting arbitrary tool output needs a new function** (`redact_structure`), added to
  `guardrails/pii.py` so there stays one definition of what redaction means.
- **Sessions get an ID.** There is none today, and without one a JSONL file is unreadable
  the moment two conversations interleave — which telephony does by design.
- **On by default**, to `logs/turns.jsonl`. A deliberate break from this project's
  optional-by-default convention; see below.
- **A `transport` label**, set by each of the four transports.

## Approach

Two alternatives were considered and rejected:

- **A JSON `logging.Handler`/formatter.** Superficially the "standard" answer, but it would
  mean untangling four existing prose loggers from structured output, and adding logging
  configuration for no gain over a direct writer. The records here are one specific shape,
  not general application logging.
- **A decorator or context manager wrapping `run_turn`.** Hides control flow and makes the
  emit point invisible at the call site — rejected for the same reason 10a rejected
  middleware around `Agent.send`.

**Chosen:** one small module, `observability/turn_log.py`, with a typed record and a single
writer, emitting once from `run_turn` — the one function that already orchestrates a turn
and holds nearly everything worth recording. A new top-level package mirrors the existing
`guardrails/` and `eval/` layout.

## On by default — a deliberate convention break

`ESCALATION_WEBHOOK_URL`, `TTS_BACKEND`, `EMBEDDING_BACKEND` and `VOYAGE_API_KEY` are all
optional-by-default: unset means the feature is a silent no-op. 10b deliberately does the
opposite and defaults **on**, writing to `logs/turns.jsonl` (gitignored).

The reasoning: observability that is off by default observes nothing, and the turns worth
having a record of are precisely the ones nobody anticipated — a hallucination, an
injection attempt, an escalation that fired wrongly. A default-off telemetry system is
reliably enabled only after the interesting event has already been lost.

`TURN_LOG_PATH=""` disables it; any other value overrides the path.

## Components

### `observability/turn_log.py` (new)

```python
@dataclass
class TurnRecord:
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

def log_turn(record: TurnRecord) -> None
```

A dataclass rather than fourteen keyword arguments, because the schema is the thing worth
documenting and it belongs in code rather than prose.

- **Redaction happens inside `log_turn`, not at the call site**, so a caller cannot forget
  it. `user_text` and `reply` go through `redact_text`; `tool_calls` through
  `redact_structure`.
- **Serialization:** `json.dumps(..., default=str)` with a `ts` field
  (`datetime.now(timezone.utc).isoformat()`) and `llm_latency_ms` rounded to an integer —
  milliseconds read better in a log than float seconds.
- **Writing:** open in append mode, write one line, close. Parent directory created if
  missing.
- **Concurrency:** a module-level `threading.Lock` around the append. The telephony
  transport handles multiple simultaneous calls in one process, and a record carrying tool
  outputs can exceed the size at which a POSIX append is atomic, so interleaved half-lines
  are a real possibility without one.
- **Never raises.** An unwritable path, a full disk, or an unserializable value must not
  break a call. Failures are caught, logged once via the module's own
  `logging.getLogger("observability.turn_log")`, and swallowed — the same discipline
  `guardrails/validators.py` follows, and for the same reason.
- Disabled (`TURN_LOG_PATH=""`) is a fast no-op: return before any formatting or redaction.

### `guardrails/pii.py` — one addition

```python
def redact_structure(value: Any) -> Any
```

Recursively redacts every string inside a nested structure — dicts, lists, tuples — leaving
dict *keys*, numbers, booleans and `None` untouched. Returns a copy; never mutates the
input, since the caller is handing us live tool output.

`redact_fields` operates on named fields of a flat mapping and is the right tool for a
handoff packet; tool outputs are arbitrary nested shapes and need this instead. Both live in
`pii.py` so there remains exactly one definition of what counts as PII here.

Because 10a's exemptions apply, order IDs, ISO dates and `TBA…US` tracking numbers survive —
so logged tool output stays useful for debugging rather than becoming a wall of
`[redacted-…]`.

### `agent/session.py` — session identity and the emit point

- `Session` gains `session_id: str` and `transport: str`. `create_session(customer_id,
  client=None, transport="unknown")` generates a `uuid4` hex.
- `run_turn` currently has **three** `return TurnOutcome(...)` sites (escalated,
  model-ended, normal). It is restructured to assign `outcome` in each branch and then log
  and return once at the tail. That single exit is what keeps the emit point from being
  duplicated three ways or silently missed on a future fourth branch.
- The `log_turn` call is wrapped so a telemetry failure cannot escape `run_turn`,
  belt-and-suspenders on top of `log_turn`'s own guarantee — the same precedent
  `create_handoff_packet` sets for `notify_escalation`.
- `hedged` is `bool(findings)`; `escalated` is `end_reason == "escalated"`.

### The four transports — one line each

`create_session(customer_id, transport="text_cli" | "voice_local" | "pipeline" |
"telephony")`. This is configuration, not business logic, so it does not breach CLAUDE.md
rule 5. `run_turn` cannot infer the channel, and "which channel did this happen on" is basic
observability — particularly once telephony and CLI turns share one file.

### `.gitignore`

`logs/` — turn records contain redacted customer conversations and must never be committed.

### `tests/conftest.py` — test isolation (found by pre-implementation review)

The autouse fixture added in 10a strips `ESCALATION_WEBHOOK_URL`/`_SECRET` so no test can
fire a real webhook. It must also set `TURN_LOG_PATH=""`.

Without it, this phase reproduces 10a's finding #4 exactly: the default path is **relative**
(`logs/turns.jsonl`), and every test that calls `run_turn` — across `test_session.py`,
`test_text_cli.py` and `test_pipecat_processors.py` — would append real records to the
repository's own log file. Test data polluting an operational log, silently, on every run.
Tests that want logging set the path explicitly with `monkeypatch.setenv`, which runs after
the fixture, exactly as the webhook tests already do.

### `transport/pipecat_processors.py` — the DTMF escalation path (found by pre-implementation review)

`_handle_dtmf_escalation` calls `create_handoff_packet` **directly**, bypassing `run_turn`
entirely — that is deliberate, and dates from Phase 9: pressing 0 is a deterministic safety
net that must work even when the model or the pipeline is misbehaving, so it runs no
`classify_turn` and no `run_turn`.

The consequence for this phase is that a "press 0 for a human" escalation would produce **no
record at all** — an observability hole precisely at the event most worth observing. This is
the same class of mistake as 10a's "redaction is applied at the two places" claim when there
were four: a path not enumerated.

So the DTMF handler emits a record too: `user_text="[DTMF] 0"`, `escalated=True`,
`escalation_reason="caller pressed 0 for a human"`, `hedged=False`, no tool calls, zero LLM
latency. One schema, one file, and a keypress escalation is legible alongside spoken ones.
This is the single exception to "transports change by one argument each", and it is
telemetry for a transport-level event rather than business logic, so CLAUDE.md rule 5 is
unaffected.

## Raw versus sanitized caller text

`user_text` in the record is the caller's **raw** text, not the sanitized string the model
received. This is deliberate: the log should show what the caller actually said, because
that is what an injection attempt looks like, and 10a's sanitizer neutralizes markers in a
way that would hide the attack from the record.

The transformation is not lost — when the sanitizer fires it appends a warning, and
`warnings` is part of the record, so a reader sees both that an attempt was made and that it
was neutralized. Logging both forms was considered and rejected as duplication for a case
the warnings already cover.

## What is deliberately NOT included

- **STT and TTS latency.** Both are measured in the voice transports, but TTS latency is not
  known until after `run_turn` has returned, so folding them in would mean either logging
  from four transports (duplicating the emit point this design exists to centralize) or
  emitting a second record type keyed by turn id. Neither earns its complexity yet. The
  audio-stage timings continue to be printed as they are today, and the limitation is
  documented rather than hidden. If 10c needs them, a second record type is the clean
  addition.
- **Log rotation, shipping, sampling.** A local JSONL file is the whole scope. YAGNI.
- **Changing the existing prose loggers or the transports' `print()` output.** The
  human-readable interactive experience stays exactly as it is; this adds a parallel
  machine-readable stream.

## Error handling

- Disabled → immediate return, no work done.
- Unwritable path / serialization failure → caught inside `log_turn`, logged once, swallowed.
- Anything unexpected escaping `log_turn` → caught by `run_turn`'s wrapper and surfaced as a
  warning on the turn rather than an exception.
- No `except BaseException` anywhere: `asyncio.CancelledError` must keep propagating or
  Pipecat barge-in breaks.

## Testing

| File | Covers |
|---|---|
| `tests/test_turn_log.py` (new) | a record writes exactly one parseable JSON line; every schema field present; `user_text`/`reply`/tool outputs redacted while a real seeded order ID survives; `TURN_LOG_PATH=""` writes nothing at all; an unwritable path does not raise; two records append rather than overwrite; `llm_latency_ms` is an integer |
| `tests/test_pii.py` | `redact_structure` on nested dicts/lists, non-string values untouched, dict keys untouched, input not mutated, real seeded identifiers preserved |
| `tests/test_session.py` | `create_session` assigns a unique `session_id` and stores the transport label; a turn emits exactly one record carrying the right `hedged`/`escalated`/`end_reason`; a `log_turn` failure surfaces as a warning and does not break the turn |

All offline; no network, no API keys. Test isolation uses `tmp_path` for the log file,
matching the `monkeypatch.setattr(mock_db, "DB_PATH", …)` convention already used
throughout.

## Checkpoint

**Automated:** the above green, and the full suite showing no regressions against the
current baseline (174 passed / 13 pre-existing API-key failures).

**Manual:** run a real conversation through `transport/text_cli.py` — including one turn
that escalates — then read `logs/turns.jsonl` and confirm the records are legible, the
conversation is reconstructable from them, PII is masked, and the order ID and dates are
still readable. That last check is the one this project has learned twice to do: Phase 11's
order-ID bug and 10a's date-destruction bug both survived every automated review and would
have been obvious in one glance at a real record.

## Files touched

**New:** `observability/turn_log.py`, `tests/test_turn_log.py`
**Modified:** `agent/session.py`, `guardrails/pii.py`, `tests/test_pii.py`,
`tests/test_session.py`, `.gitignore`, and one line in each of `transport/text_cli.py`,
`transport/voice_local.py`, `transport/pipeline.py`, `transport/telephony.py`
**Untouched:** `agent/core.py` (unchanged since Phase 4), `data/mock_db.py`, `eval/` (10c),
every other guardrail and tool module

## Self-review

- **Placeholders:** none — every component has a concrete signature, behavior and test plan.
- **Internal consistency:** the record schema, the emit point, the test matrix and the
  files-touched list all describe the same design.
- **Scope:** one coherent subsystem. The three things most likely to creep — audio latency,
  rotation, and rewriting existing loggers — are explicitly excluded above with reasons.
- **Ambiguity:** the two places this could be read two ways — the default-on decision and
  why the transports are touched at all — are argued explicitly rather than asserted.
