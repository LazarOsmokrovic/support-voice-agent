# Phase 11 — AI automation: escalation notifications (n8n)

## Context

This project's `create_handoff_packet` (`agent/tools/escalation.py:228`) builds a
structured handoff packet whenever `EscalationTracker` decides a human needs to take
over, then only writes that packet to SQLite (`log_escalation`). No human is ever
actually notified — Phase 10's "real warm handoff" is about transferring the *phone
call*; it says nothing about alerting a person that a handoff has happened at all. That
is the actual gap this phase closes.

Built as its own phase — **Phase 11**, positioned after Phase 10 in `PROJECT_PLAN.md`/
`PROGRESS.md` but **not dependent on Phase 10 being started**. This is a deliberate,
explicit exception to CLAUDE.md rule 1 ("implement one phase at a time, in the order
given"), made at the project owner's direction rather than a silent deviation — recorded
here and in the phase's own `PROJECT_PLAN.md` section, not by editing the general rule.

Decisions locked in (via `AskUserQuestion`, this session):
- **Trigger scope:** escalations only, not every closed ticket. High-value refunds
  already escalate through this same path (Phase 6's `escalate: true` convention), so
  they're covered without any extra work.
- **Target platform:** n8n, self-hosted — matches this project's own-your-stack pattern
  (local SQLite, local Chroma) and is free to actually run for verifying the checkpoint.

## Approach

Three options were weighed:

1. **Fire-and-forget background task.** Zero added latency on the escalation turn, but
   breaks from this codebase's consistent "plain `await` chain, no background tasks"
   style, and is harder to test deterministically (task lifetime, no synchronous return
   value to assert on).
2. **Durable outbox queue** — a new table plus a separate poller/worker with its own
   retry loop. Genuinely production-grade (survives a process crash mid-delivery), but
   heavy new machinery for a path `EscalationTracker`'s thresholds already keep
   low-volume. Rejected as YAGNI for this project's scope.
3. **Chosen: a synchronous `await` inside `create_handoff_packet`, with a small bounded
   retry+backoff**, run immediately after `log_escalation` persists the row. Matches the
   existing pattern exactly — the same function already makes one synchronous LLM call
   before this point — and is trivially testable with `pytest-httpx`, the same way
   `tests/test_tts.py` already tests the swappable TTS backends. The added latency is
   capped (see Components) and lands at a point the call is already ending.

## Components

### New file: `agent/tools/notifications.py`

`async def notify_escalation(packet: dict, *, client: httpx.AsyncClient | None = None) -> bool`

- **Config:** reads `ESCALATION_WEBHOOK_URL` from env. Unset → silent no-op, returns
  `False` immediately, no HTTP call at all — the same optional-by-default convention as
  `TTS_BACKEND`/`EMBEDDING_BACKEND`/`VOYAGE_API_KEY`. A dev machine with nothing
  configured is a normal, fully-working state.
- **Redaction (new, intentionally scoped to this file only):** before serializing,
  masks emails, phone-like digit sequences, and card-like digit runs (13-19 digits) in
  the packet's free-text fields (`customer_intent`, `conversation_summary`,
  `verified_account_info`, `actions_taken`) via three small regexes. This is
  deliberately *not* Phase 10's `guardrails/pii.py` (left untouched) — a self-contained
  pass whose only job is protecting this one outbound payload, so this phase carries
  zero dependency on Phase 10. Documented as a narrower tool than Phase 10's eventual
  real redaction pipeline, not a substitute for it.
- **Signing:** if `ESCALATION_WEBHOOK_SECRET` is set, HMAC-SHA256-signs the serialized
  JSON body and sends it as `X-Signature-256` — the mirror image of what
  `transport/telephony.py` already does for *inbound* Twilio requests via
  `RequestValidator`, applied outbound this time. Unset → header omitted; called out in
  the README as reduced security, never silently presented as secure.
- **Delivery:** up to 3 attempts, 5s timeout each. Retries only on a connection/timeout
  error or a 5xx response (transient); a 4xx (bad URL/secret/payload) is treated as
  permanent and not retried. Backoff: 0.5s after attempt 1, 1.5s after attempt 2. The
  whole function **never raises** — a broken or misconfigured webhook must never affect
  the escalation itself completing. Logs through the existing `logging.getLogger`
  convention (`agent.tools.notifications`), matching every other module.
- Includes `escalation_id` (already unique, from `log_escalation`) in the payload as a
  natural idempotency key, in case a retry succeeds after an earlier attempt's response
  was lost in transit.

### `agent/tools/escalation.py` — two additions

- `mark_notified(escalation_id, delivered, notified_at=None)` — one `UPDATE escalations
  SET notified = ?, notified_at = ? WHERE escalation_id = ?`, mirroring
  `log_escalation`'s existing style exactly.
- `create_handoff_packet` gets one new step, right after assembling the packet: call
  `notify_escalation(packet)` inside its own try/except (belt-and-suspenders on top of
  `notify_escalation`'s own internal safety), then `mark_notified(...)` with the
  outcome. **The packet is still returned and still logged even if notification fails
  or raises** — persistence must never depend on delivery succeeding. No signature
  change to `create_handoff_packet`, so its callers (`agent/session.py::run_turn`,
  `transport/pipecat_processors.py`'s DTMF handler) need zero changes — CLAUDE.md
  rule 5's decoupling holds exactly: nothing in `transport/`, `agent/core.py`, or
  `agent/session.py` changes at all.

### `data/mock_db.py` — schema addition

Two new columns on the existing `escalations` table: `notified INTEGER NOT NULL
DEFAULT 0` and `notified_at TEXT`. This project has no migration system — every prior
phase added tables/columns the same way (a `CREATE TABLE IF NOT EXISTS` edit, with the
"the `.db` file is gitignored, regenerated locally" convention already documented in
`PROGRESS.md`'s Phase 0 entry). Same here: running `python -m data.mock_db` after
pulling this phase picks up the new columns via a fresh `reset_and_seed()`. Called out
explicitly in the phase's `PROGRESS.md` note, not left implicit.

## Data flow

```
EscalationTracker fires
  -> create_handoff_packet(...)
       -> _infer_handoff_fields(...)      [existing, unchanged]
       -> log_escalation(...)             [existing, unchanged]
       -> notify_escalation(packet)       [NEW]
            -> redact free-text fields
            -> sign body if secret configured
            -> POST to ESCALATION_WEBHOOK_URL, up to 3 attempts
            -> return True/False, never raises
       -> mark_notified(escalation_id, delivered)   [NEW]
       -> return packet   [unchanged shape]
```

## Error handling

- No `ESCALATION_WEBHOOK_URL`: no-op, `notified=0` recorded, nothing else changes.
- Webhook unreachable / times out / 5xx: retried per the backoff above, then gives up
  quietly — `notified=0`, a warning logged, the customer-facing handoff notice and the
  `escalations` row completely unaffected.
- Webhook returns 4xx: logged once, not retried, `notified=0`.
- Anything else unexpected inside `notify_escalation`: caught, logged, treated as a
  failed delivery — never propagates into `create_handoff_packet`.

## Testing

New `tests/test_notifications.py` (mirrors `tests/test_tts.py`'s `pytest-httpx`
pattern):
- redaction masks emails/phones/card-like sequences; ordinary text passes through
  untouched.
- signing: HMAC computed correctly over the exact serialized body.
- no `ESCALATION_WEBHOOK_URL` → returns `False` immediately, zero HTTP requests made.
- succeeds on the first attempt (2xx).
- succeeds after one retry (first response 503, second 200) — asserts the backoff sleep
  happened (mocked) and exactly 2 requests were made.
- exhausts all 3 attempts and returns `False` on persistent 5xx/timeout — never raises.
- a 4xx response is not retried (exactly 1 request made).
- the signature header is present and correctly computed when the secret is set, absent
  when it isn't.

`tests/test_escalation.py` new cases:
- `create_handoff_packet` calls `mark_notified` with the right `delivered` value,
  monkeypatching `notifications.notify_escalation` directly (same style
  `tests/test_session.py` already uses for `escalation.classify_turn`) — one case for
  success, one for a raised exception, confirming the packet is still returned and
  still logged in both cases.

## Docs to update

- `PROJECT_PLAN.md`: new `## Phase 11 — AI automation: escalation notifications (n8n)`
  section, formatted like the existing phase sections (bullets + Checkpoint), stating
  plainly that it depends only on Phase 4 (escalation) being Done, not Phase 10, and
  naming the CLAUDE.md rule 1 exception explicitly.
- `PROGRESS.md`: new row, `Phase 11 | AI automation... | Not started | |`, positioned
  after Phase 10's row (table order only, not a dependency).
- `README.md` / `.env.example`: `ESCALATION_WEBHOOK_URL`, `ESCALATION_WEBHOOK_SECRET`
  env vars, plus a short "set up n8n locally" walkthrough (Docker, a Webhook-trigger
  node, optionally a Code node verifying `X-Signature-256`, → a Slack/console output
  node) at the same level of detail the README already gives other setup steps.

## Checkpoint

Automated: all new tests pass, offline, no real n8n/network needed — same as every
other backend-swap test in this project.

Manual (same honesty convention as Phase 7/9's real-mic/real-call checkpoints): run n8n
locally via Docker, wire a Webhook node to a Slack (or console) output, set
`ESCALATION_WEBHOOK_URL` pointing at it, run `transport/text_cli.py`, trigger a real
escalation (e.g. ask for a human), and confirm the notification arrives in n8n/Slack
with a legible, redacted payload.

## Self-review

- No placeholders/TBDs remain — every component above has a concrete signature,
  behavior, and test plan.
- Internal consistency: the data-flow diagram matches the component descriptions;
  error-handling section matches what the tests assert.
- Scope: single, focused change (one new module + two small additions to one existing
  file + one schema addition + docs). Does not need further decomposition.
- Ambiguity check: "escalations only" (not tickets) and "n8n, self-hosted" (not
  Zapier/Make) were both explicit user decisions, recorded above rather than left open.
