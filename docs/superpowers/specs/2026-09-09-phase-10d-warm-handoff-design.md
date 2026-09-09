# Phase 10d — Real warm handoff: transferring a live call to a human

Date: 2026-09-09
Status: approved, not yet implemented
Depends on: Phase 4 (`create_handoff_packet`), Phase 9 (Twilio Media Streams). Independent of 10c/10e.

## Why this phase exists

`PROJECT_PLAN.md:187` promises "an actual Twilio call transfer using the Phase 4 handoff
packet, so a human receives full context instead of a cold transfer."

Today the agent *says* "Connecting you with a human agent" and then the call ends. Nothing
transfers. `create_handoff_packet` assembles customer intent, conversation summary, verified
account info, actions taken, reason and sentiment — and that packet reaches SQLite (Phase 4)
and an n8n webhook (Phase 11), but never reaches a human on the phone while the customer is
still holding.

The project's own stated principle (`PROJECT_PLAN.md:413`) is that "most bot-to-human handoffs
fail because context gets lost, not because the escalation decision was wrong." This phase is
where that principle either becomes real or stays a comment.

## Decisions taken by the project owner

Fixed inputs, not open questions:

1. **Whisper, then bridge.** The AI dials the human, plays them a spoken briefing from the
   handoff packet while the customer waits, then connects the two. Blind transfer was
   rejected — it is precisely the context loss the packet exists to prevent. A three-way
   conference was rejected as more moving parts than a demo project needs.
2. **On no answer, the AI returns and offers a callback** using Phase 5's `book_appointment`.
   The customer is never dropped. Note this is a *proposal*, not an immediate booking:
   `book_appointment` sits behind `PendingActionGate` (CLAUDE.md rule 6), so the AI offers a
   slot and the customer confirms before anything is written. The failure path inherits that
   safeguard rather than working around it.
3. **`HUMAN_AGENT_NUMBER` is a single env var** in E.164 form, matching how every other
   integration in this project is configured. No DB-driven agent routing.
4. **The live checkpoint waits for the owner's Twilio key.** Everything else is verified
   offline.

## The constraint that shapes everything

The call is inside `<Connect><Stream>` — a bidirectional Media Stream (Phase 9). Pipecat
cannot emit TwiML mid-call; it speaks audio, not call control. So the only way to transfer is
the Twilio REST API redirecting the live call away from the stream:

```
client.calls(call_sid).update(twiml=<the Dial verb>)
```

`call_sid` is already in hand — `transport/telephony.py:117` reads it from Twilio's `start`
event to build the serializer.

Redirecting replaces the call's current TwiML, which ends the Media Stream. The WebSocket
closes, `runner.run()` returns, and the existing `close_session` call at
`transport/telephony.py:138` fires on its own. The ticket and post-call summary get written
with no new machinery — the teardown path this phase needs already exists.

## Verified Twilio semantics

Checked against Twilio's documentation while designing this, because the whole mechanism
rests on them:

- **`<Number url="...">` is the whisper.** The URL returns TwiML "to be run on the called
  party's end, after they answer, but before the parties are connected." Exactly one-sided,
  exactly before bridging. Constraint: whisper TwiML **may not contain `<Dial>`**.
- **`<Dial action="...">` takes over the parent call.** Twilio POSTs to it when the dial ends,
  and "the parent call continues under the action URL's control — subsequent verbs in the
  original document become unreachable." This is what makes the no-answer path work: the
  action URL, not leftover TwiML, decides what the customer hears next.
- **`DialCallStatus`** is one of `completed`, `answered`, `busy`, `no-answer`, `failed`,
  `canceled`. Alongside it: `DialCallSid`, `DialCallDuration`, `DialBridged`.
- **`timeout`** defaults to 30s, minimum 5, maximum 600. This design uses **20s** — long
  enough for a real pickup, short enough that a held customer is not abandoned.

## The transfer TwiML

```xml
<Response>
  <Say>Connecting you now. Please hold.</Say>
  <Dial action="/transfer-status?escalation_id=42" timeout="20" callerId="<twilio number>">
    <Number url="/whisper?escalation_id=42">+3816…</Number>
  </Dial>
</Response>
```

`callerId` must be a Twilio-owned number: the customer's own caller ID cannot legally be
presented on the outbound leg.

## Two trigger points, not one

This is the phase's most likely defect, and it is the same trap Phase 10b fell into.
Escalation reaches the transport through **two independent paths**:

1. **Model-driven** — `run_turn` sets `end_reason="escalated"`, surfacing at
   `transport/pipecat_processors.py:223` via `outcome.notice`.
2. **DTMF "press 0"** — `_handle_dtmf_escalation` (`pipecat_processors.py:152`) calls
   `create_handoff_packet` directly and **bypasses `run_turn` entirely**. Phase 9 built it
   deliberately as a safety net independent of the AI.

A transfer wired only into the first path would leave the "press 0 for a human" button doing
nothing but printing a notice — the worst possible failure, because that button exists
precisely for when the AI is failing. **Both paths must fire the transfer**, and the plan
must contain a test for each.

## Keeping Twilio out of `agent/` and out of the shared processor

CLAUDE.md rule 5 requires `agent/` to stay decoupled from the I/O layer, and Phase 9
deliberately kept Twilio out of `transport/pipecat_processors.py` so the local-mic pipeline
could share it. Neither may import Twilio now.

`build_pipeline` already takes keyword-only extras
(`build_pipeline(transport, session, *, mute_mic_during_tts=False)`), so it gains one more:

```
on_escalation: Callable[[int | None], Awaitable[bool]] | None = None
```

`ClaudeTurnProcessor` awaits it at both trigger points when it is set, and does nothing when
it is not. `transport/telephony.py` — the one file that already imports Twilio — supplies the
Twilio-specific implementation. `transport/pipeline.py` passes nothing and behaves exactly as
today. Nothing under `agent/` changes at all.

The callback returns `bool`: `True` if the redirect was issued, `False` otherwise. On `False`
the processor keeps today's behaviour (speak the notice, end the turn), so a transfer failure
degrades to the current experience rather than a silent dead line.

## The no-answer path, and session continuity

`/transfer-status` receives `DialCallStatus`. On `completed`, the call is over — return
`<Hangup/>`. On `busy`, `no-answer`, `failed` or `canceled`, the AI comes back.

What the AI does on return is left to the model, not scripted by the transport: it resumes
with full history, so it can apologise and offer a callback through `book_appointment` the
same way it would in any other conversation. That keeps rule 6 intact — the tool proposes and
the customer confirms — instead of the transport booking something behind the customer's back.

Coming back means new TwiML — `<Connect><Stream>` again — which would normally build a **fresh
session with no memory of the conversation**. A customer who waited on hold and then got
greeted from scratch would be worse off than if the transfer had never been attempted.

**Fix: resume the existing session.** The stream URL carries the session id:

```
<Connect><Stream url="wss://<host>/media-stream?session=<session_id>"/></Connect>
```

`media_stream` looks the id up in a module-level registry and reuses that `Session` — full
conversation history intact — instead of calling `create_session`. The AI then apologises with
context and can offer a callback via `book_appointment`.

**Stated limitation:** the registry is an in-process dict, so this assumes a single uvicorn
worker. That is true of this project today (`transport/telephony.py:150` runs one process) and
a distributed session store would be machinery a mock project cannot justify. Documented rather
than solved, and the registry entry is removed when the session finally closes so it cannot
grow without bound.

## Security: two new public webhooks

`/whisper` and `/transfer-status` are internet-reachable and carry an `escalation_id`. Both
**must validate `X-Twilio-Signature`** exactly as `/voice` already does
(`transport/telephony.py:82-86`). Without that, anyone who guesses a URL can read a customer's
handoff briefing — intent, summary, account info — by requesting `/whisper?escalation_id=1`.

The signature check is the whole access control here, so the plan must test it: a request with
a bad signature returns 403 and leaks nothing.

The whisper text is built from the packet **as stored**, which `guardrails/pii.py` already
redacted at `create_handoff_packet` (Phase 10a). The human agent hears masked contact details
and an intact order ID — which is the right split, since the order ID is what lets them act.

## Components

| File | Change |
|---|---|
| `transport/telephony.py` | `/whisper` and `/transfer-status` endpoints; `transfer_to_human()`; the transfer registry; pass `on_escalation` into `build_pipeline` |
| `transport/pipecat_processors.py` | `build_pipeline` gains `on_escalation`; `ClaudeTurnProcessor` awaits it at BOTH trigger points |
| `.env.example` / `README.md` | `HUMAN_AGENT_NUMBER`, `TWILIO_ACCOUNT_SID` |

**Zero changes under `agent/`** — not even a read query. The first draft of this design added
a `get_handoff_packet(escalation_id)` lookup so `/whisper` could fetch the packet in its own
HTTP request. That is unnecessary: `transfer_to_human` already holds the packet at the moment
it issues the redirect, so it stashes the rendered whisper text in the same in-process
registry the session resume needs, keyed by `escalation_id`. `/whisper` reads it back.

That removes a database round-trip, keeps `agent/` untouched for the third phase running, and
means the whisper text is built once by the code that has full context rather than
reconstructed later from stored columns. The already-specified fallback covers the only case
it loses — a process restart between transfer and whisper — with a generic briefing.

## Error handling

- **`HUMAN_AGENT_NUMBER` unset** — no transfer attempted, callback returns `False`, today's
  behaviour preserved. Same optional-by-default convention as `ESCALATION_WEBHOOK_URL`.
- **REST redirect fails** (bad credentials, call already ended, Twilio 4xx/5xx) — logged,
  returns `False`, the agent speaks the notice as it does today. A failed transfer must never
  drop the call.
- **`escalation_id` not found** at `/whisper` — a generic briefing ("A customer is waiting,
  no context available"), never a stack trace and never silence. The human still gets the call.
- **Unknown `session` id** at `/media-stream` — fall back to `create_session`. A cold restart
  is worse than resuming, but far better than a 500 and a dropped call.
- **Transfer fires twice** (model escalation immediately followed by a DTMF press) — the
  redirect is idempotent per call: a flag on the session marks the transfer as issued, and the
  second attempt is a no-op.

## Testing — all offline

The Twilio REST client is faked, the way `eval/` already fakes Anthropic. No live calls.

- `transfer_to_human` issues `calls(sid).update()` with TwiML containing the right number,
  the whisper URL, the action URL and `timeout="20"`; asserted as parsed XML, not string
  matching.
- Both trigger paths fire it: one test driving a model escalation through `run_turn`, one
  driving an `InputDTMFFrame` with `KeypadEntry.ZERO`.
- `on_escalation=None` (the local-mic pipeline) changes nothing.
- Second transfer attempt on the same call is a no-op.
- `/whisper` returns TwiML containing the packet's summary; with a bad signature returns 403
  and no packet content.
- `/transfer-status` with `DialCallStatus=completed` returns `<Hangup/>`; with `no-answer`,
  `busy` and `failed` returns `<Connect><Stream>` carrying the session id.
- `/media-stream?session=<known id>` resumes that session (history preserved); an unknown id
  creates a new one.
- Redirect failure returns `False` and does not raise.
- The registry entry is removed on session close.

Values come from `data/mock_db.py` at runtime — never hard-coded — per the defect class that
produced two Criticals in this project.

## Checkpoint

**Automated:** all new tests pass offline with no API keys, no regressions in the existing
suite.

**Manual (the owner's, needs a funded Twilio key):** place a real call, trigger an escalation
both ways — once by asking for a human, once by pressing 0 — and confirm the human's phone
rings, the whisper is heard *before* the customer is connected, and the two are then bridged.
Then let the transfer time out unanswered and confirm the AI returns with the conversation
intact and books a callback.

**Cost note for the checkpoint.** Twilio's Serbian mobile termination is $0.8211/min (landline
$0.5970). `HUMAN_AGENT_NUMBER` accepts any E.164 number, so a US Twilio number (~$0.014/min) or
a Twilio Voice SDK browser client (no PSTN leg) tests the same code path for a fraction of that.
The browser client is also the better demo: the whisper is visible on screen as it arrives.

## Out of scope

- Agent availability, queueing or routing to more than one human. One number, one attempt.
- Call recording, and the PII questions it would raise.
- A distributed session store. Single-worker is stated, not solved.
- Transferring back to the AI *after* a successful bridge. Once a human is on, the AI is out.
