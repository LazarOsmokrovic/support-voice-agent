# Support Voice Agent

Python LLM voice agent for customer support — see `CLAUDE.md` for how this project should be built (Claude Code reads this automatically), `PROJECT_PLAN.md` for the full ten-phase plan, and `PROGRESS.md` for current status.

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt   # created in Phase 0
cp .env.example .env              # then fill in API keys
```

## Running Claude Code on this project

```bash
cd "Support voice agent"
claude
```

First prompt to give it: *"Read CLAUDE.md and PROJECT_PLAN.md, then implement Phase 0. Stop when its checkpoint passes and update PROGRESS.md."*

---

## Phase 0 — Foundations (Done)

The plumbing every later phase builds on: a Claude tool-use loop and a mock database, with nothing support-specific wired in yet (no order lookups, no policies, no tools at all — that starts in Phase 1).

### `agent/core.py` — the tool-use loop

An `Agent` class wraps the Claude Messages API (`anthropic.AsyncAnthropic`) and owns one conversation's worth of state:

- `Agent(system=..., tools=..., tool_executor=...)` — `tools` and `tool_executor` are optional and unused so far; Phase 1 will pass them in without touching this file.
- `await agent.send(user_text)` runs the loop by hand (not the SDK's beta `tool_runner`, so the request/response shape stays fully visible): call the API → if `stop_reason == "tool_use"`, run every requested tool, send all results back in one message (required for parallel tool calls) → repeat until Claude stops asking for tools → return a `TurnResult(reply, messages, tool_calls)`.
- A tool that raises is caught and turned into a `tool_result` with `is_error: true` instead of crashing the conversation — the model gets to react to the failure instead of the process dying. (This groundwork is really for Phase 6's refund/return error handling, but doing it now means `core.py` shouldn't need to change when that phase arrives.)
- The whole module is `async` end-to-end, per the project's "Python 3.12+, async/await throughout" rule — so Phase 1's sync CLI just wraps it in `asyncio.run()`, and Phases 7–9 (voice, Pipecat, Twilio) can call it natively without a rewrite.
- Model defaults to `claude-opus-5`, overridable via the `ANTHROPIC_MODEL` env var (e.g. to `claude-sonnet-5` for lower-latency production voice traffic later) without touching code.
- Running `python -m agent.core` directly sends "hello" and prints the reply — the manual/live half of this phase's checkpoint.

### `data/mock_db.py` — the mock store

A local SQLite database simulating an **Amazon-style storefront**: `customers`, `orders`, `tickets`, and `appointments` tables. Order IDs follow Amazon's real "`NNN-NNNNNNN-NNNNNNN`" shape and items are the kind of products you'd actually see there (Echo Dot, Kindle Paperwhite, Instant Pot, etc.) — all fictional data, seeded locally, not touching any real Amazon system.

- `init_db(reset=False)` creates the schema; `seed_db()` fills it with fake data and is idempotent (safe to call repeatedly without duplicating rows); `reset_and_seed()` does both from scratch.
- `get_connection()` is a context manager yielding a `sqlite3.Connection` with foreign keys enforced and row access by column name.
- Deliberately just a data layer — no `get_order_status`-style query functions live here. Those belong in the tool that needs them (`agent/tools/orders.py`, Phase 1), so this file stays a dumb, reusable store.
- The actual `.db` file (`data/mock_data.db`) is gitignored — it's regenerated locally by running `python -m data.mock_db`.

### Tests (`pytest.ini`, `tests/`)

- `test_core.py` — a mocked unit test (no network, no API key needed) verifying the loop's control flow end-to-end; this is the checkpoint's required "one passing trivial test". A second test does a real "hello" round trip to Claude, automatically skipped when `ANTHROPIC_API_KEY` isn't set.
- `test_mock_db.py` — schema creation, idempotent seeding, and that every seeded order references a valid Amazon-shaped ID and a real customer.
- Run everything: `pytest -v` (from inside the venv, after `pip install -r requirements.txt`).

### Checkpoint result

All 5 tests pass, including the live "hello" round trip — confirmed 2026-08-23 once the connected Anthropic account had a valid, funded key (it had been blocked until then by account-side issues: first no credit balance, then a since-replaced invalid key — never a code defect, but worth being honest that it went unverified for a while).

---

## Phase 1 — Order & account status lookup (Done)

The first real capability: the agent can look up an order in the Amazon-style mock store. Per `PROJECT_PLAN.md`'s guidance for this phase, it's implemented as a **plain deterministic function**, not open-ended model judgment — predictable, single-step lookups don't need the model to improvise.

### `agent/tools/orders.py` — the `get_order_status` tool

- `TOOL_SCHEMA` is the Claude tool definition (name, description, JSON `input_schema` for `order_id`) — this is what gets passed in the `tools` list on each API request.
- `get_order_status(order_id)` is the actual implementation: validates the ID against Amazon's real order-number shape (`NNN-NNNNNNN-NNNNNNN`) with a regex *before* touching the database, then queries `data/mock_db.py`. It never raises — every outcome comes back as a plain dict:
  - malformed ID → `{"found": False, "error": "invalid_order_id", ...}`
  - well-formed but missing → `{"found": False, "error": "not_found", ...}`
  - found → `{"found": True, "item": ..., "status": ..., "tracking_number": ..., ...}`
- **Known simplification, called out deliberately:** this tool doesn't verify the caller is actually the order's owner — anyone who knows or guesses a valid order ID gets its status. That's fine for a read-only, no-PII-beyond-a-tracking-number demo tool, but it's worth revisiting before this pattern gets reused for anything more sensitive (Phase 6's refund flow explicitly calls for a "verify purchase" step first).

### `agent/prompts.py` — the first real system prompt

`SYSTEM_PROMPT` scopes the agent to this storefront's support topics (orders, shipping, returns, refunds, account issues), tells it to decline unrelated questions instead of answering them, and describes the one tool it has and when to use it. This is the single place later phases extend as more tools come online — `agent/core.py`'s loop doesn't need to know what's in it.

### `transport/text_cli.py` — the REPL, and the first tool-wiring point

A plain `input()`/`print()` loop that builds one `Agent` with `SYSTEM_PROMPT` and the current tool set, then forwards each line to `agent.send()`. `TOOLS` (the schema list) and `TOOL_HANDLERS` (name → function) are the registry every later phase (2 through 6) adds to — this file is reused as-is across all of them, per `PROJECT_PLAN.md`'s design. Run it with `python -m transport.text_cli`; type `quit` or `exit` to leave.

### Tests

- `tests/test_orders.py` — the checkpoint's required scripted cases: a **valid** order (full details come back), an **invalid** order-ID format (rejected before it ever reaches the database), and a well-formed but **not-found** order — plus an empty-string edge case.
- `tests/test_text_cli.py` — proves the tool is actually wired into the loop, not just callable standalone: a mocked Claude client scripts a `tool_use` turn for `get_order_status` followed by a text turn, and asserts `Agent.send()` runs the real `dispatch_tool` from `text_cli.py` and returns the right final reply.

### Checkpoint result

9 of 9 relevant tests passing (all of Phase 0's plus this phase's 5 new ones). `python -m transport.text_cli` starts, prints its prompt, and exits cleanly on `quit` — confirming the wiring (imports, `Agent` construction, tool registry) is sound. A full live conversation through the REPL is still blocked by the same Anthropic account billing issue noted in Phase 0 (no credit balance) — nothing code-related, and it'll work as soon as credits are added.

---

## Phase 2 — Post-session summary & CRM logging (Done)

At the end of a conversation, the agent now produces a structured summary of what happened and logs it as a ticket. Unlike `get_order_status`, this isn't a tool Claude chooses to call mid-chat — it's a single, separate structured-output call the *application* makes once the session is over.

### `agent/prompts.py` — `SUMMARY_PROMPT`

A second prompt, independent of `SYSTEM_PROMPT`: it asks Claude to read a finished conversation and produce four things — `issue`, `resolution`, `sentiment`, `follow_up_needed` — with a `{transcript}` placeholder for the actual conversation text.

### `agent/tools/summary.py` — the structured-output call and the CRM write

- **`SessionSummary`** (a Pydantic model) *is* "the JSON schema" the plan calls for — `issue: str`, `resolution: str`, `sentiment: Literal["positive","neutral","negative"]`, `follow_up_needed: bool` — expressed as a typed class instead of a hand-written schema dict.
- **`summarize_session(messages, client=None, model=...)`** is the one structured-output call: it flattens the conversation into readable text, plugs it into `SUMMARY_PROMPT`, and calls `client.messages.parse(..., output_format=SessionSummary)`. The SDK both constrains Claude's output to that shape *and* validates the response, handing back an already-parsed `SessionSummary` rather than a string to `json.loads()` yourself.
- **`_format_transcript(messages)`** flattens `Agent.messages` (a mix of plain dicts and raw SDK content-block objects) into plain lines like `user: Where's my order?` / `assistant: [used tool: get_order_status]`, so the summarizer sees a readable conversation instead of raw API JSON.
- **`log_ticket(customer_id, summary, created_at=None)`** writes one row into the `tickets` table from Phase 0.
- **`close_session(customer_id, messages, client=None)`** does both of the above back to back and returns `(summary, ticket_id)` — this is what an interface calls when a session ends.

### `transport/text_cli.py` — wired at both ends

- **At session start**, the REPL now asks `Customer ID [CUST-1001]:`. There's no login/auth phase yet, so — same simplification as Phase 1's order lookup — it just asks; the ID must already exist in the mock DB, since `tickets.customer_id` has a foreign-key constraint.
- **At session end** (on `quit`/`exit`/Ctrl-C/Ctrl-D), if any messages were exchanged, it calls `summary.close_session(...)` and prints e.g. `Session logged as ticket #7 (sentiment=neutral, follow_up_needed=False)`, wrapped in a `try/except` so a failure here (no API credit, bad customer ID) prints a plain message instead of a raw traceback on exit.

### Tests (`tests/test_summary.py`)

- Four offline/mocked tests (no network): transcript formatting, a direct DB write via `log_ticket`, `summarize_session` against a mocked client, and `close_session`'s summarize-then-log combination end to end.
- **The actual Phase 2 checkpoint**, `test_summarize_session_always_validates_against_schema`: calls the real Claude API 20 times against a fixed transcript and asserts every result validates against `SessionSummary` — LLM structured output can occasionally drift, so this is an empirical check across many calls, not trust in a single one. `skipif`'d without a real `ANTHROPIC_API_KEY`, same pattern as Phase 0/1's live tests.

### Two fixes from live REPL testing

Running an actual live conversation (once the account had a funded, valid key) surfaced two real gaps, both fixed:

- **Logging noise in the chat.** `configure_logging()` defaulted to `INFO`, which also surfaced the SDK's own HTTP request logs and our `tool_call` log line interleaved with the visible conversation. It now defaults to `WARNING` — a clean chat — and reads an optional `LOG_LEVEL` env var (`INFO`/`DEBUG`) to turn verbosity back on for troubleshooting.
- **No way for the agent to end things on its own.** Previously the only way to end a session was the customer typing `quit`/`exit` — the agent had no way to recognize "we're done" from a natural goodbye. Recognizing that is a genuinely ambiguous judgment call (not a fixed keyword check), so it's handled the same way as `get_order_status`: a new tool, `end_conversation` (schema + handler in `agent/tools/summary.py`, since it's tied to session-end behavior), that the model calls once it judges the issue is resolved and the customer has signed off. `transport/text_cli.py`'s `should_end_session(tool_calls)` watches for that call and breaks the loop right after printing the agent's closing reply — which then flows straight into the existing `close_session` logging, no special-casing needed there.

### Checkpoint result

All 18 tests pass, including **the phase's actual checkpoint** — `test_summarize_session_always_validates_against_schema`, 20 real calls to Claude, every one validating against `SessionSummary` — confirmed 2026-08-23 once the account had a valid, funded key. Phase 0's live "hello" test passed in the same run.

---

## Phase 3 — FAQ / policy Q&A via RAG (Done)

The agent can now answer policy questions — returns, refunds, shipping, warranty, and 12 more topics — by retrieving from real policy documents instead of guessing. This is the project's first hallucination-avoidance mechanism, and the plan's own checkpoint frames it as exactly that: "ask questions not covered by the docs and confirm the agent doesn't invent an answer."

### A decision made before writing any code: swappable embeddings

`PROJECT_PLAN.md`'s tech stack specifies Voyage AI for embeddings, but `VOYAGE_API_KEY` was empty — a separate signup from Anthropic's, with the same wait we'd already been through once. Rather than block Phase 3 on that or silently swap the plan's stated choice for something else, `agent/tools/policy_rag.py` implements both behind one `EmbeddingBackend` interface (`embed_documents` / `embed_query`), selected by an `EMBEDDING_BACKEND` env var:
- **`local`** (default) — chromadb's bundled MiniLM ONNX model. Free, no signup, works immediately; downloads its small model on first use, then runs fully offline.
- **`voyage`** — Voyage AI, exactly as the plan specifies, using its asymmetric document/query embeddings (`input_type="document"` at ingestion, `"query"` at search — Voyage's own recommendation for retrieval quality). Needs `VOYAGE_API_KEY`; model configurable via `VOYAGE_MODEL` (default `voyage-3`).

Nothing else in this file or its callers changes based on which backend is active.

### `data/policies/*.md` — 16 fake policy documents

Returns, refunds, shipping, international shipping, warranty, damaged/defective items, cancellations, price adjustments, gift returns, payment methods, account security, gift cards, digital purchases, lost/stolen packages, restocking fees, and subscription deliveries. All fictional, written for this demo store — not a claim about any real company's actual policies.

### `agent/tools/policy_rag.py` — chunking, embedding, retrieval

- **`_chunk_document`** splits each doc into paragraph-level chunks and prefixes every chunk with the doc's title — a chunk like "...must be in original packaging." means nothing on its own without knowing it's from the Returns policy, so every chunk is self-contained context.
- **`ingest_policies`** wipes and rebuilds the Chroma collection from scratch each run — same convention as `data/mock_db.py`'s `reset_and_seed()`: the docs are static fixture data, not something worth diffing incrementally. 16 docs → 58 chunks.
- **`search_policy`** embeds the question, queries Chroma (configured for cosine distance), and filters out anything above `RELEVANCE_THRESHOLD` before returning results — a chunk that's merely topically adjacent but doesn't actually answer the question shouldn't reach the model at all.
- **The threshold was tuned against real data, not guessed.** A first pass at 0.8 let a real false positive through: "Can I get a discount code for my birthday?" (not covered by any policy) still matched `gift_cards.md` at distance 0.572. Checking 8 hand-labeled questions (4 genuinely covered, 4 genuinely not) against the real corpus showed a clean gap — in-scope questions topped out at 0.504, uncovered ones started at 0.572 — so the threshold is set at 0.55, right in that gap. This was checked with the local backend only; switching to Voyage means re-checking it, since a different embedding model has a different distance distribution.

### `agent/prompts.py` — the second layer of defense

A good retrieval threshold narrows what reaches the model, but the model still has to actually use it correctly. `SYSTEM_PROMPT` now has an explicit "Policy and FAQ questions" section: always call `search_policy` for policy questions (never answer from memory, however confident), answer only from what came back, and if `found: false`, say so honestly rather than filling the gap. Retrieval filtering and prompt instruction are deliberately two independent safeguards against the same failure mode.

### `transport/text_cli.py`

`search_policy` registered in `TOOLS`/`TOOL_HANDLERS`, same pattern as the other two tools — no changes to the loop itself.

### Tests

- `tests/test_policy_rag.py` — chunking logic, loading the real 16-doc corpus, and ingest/search against a synthetic 2-doc corpus with a deterministic fake embedding backend (a bag-of-words hash: meaningfully similar for shared vocabulary, ~orthogonal otherwise) on an in-memory Chroma collection — fully offline, no network, no model download. One more test uses the *real* 16-doc corpus and the real local embedding model (still no API key — that's the default) to confirm retrieval genuinely abstains on a real uncovered question.
- `tests/test_text_cli.py` — a mocked wiring test proving `search_policy` reaches `dispatch_tool` correctly, plus **the actual Phase 3 checkpoint**: a live Claude call asked a question none of the 16 real docs cover, checking it calls `search_policy`, sees `found: false`, and responds with an honest "I don't know" rather than a fabricated answer (a keyword-based check — a best-effort automated proxy, not a substitute for actually reading the reply).

### Checkpoint result

All 27 tests pass, including the live hallucination checkpoint — confirmed 2026-08-24. A bug surfaced along the way and was fixed before any of this: `LocalEmbeddingBackend` was returning `numpy.float32` scalars inside a Python list, which Chroma's `add()` path silently tolerated but its `query()` path rejected outright — ingestion "worked" while every actual search would have crashed. Caught by actually running a query, not just ingestion, before calling it done.

---

## Phase 4 — Ticket triage & escalation (Done)

The agent can now recognize when a conversation needs a human — an explicit ask, sustained frustration, repeated failed lookups, or a policy-restricted topic — and hand off a structured packet instead of just a flag. Per `PROJECT_PLAN.md`: "the packet is the point, not the escalation flag itself."

### The core design split (per CLAUDE.md rule 7: deterministic for the predictable, model judgment for the ambiguous)

- **Genuinely ambiguous → the model decides.** Reading a message's intent, its emotional tone, and whether it touches something needing human review "regardless of tone" all require real language understanding. That's `classify_turn` — one lightweight structured-output call after every turn.
- **Objective and countable → plain code decides.** "Is this the second bad turn in a row" isn't ambiguous once you have the classifications — it's arithmetic. `EscalationTracker` is a small dataclass with two counters, no LLM call involved in the decision itself.

### `agent/tools/escalation.py`

- **`TurnClassification`** (Pydantic): `intent` (one of 7 categories, including `request_human`), `sentiment`, `policy_restricted` — kept to 3 fields on purpose, matching the plan's "lightweight" framing. `explicit_human_request` was cut as a separate field during design — it would have just duplicated `intent == "request_human"`.
- **`EscalationTracker.record_turn`** — the four triggers: `request_human` intent and `policy_restricted` both escalate immediately (no need to wait for a pattern); negative sentiment and failed lookups both need **2 consecutive** occurrences, so one grumpy word or one bad search doesn't trip it. A turn that goes well (positive/neutral sentiment, or a lookup that succeeds) resets its streak back to zero.
- **`create_handoff_packet`** — deliberately **not** a tool the model calls itself, unlike `get_order_status`/`search_policy`/`end_conversation`. The escalation *decision* is already made by the time this runs (by the tracker), so there's nothing left for the model to decide by invoking it — it's triggered by the application, the same way Phase 2's `close_session` is. It's one more structured-output call (`HandoffFields`: customer intent, conversation summary, verified account info, actions taken, sentiment) over the transcript, then a write to a new `escalations` table.
- Both structured-output calls reuse `format_transcript` from `agent/tools/summary.py` (promoted from a private `_format_transcript` to a shared public utility, rather than duplicating the same logic in a second file).

### `data/mock_db.py` — a new `escalations` table

Extends Phase 0's schema: `escalation_id`, `customer_id`, `reason`, `customer_intent`, `conversation_summary`, `verified_account_info`, `actions_taken`, `sentiment`, `created_at`. "For now, transfer to human just logs the packet" (the plan's words) reads as something more durable than a console print, so it's a real table — also sets up nicely for Phase 10's observability work.

### `agent/core.py` — one small, generically useful addition

`TurnResult.tool_calls` previously only recorded a tool's *name* and *input* — not what it actually returned. The escalation tracker needs to know whether a lookup *succeeded*, so each entry now also carries `"output"`: the tool's raw return value (or `None` if it raised). This isn't escalation-specific — it's useful telemetry for anything downstream, and Phase 10's "structured per-turn logs" will likely want it too.

### `agent/prompts.py`

A short "Escalation" section tells the model it doesn't need to manage any of this itself — just keep being honest — but to acknowledge warmly if a customer explicitly asks for a human, so its own reply doesn't feel disconnected from the handoff that's about to happen.

### `transport/text_cli.py`

After every turn: check escalation (via the shared `escalation.check_escalation` helper — classify + record in one call, so the REPL and the tests can't drift apart) *before* checking whether the model called `end_conversation` — an escalation always outranks the model's own "we're done here." If it fires, the packet is created, a transfer notice prints with the handoff ID, and the loop ends (still flowing into Phase 2's `close_session` logging afterward, same as any other exit).

### Tests

- `tests/test_escalation.py` — `EscalationTracker`'s rules tested directly and deterministically (no network): each of the 4 triggers, plus streak-reset behavior (a calm turn resets the negative streak; a successful lookup resets the failure streak; a turn with no lookup at all leaves the failure streak untouched). `log_escalation`/`create_handoff_packet` tested with a mocked client. Three live tests check `classify_turn`'s actual judgment quality on realistic messages.
- `tests/test_text_cli.py` — **the actual Phase 4 checkpoint**, three scripted live conversations: an explicit human request escalates on turn 1 (not too late); one annoyed message doesn't escalate but a second consecutive frustrated one does (neither too eager nor too late); a calm, satisfied two-turn conversation never escalates at all (not too eager).

### Checkpoint result

All 44 tests pass, including the three scripted-conversation checkpoints — confirmed 2026-08-24. The 2-consecutive thresholds held up exactly as designed on the first try: no retuning needed, unlike Phase 3's relevance threshold.

---

## Phase 5 — Appointment / callback scheduling (Done)

A mock calendar over the `appointments` table Phase 0 already seeded: `find_available_slots`, `book_appointment`, `cancel_appointment`. The plan's own framing — "the first feature requiring multi-turn state and negotiation, not just single lookups" — turned out to be right, mostly because of one thing already sitting in this project's own rules.

### The rule that shaped everything: CLAUDE.md #6

*"Never let a tool execute an irreversible action (issuing a refund, **booking/cancelling**) without an explicit confirmation turn from the user first."* Booking and cancelling are named explicitly — this couldn't just be a prompt asking the model nicely. It needed real, code-level enforcement.

**The mechanism, without adding a 4th tool:** `PROJECT_PLAN.md` lists exactly three scheduling tools, no separate `confirm_*`. So `book_appointment`/`cancel_appointment` became stateful across calls instead: the *first* call for a given action only proposes it and returns a `pending_confirmation` status — nothing is written to the database. Only a *second* call, referencing the same pending proposal, **in a later conversational turn** (never the same one — enforced by comparing turn numbers, not by trusting the model), actually commits it. This makes "propose and immediately book in one breath" structurally impossible.

This same mechanism handles two more things for free:
- **A slot vanishing before confirmation** (this phase's other checkpoint scenario): availability is re-checked fresh at *both* the propose and the confirm step, so a slot someone else grabbed in between is caught cleanly at confirm time even though it was free when first proposed.
- **Mid-conversation corrections** ("actually, next week instead"): a new proposal for a different slot just overwrites the old pending one — no special-casing needed.

### `agent/tools/scheduling.py`

- **`find_available_slots`** generates a fixed business-hours grid (9 AM-5 PM, Mon-Fri, 30-minute slots) over the next 5 business days and filters out anything already booked. Naive (timezone-less) datetimes on purpose — Phase 0's `appointments.scheduled_time` is already stored that way, and the two need to string-match exactly for availability checks to work.
- **A per-session pending-action gate** (originally a bespoke `SchedulingState`, since Phase 6 the shared `PendingActionGate` — see that phase's write-up) — just a turn counter and one pending action. Explicitly *not* a module-level singleton, because Phase 9's telephony server will handle multiple simultaneous calls in one process, and global state would leak between them.
- **`book_appointment`/`cancel_appointment`** implement the propose-then-confirm mechanism above. `cancel_appointment` also checks the appointment actually belongs to the requesting customer — an ownership check Phase 1's `get_order_status` deliberately skipped (low stakes, read-only) but that matters more once an action can change someone else's data.

### A real architecture change: tool dispatch became a factory

Every tool through Phase 4 was a pure function of its arguments, so `TOOL_HANDLERS` could be one static module-level dict. Booking/cancelling need to know *which customer* is acting and carry *per-session* pending-proposal state — the first tools that aren't stateless. `transport/text_cli.py`'s `build_dispatch_tool(customer_id)` now assembles a fresh dispatcher (closures bound to that session's state) once per session instead. `TOOLS` (the schema list) stays a static constant — only *dispatch* needed to change.

**Deliberately not done (at the time):** generalizing the propose-then-confirm mechanism into a shared, reusable utility, even though Phase 6 (refunds) was about to need the identical protection for "issuing a refund." One concrete use case wasn't enough to responsibly generalize from — see Phase 6 for what happened once there were two.

### Tests

- `tests/test_scheduling.py` — the deterministic half, no network: slot generation respects business hours; booking/cancelling only commit on a later-turn confirmation, never the same turn; **double-booking is rejected** (checkpoint); **a slot vanishing between propose and confirm is caught and cleanly rejected** (checkpoint); a correction mid-negotiation replaces the pending proposal; **cancellation actually frees the slot and updates status** (checkpoint); cancelling someone else's appointment is refused; ambiguous (multiple scheduled) and not-found cases are both reported clearly.
- `tests/test_text_cli.py` — **the actual "reschedule" checkpoint** (inherently conversational, so it's a live test): a 6-turn scripted conversation books a slot, then reschedules it. Assertions check end state (exactly one appointment scheduled, exactly one cancelled, and they're different slots) rather than each turn's exact wording, since minor phrasing variation from the model shouldn't break the test.

### Checkpoint result

All 61 tests pass, including the live reschedule conversation — confirmed 2026-08-24, on the first attempt (no retuning needed, unlike Phase 3's threshold). Blocked briefly mid-phase by the same recurring Anthropic key expiry seen in earlier phases — refreshed, then all live tests passed cleanly.

---

## Phase 6 — Returns & refunds workflow (Done)

`PROJECT_PLAN.md` calls this "the hardest business logic" — the first phase to tie together order lookup, policy RAG, and escalation, and the first tool that moves money. This phase was planned in full (using Claude Code's plan mode) before any code was written, including one explicit architecture decision put to a vote.

### The decision point: generalize the confirmation gate, or copy it again?

Phase 5 deliberately didn't extract its propose-then-confirm mechanism into a shared utility — "one use case isn't enough to know the right shape yet." Phase 6's `issue_refund` needed the exact same protection (CLAUDE.md rule 6 names "issuing a refund" explicitly, right alongside booking/cancelling). With a second real use case in hand, this was the moment to decide: copy the ~20-line pattern a second time, or extract it. Chose to extract.

**`agent/confirmation.py` (new)** — `PendingActionGate`: a turn counter, one pending action, and a single method, `check(key)`, that returns `True` only if `key` matches a proposal from a strictly earlier turn. `agent/tools/scheduling.py` was refactored to use it too — `book_appointment`/`cancel_appointment`'s hand-rolled dict comparisons became one-line `state.check(key=...)` calls, a genuine simplification, not just a rename. Scheduling keeps one shared gate between booking and cancelling; refunds get their own, separate gate, so a pending refund and a pending booking in the same conversation can't clobber each other.

### `agent/tools/refunds.py` (new) — `issue_refund`

One tool, not two — like `book_appointment`, the first call already doubles as the eligibility check. Follows the plan's decision path exactly:

1. **Verify purchase** — valid order ID, belongs to the requesting customer (ownership check, same pattern as `cancel_appointment`), not already refunded, actually delivered (this tool handles returns of delivered items; cancelling an order before it ships is a different, unmodeled policy path).
2. **Check the window against policy** — three conditions, each with its own eligibility:
   - `unopened_or_unwanted` — 30 days (`data/policies/returns_policy.md`)
   - `damaged_or_defective` — 14 days (`data/policies/damaged_or_defective_items.md`)
   - `opened_software_or_digital` — never eligible, any window
   
   The tool also calls the existing `search_policy` for the real retrieved policy text, included as `policy_reference` — the literal "check against policy" tie-in — while the actual eligibility math stays deterministic and testable, not left to retrieval.
3. **Calculate the amount** — `price × quantity`. Known simplifications, stated plainly: no restocking fee (`restocking_fees.md`) since the mock orders have no product-category/size data to key one off, and no separate shipping-refund line item, since the schema doesn't track shipping as its own charge.
4. **Auto-escalate above $150** (`HIGH_VALUE_REFUND_THRESHOLD`, splitting the seeded delivered orders meaningfully — Echo Dot and the Nike shoes stay under it, the Sony headphones go over) — and per the plan's literal ordering, this **replaces** the confirmation step entirely rather than following it. A human approves a high-value refund; the AI doesn't, even with the customer's own agreement.
5. **Require confirmation** (everything at or under the threshold) via `PendingActionGate`, then **issue**: a new `refunds` row, and the order's status flips to `Refunded`.

### `agent/tools/escalation.py` — one new, deliberately generic trigger

`EscalationTracker` gets a fourth immediate trigger: any tool call this turn with a truthy `escalate` in its output escalates, using that output's `escalation_reason`. `issue_refund` is the first tool to use it, but the tracker doesn't know anything refund-specific — any future tool could set the same flag and it would just work. This is the literal "ties together... escalation" integration the plan calls for.

### Tests

- `tests/test_refunds.py` (new) — every branch of the decision path, no network: invalid/unknown/wrong-owner orders, not-yet-delivered, the permanently-ineligible category, both windows tested on their own boundary (14 vs. 30 days, since a fixed date sits between them), propose-then-confirm (including same-turn rejection), a real commit that writes `refunds` and flips the order's status, double-refund rejected, and the high-value path skipping confirmation entirely.
- `tests/test_escalation.py` — one new test confirming the tracker honors a tool's `escalate` flag, independent of what the classifier itself thinks of the turn.
- `tests/test_scheduling.py` — mechanical: every `SchedulingState` reference became `PendingActionGate` (identical shape), confirming the refactor didn't change scheduling's behavior at all.
- `tests/test_text_cli.py` — **two live conversations**, the actual Phase 6 checkpoint: a normal refund (propose → confirm → verify the DB), and a high-value one (verify it escalates — checked structurally via the `escalate` flag and a zero-row `refunds` table, plus confirming `EscalationTracker` actually recognizes it) rather than asking to confirm.

One noted, honest limitation: the live refund tests check eligibility against the *real* current date vs. the seeded orders' fixed 2026-08 delivery dates — valid for the foreseeable future from when this was written, but bound to drift out of window eventually as real time passes. The deterministic tests inject `now` explicitly and don't have this problem; they're what actually proves the window logic, not the live ones.

### Checkpoint result

All 78 tests pass, including both live refund conversations — confirmed 2026-08-25. Blocked once mid-phase by the same recurring Anthropic key issue as every prior phase; refreshed, then everything passed cleanly.

---

## Phase 7 — Voice I/O, local loop (Done)

The agent stops being a terminal script: a real microphone and real speakers now drive the same brain that's been running since Phase 1. `PROJECT_PLAN.md`'s framing for this phase — mic → Deepgram STT → *the same* `agent/core.py` loop, unchanged → TTS → speaker playback, with per-turn latency logged from the start — is exactly what got built, plus one refactor that had to happen first.

### A refactor first: `agent/session.py` (new)

`transport/text_cli.py` owned all the turn-orchestration logic through Phase 6: which tools exist, how a turn advances the confirmation gates, checks escalation, maybe hands off, and detects the model ending the conversation. `transport/voice_local.py` needs that *exact* behavior — a transport depending on another transport module would be backwards, and copying the logic risks the two drifting the moment a 7th tool gets added. This is the same "extract on the second real use case" call made for `agent/confirmation.py` in Phase 6, and precisely what CLAUDE.md rule 5 is watching for: a new I/O layer forcing a change to how business logic is organized.

**To be explicit about what did and didn't change**, since this is a refactor and not new behavior: `TOOLS`, `SessionGates`, `build_dispatch_tool`, `should_end_session`, and `DEFAULT_CUSTOMER_ID` moved out of `text_cli.py` as-is. New in `agent/session.py`: a `Session` dataclass bundling everything one conversation needs; `create_session(customer_id)` to build one; a `TurnOutcome` dataclass (`reply`, `ended`, `end_reason`, `notice`, `llm_latency_seconds`, `warnings`) capturing what one turn produced as data instead of printing it; `run_turn(session, user_text)`, doing exactly what `text_cli.py`'s loop body did inline (advance gates → time the LLM call → check escalation → maybe hand off → check for `end_conversation`), just returning that structured result; and `close_session(session)`, wrapping Phase 2's summarize-and-log with the same try/except `text_cli.py` already had. `transport/text_cli.py` itself shrank to real I/O only — `input()`/`print()` around `create_session`/`run_turn`/`close_session` — but text chat works identically to before; nothing about talking to the agent by typing was removed, only *where the shared logic lives* changed. Verified by re-running all 78 pre-existing tests after the move, unchanged.

### `transport/tts.py` (new) — the swappable TTS backend

Mirrors `agent/tools/policy_rag.py`'s `EmbeddingBackend` pattern from Phase 3, for the same underlying reason: Deepgram is mandatory anyway (Flux, for STT, has no alternative), so defaulting TTS to Deepgram too means no second signup, while Cartesia Sonic — independently benchmarked as more natural-sounding — stays one env var away for whoever wants it. `TTSBackend` is a one-method protocol (`synthesize(text) -> bytes`); `DeepgramTTSBackend` and `CartesiaTTSBackend` each wrap their provider's one-shot REST endpoint (not the streaming APIs both providers also offer — Phase 7 is a sequential local loop, not real-time streaming; that's Phase 8's job via Pipecat); `get_tts_backend()` reads `TTS_BACKEND` (`"deepgram"` default, `"cartesia"` alternative).

**One fix caught before it caused a bad bug:** Deepgram's `/v1/speak` endpoint defaults to MP3 if you don't ask otherwise, but the playback code decodes WAV via Python's stdlib `wave` module, which can't parse MP3 at all. Fixed by explicitly requesting `encoding=linear16&container=wav` in every Deepgram TTS request — an ambiguity worth catching before it turned into "TTS silently returns unplayable audio."

### `transport/voice_local.py` (new) — the actual voice loop

- **`listen_and_transcribe()`** opens a live Deepgram Flux WebSocket connection (`client.listen.v2.connect(model="flux-general-en", encoding="linear16", sample_rate=16000)`) and streams mic audio into it via `sounddevice.RawInputStream`, using an `asyncio.Queue` to hand chunks from PortAudio's callback thread to the async socket-sender task. Flux's whole value is built-in end-of-turn detection — the function just watches for its `EndOfTurn` event (part of a state machine also including `StartOfTurn`/`Update`/`EagerEndOfTurn`/`TurnResumed`) and returns that turn's final transcript, with zero hand-rolled silence detection. The exact request/response shapes here came from reading the installed `deepgram-sdk` source directly, not from web docs — the vendor's own published guides left real gaps (how audio bytes actually get sent, the exact event-type names), and guessing at an SDK's usage is exactly the kind of thing worth verifying against real source instead.
- **`speak()`** calls the active `TTSBackend`, decodes the returned WAV via stdlib `wave` into a numpy array, and plays it with `sounddevice.play()` — blocking until playback finishes, since Phase 7's loop is strictly sequential (the agent finishes speaking, *then* starts listening again). No barge-in, no partial-transcript handling — explicitly Phase 8's job once Pipecat is in the picture.
- **`main()`** has the same shape as `text_cli.py`'s loop, swapping `input()`/`print()` for `listen_and_transcribe()`/`speak()` around the identical `create_session`/`run_turn`/`close_session` calls. After every turn it logs one line — `STT: Xms | LLM: Yms | TTS: Zms | Total: Wms` — the literal "log per-turn latency from the start" requirement, giving Phase 8 a real baseline to improve on.

Run it with `python -m transport.voice_local`.

### Tests

- `tests/test_session.py` (new) — the orchestration logic moved out of `text_cli.py`'s implicit coverage: `run_turn` with a mocked Claude client (plain reply, a tool call, an escalating turn producing a notice, the model ending the conversation), and a classifier failure surfacing as a warning instead of crashing the turn.
- `tests/test_tts.py` (new) — both backends against mocked HTTP responses (`pytest-httpx`, no real network) verifying the exact request shape each provider expects, plus `get_tts_backend()`'s env-var switching.
- `tests/test_voice_local.py` (new) — `listen_and_transcribe`'s control flow against a fake Flux socket (stopping exactly at `EndOfTurn`, raising cleanly on a fatal error), WAV encode/decode round-tripping, and `speak()`'s playback call, all without touching a real mic or network. Plus **a live, fully automated round-trip test**: synthesize a known phrase through the real TTS backend, feed the resulting audio straight into a real Flux connection, and assert the transcript reasonably matches — proving the STT↔TTS integration end-to-end with zero human voice needed.

### Two things worth being upfront about

- **A one-time local environment fix, not a code issue:** this machine's Python.org build doesn't wire the stdlib `ssl` module into macOS's system certificate store, which makes the `websockets` library (used for the Flux connection) fail its TLS handshake with `CERTIFICATE_VERIFY_FAILED` — `httpx` calls (the TTS side) are unaffected since it bundles its own CA bundle via `certifi`. Fixed locally by setting `SSL_CERT_FILE` to `certifi`'s bundle before running voice code; the standard permanent fix is running the "Install Certificates.command" that ships alongside python.org's macOS installers.
- **What no automated test can cover:** an actual person speaking into an actual microphone. Every test above proves the pieces (STT, TTS, orchestration) work and even that they work *together* automatically — but the phase's literal checkpoint, "a full spoken conversation for at least two of the six features," needs a real voice and real ears. That part is yours to run: `python -m transport.voice_local`, have a real exchange covering e.g. an order-status question and a policy question, confirm it feels right.

### Checkpoint result

93 of 94 tests pass, including the live automated TTS→Flux round-trip — confirmed 2026-08-25. The one failure (`test_high_value_refund_conversation_escalates_instead_of_confirming`, a Phase 6 live test) is the same recurring Anthropic account credit-balance issue seen in earlier phases, not a Phase 7 regression. The hands-on spoken-conversation checkpoint is still outstanding and needs to be run in person.
