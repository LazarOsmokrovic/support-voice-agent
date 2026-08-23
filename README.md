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
