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

4 passed (control-flow logic + all 3 mock-DB tests), 1 skipped/environment-dependent (the live Claude round trip — it correctly reaches the API and gets a real, structured response back; it currently fails only because the connected Anthropic account has no credit balance, which is a billing state, not a code defect).
