# Support Voice Agent — Project Context

A Python LLM voice agent for customer support, covering six capabilities behind one agent: order status, refunds/returns, FAQ/policy Q&A, ticket triage & escalation, appointment scheduling, and post-call summary/logging. Built text-first, then voice, then a real phone line.

Full phase-by-phase plan: `PROJECT_PLAN.md`. Live status: `PROGRESS.md`.

## Tech stack
- LLM: Claude (Messages API, tool use)
- STT: Deepgram Flux
- TTS: Cartesia Sonic / Deepgram Aura-2
- Real-time pipeline: Pipecat
- Telephony: Twilio (Media Streams)
- RAG store: Chroma
- Data: SQLite (mock orders/customers/tickets/appointments)
- Web/webhooks: FastAPI
- Python 3.12+, async/await throughout

## Working rules (follow these every session)

1. Implement **one phase at a time**, in the order given in `PROJECT_PLAN.md`. Do not write code for a later phase while an earlier one is incomplete.
2. Before starting work, check `PROGRESS.md` to confirm the previous phase is marked Done.
3. Each phase in `PROJECT_PLAN.md` has a "Checkpoint" — implement it, then write and run the tests/checks it describes. A phase isn't done until its checkpoint passes.
4. When a checkpoint passes: update `PROGRESS.md` (status → Done, add the date and a one-line note), then **stop and wait for explicit confirmation** before starting the next phase. Don't self-approve and continue.
5. Keep `agent/core.py` (the Claude tool-use loop) and everything under `agent/tools/` completely decoupled from the active I/O layer (text CLI → local voice → Pipecat → Twilio). If wiring up voice or telephony ever requires changing business logic in `agent/`, stop and flag it — that means the abstraction broke.
6. Never let a tool execute an irreversible action (issuing a refund, booking/cancelling) without an explicit confirmation turn from the user first.
7. Use deterministic tool calls for predictable steps (lookups, bookings, refunds). Reserve open-ended model judgment for genuinely ambiguous tasks (FAQ answering, triage/sentiment).
8. If a design decision in `PROJECT_PLAN.md` seems wrong once you're actually implementing it, say so and propose the change — don't silently deviate.

## Repo layout

See `PROJECT_PLAN.md` for the full annotated tree. Top level: `agent/` (brain + tools), `data/` (mock DB + policy docs), `io/` (transport layers per phase), `guardrails/`, `eval/`, `tests/`.
