# LLM Voice Agent for Customer Support — Unified Project Plan

**Goal:** one project that implements all six support capabilities (order status, refunds/returns, FAQ/policy Q&A, ticket triage & escalation, appointment scheduling, post-call summary/logging) behind a single Python voice agent, built in difficulty order from a text-only core out to a real phone line.

**Guiding principle:** build and fully test the *business logic* as a text chatbot first. Voice and telephony are transport layers bolted on top later — if the agent's brain is decoupled from I/O, adding audio and then a phone number becomes plumbing work, not a rewrite. This is the single biggest thing that keeps a project like this from becoming unmanageable.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| LLM | Claude (Messages API, tool use) | Structured tool calling, good at following policy constraints |
| STT | Deepgram Flux | Lowest end-of-speech latency, built for voice agents |
| TTS | Cartesia Sonic or Deepgram Aura-2 | Sub-100ms time-to-first-audio |
| Real-time pipeline | Pipecat | Python-native, you control every processing step |
| Telephony | Twilio (Media Streams) | Standard, well-documented, works with Pipecat |
| Vector store (RAG) | Chroma (local) | Zero-infra, fine for a learning project |
| Data | SQLite | Mock orders/customers/tickets/appointments |
| Web/webhooks | FastAPI | Receives Twilio webhooks, async-native |

---

## Repo structure

```
support_voice_agent/
  agent/
    core.py          # Claude client + tool-use loop (the "brain")
    tools/
      orders.py
      policy_rag.py
      escalation.py
      scheduling.py
      refunds.py
      summary.py
    prompts.py
  data/
    mock_db.py        # SQLite seed + access layer
    policies/          # fake policy docs for RAG
  transport/
    text_cli.py        # Phase 1-6 interface
    voice_local.py      # Phase 7 interface
    pipeline.py          # Phase 8 Pipecat pipeline
    telephony.py          # Phase 9 Twilio webhook server
  guardrails/
    pii.py
    validators.py
  eval/
    scenarios.py
    run_eval.py
  tests/
  .env.example
  README.md
```

Keeping `agent/core.py` identical across phases 1–9 is the test of whether the architecture is actually decoupled — if you find yourself editing business logic to make voice work, that's a signal the boundary is in the wrong place.

Note: the transport-layer directory is named `transport/` rather than `io/` (as an earlier draft had it) to avoid shadowing Python's built-in `io` module once it's a real importable package.

---

## Phase 0 — Foundations (setup)

- Scaffold the repo above; `.env` for `ANTHROPIC_API_KEY`, `DEEPGRAM_API_KEY`, etc.
- Build `mock_db.py`: SQLite with `customers`, `orders`, `tickets`, `appointments` tables, seeded with fake data.
- Build `agent/core.py`: a thin wrapper around the Claude Messages API that runs the standard tool-use loop (send message → if `tool_use` in response, run the tool, send result back → repeat until a final text reply).
- `pytest` scaffold + logging setup.

**Checkpoint:** a script that sends "hello" and gets a reply; one passing trivial test.

---

## Phase 1 — Order & account status lookup *(easiest)*

Single deterministic tool, no side effects, clear pass/fail.

- Define a `get_order_status(order_id)` tool schema and implement it against the mock DB.
- Wire into the Phase 0 tool-use loop; write a system prompt that scopes the agent to support topics only.
- Build `transport/text_cli.py` — a simple REPL chat loop.

**Best practice:** this is a predictable, single-step task — use a plain deterministic function call rather than giving the model open-ended freedom here. Save autonomy for where it's actually needed (Phase 3).

**Checkpoint:** scripted tests for a valid order, an invalid order number, and a not-found order.

---

## Phase 2 — Post-session summary & CRM logging *(easy–medium)*

Comes second because it needs a working conversation to summarize, but the mechanics are simple: one structured-output call.

- Define a JSON schema (issue, resolution, sentiment, follow-up needed).
- At session end, call Claude once to produce it; write to a `tickets` table.
- Test: the output always validates against the schema (run it 20+ times — LLM structured output can occasionally drift).

---

## Phase 3 — FAQ / policy Q&A via RAG *(medium)*

Introduces retrieval — the first real jump in complexity.

- Write 10–20 fake policy documents (returns window, shipping, warranty, etc.).
- Chunk + embed them (Voyage AI embeddings pair well with Claude) into Chroma.
- Add a `search_policy(query)` tool; instruct the model to answer *only* from retrieved chunks and say "I don't know, let me check" otherwise.

**Checkpoint:** ask questions not covered by the docs and confirm the agent doesn't invent an answer. This is your first hallucination test — treat it as a real test, not a demo.

---

## Phase 4 — Ticket triage & escalation *(medium–high)*

Requires classification plus a genuinely structured handoff, not just a flag.

- Add lightweight intent/sentiment classification per turn.
- Define explicit escalation triggers: explicit request for a human, negative sentiment, repeated failed attempts, policy-restricted topics.
- Build a `create_handoff_packet` tool that assembles: customer intent, conversation summary, verified account info, actions already taken, reason for escalation, sentiment. (Research consistently flags that most bot handoffs lose context — the packet is the point, not the escalation flag itself.)
- For now, "transfer to human" just logs the packet — real transfer comes in Phase 10.

**Checkpoint:** run scripted frustrated-customer conversations; confirm escalation fires neither too eagerly nor too late.

---

## Phase 5 — Appointment / callback scheduling *(medium–high)*

First feature requiring multi-turn state and negotiation, not just single lookups.

- Mock calendar API (or real Google Calendar API for extra practice).
- Tools: `find_available_slots`, `book_appointment`, `cancel_appointment`.
- Handle corrections mid-conversation ("actually, next week instead").

**Checkpoint:** double-booking attempt, reschedule, cancellation, and a slot that no longer exists by the time of confirmation.

---

## Phase 6 — Returns & refunds workflow *(hardest business logic)*

Ties together everything before it: order lookup, policy RAG, and escalation, plus new validation gates and money movement.

- Design the decision path explicitly: verify purchase → check return window against policy → assess stated item condition → calculate refund amount → auto-escalate if above a $ threshold → require explicit caller confirmation → issue refund.
- Tool errors should return structured error info the model can reason about, not raise exceptions it can't parse.
- **Hard rule:** the refund tool must never execute without an explicit confirmation turn from the caller. Never let a model-initiated action move money unconfirmed.

**Checkpoint:** valid return, expired window, missing receipt, and a high-value refund that correctly escalates instead of auto-approving.

At this point all six original ideas work end-to-end over text chat, fully tested. That's a complete, demoable project even before touching audio.

---

## Phase 7 — Add voice I/O (local loop, no telephony yet)

- Mic capture (`sounddevice`), send to Deepgram STT.
- Feed the transcript into the *same* `agent/core.py` from phases 1–6 — no business-logic changes.
- Take the reply, send to TTS, play back audio.
- Log per-turn latency (STT time / LLM time / TTS time) from the start — you'll need this baseline later.

**Checkpoint:** a full spoken conversation for at least two of the six features.

---

## Phase 8 — Real-time streaming pipeline (Pipecat)

- Rebuild the Phase 7 loop as a Pipecat pipeline of frame processors.
- Add voice-activity detection and barge-in handling (stop TTS playback the instant the caller starts talking).
- Handle partial/interim transcripts, not just final ones.

**Checkpoint:** stress-test with rapid interruptions and overlapping speech; target round-trip latency under ~1s.

---

## Phase 9 — Telephony (Twilio)

- Provision a Twilio number; build a FastAPI webhook that accepts Twilio Media Streams over WebSocket.
- Route that audio stream into the Pipecat pipeline instead of the local mic.
- Add a DTMF fallback ("press 0 for a human") as a safety net independent of the AI.
- Deploy the webhook server somewhere reachable (ngrok for dev, a small VM/Fly.io/Render after).

**Checkpoint:** place a real call from your phone and run through order-status, FAQ, and returns end to end.

---

## Phase 10 — Guardrails & production hardening

- **Pre-LLM:** PII redaction on transcripts before they're logged or stored; least-privilege DB access.
- **Post-LLM:** check replies against actual tool output before they're spoken, to catch invented policy claims.
- **Injection defense:** sanitize caller speech before it's allowed to influence tool-call arguments.
- **Real warm handoff:** implement an actual Twilio call transfer using the Phase 4 handoff packet, so a human receives full context instead of a cold transfer.
- **Observability:** structured per-turn logs (transcript, tool calls, latency, escalation events).
- **Eval suite:** 10–20 scripted scenarios across all six features, run automatically, pass/fail — reliability checked before you'd ever call this "done."
- **Deploy:** Dockerize, document env vars, write the README.

---

## Rough effort sizing

| Phase | Size |
|---|---|
| 0 — Foundations | S |
| 1 — Order status | S |
| 2 — Summary/logging | S |
| 3 — FAQ/RAG | M |
| 4 — Triage/escalation | M |
| 5 — Scheduling | M |
| 6 — Refunds | L |
| 7 — Voice I/O | M |
| 8 — Pipecat streaming | L |
| 9 — Telephony | M |
| 10 — Hardening | L |

(S/M/L = relative effort, not calendar time — depends entirely on your pace.)

---

## Best-practice principles behind this ordering

- Validate business logic in text before adding real-time audio — fewer variables when something breaks.
- Use deterministic tool calls for predictable steps (status lookup, booking); reserve open-ended model judgment for genuinely ambiguous tasks (FAQ answering, triage). This mirrors Anthropic's own guidance on when to use full agent autonomy versus a fixed workflow.
- Escalation is a structured handoff, not a flag — most bot-to-human handoffs fail because context gets lost, not because the escalation decision was wrong.
- Guardrails belong both before the model sees input (PII redaction, injection sanitization) and after it generates output (fact-checking against tool results).
- Anything that moves money or is otherwise irreversible requires explicit human confirmation, never silent model-initiated execution.

## Sources

- [Voice agent frameworks: Pipecat, LiveKit Agents, and friends](https://soniox.com/wiki/voice-agent-frameworks)
- [Speech-to-Text APIs in 2026: Benchmarks, Pricing, and a Developer's Decision Guide](https://futureagi.substack.com/p/speech-to-text-apis-in-2026-benchmarks)
- [PII Redaction for Voice Agent Transcripts: The Complete Implementation Guide](https://hamming.ai/resources/pii-redaction-voice-agents)
- [Data Privacy Best Practices for AI Voice Models (2026)](https://www.cekura.ai/blogs/data-privacy-best-practices-ai-voice-models)
- [AI Agent Guardrails: Pre-LLM & Post-LLM Best Practices](https://www.arthur.ai/blog/best-practices-for-building-agents-guardrails)
- [Building AI Agents with Claude in 2026](https://www.blockchain-council.org/claude-ai/building-ai-agents-with-claude-2026-tool-use-workflows-automation-best-practices/)
- [AI-to-Human Handoff: Best Practices for Support Escalation in 2026](https://bluetweak.com/blog/ai-to-human-handoff)
- [AI-to-Human Handoff in Ecommerce: 7-Step Context Transfer](https://alhena.ai/blog/ai-human-escalation-chatbot-handoff-best-practices/)
