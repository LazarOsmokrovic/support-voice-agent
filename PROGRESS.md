# Progress Tracker

Update this file the moment a phase's checkpoint passes — status, date, one-line note. Do not start a phase whose predecessor isn't marked **Done**.

| # | Phase | Status | Date | Notes |
|---|---|---|---|---|
| 0 | Foundations (repo, mock DB, tool-use loop) | Done | 2026-08-22 | Async `Agent` tool-use loop in `agent/core.py`; SQLite mock DB (Amazon-style orders) in `data/mock_db.py`; pytest scaffold. Live "hello" checkpoint verified 2026-08-23 once the account had a funded key. |
| 1 | Order & account status lookup | Done | 2026-08-22 | `get_order_status` tool wired into the Phase 0 loop; `transport/text_cli.py` REPL; 5 new tests (valid/invalid/not-found + wiring), all passing |
| 2 | Post-session summary & CRM logging | Done | 2026-08-23 | `SessionSummary` structured-output call + ticket logging, wired into `text_cli.py`'s exit path. Added `end_conversation` tool + quieter default logging after live REPL testing surfaced both gaps. All 18 tests pass, incl. the live 20x schema-validation checkpoint. |
| 3 | FAQ / policy Q&A (RAG) | Done | 2026-08-24 | 16 policy docs chunked + embedded into Chroma (`agent/tools/policy_rag.py`); `search_policy` tool wired in with a relevance-threshold filter tuned against real data. Embedding backend is swappable (`EMBEDDING_BACKEND=local` default, no signup; `voyage` per PROJECT_PLAN.md when a key exists). All 27 tests pass, incl. the live hallucination checkpoint. |
| 4 | Ticket triage & escalation | Done | 2026-08-24 | Per-turn `classify_turn` (intent/sentiment/policy-restricted) + deterministic `EscalationTracker` (2-consecutive thresholds) in `agent/tools/escalation.py`; `create_handoff_packet` assembles + persists to a new `escalations` table. All 44 tests pass, incl. 3 live scripted conversations confirming escalation fires neither too eagerly nor too late. |
| 5 | Appointment / callback scheduling | Done | 2026-08-24 | `find_available_slots`/`book_appointment`/`cancel_appointment` in `agent/tools/scheduling.py`, mock calendar over the Phase 0 `appointments` table. Booking/cancelling enforce CLAUDE.md rule 6 (irreversible actions need real confirmation) via a propose-then-confirm-in-a-later-turn mechanism, not just prompting. Tool dispatch became a per-session factory (`build_dispatch_tool`) to support this. All 61 tests pass, incl. a live scripted book-then-reschedule conversation. |
| 6 | Returns & refunds workflow | Not started | | |
| 7 | Voice I/O (local, no telephony) | Not started | | |
| 8 | Real-time pipeline (Pipecat) | Not started | | |
| 9 | Telephony (Twilio) | Not started | | |
| 10 | Guardrails & production hardening | Not started | | |

Status values: `Not started` → `In progress` → `Done`.
