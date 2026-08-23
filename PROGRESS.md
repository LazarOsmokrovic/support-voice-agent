# Progress Tracker

Update this file the moment a phase's checkpoint passes — status, date, one-line note. Do not start a phase whose predecessor isn't marked **Done**.

| # | Phase | Status | Date | Notes |
|---|---|---|---|---|
| 0 | Foundations (repo, mock DB, tool-use loop) | Done | 2026-08-22 | Async `Agent` tool-use loop in `agent/core.py`; SQLite mock DB (Amazon-style orders) in `data/mock_db.py`; pytest scaffold. Live "hello" checkpoint verified 2026-08-23 once the account had a funded key. |
| 1 | Order & account status lookup | Done | 2026-08-22 | `get_order_status` tool wired into the Phase 0 loop; `transport/text_cli.py` REPL; 5 new tests (valid/invalid/not-found + wiring), all passing |
| 2 | Post-session summary & CRM logging | Done | 2026-08-23 | `SessionSummary` structured-output call + ticket logging, wired into `text_cli.py`'s exit path. Added `end_conversation` tool + quieter default logging after live REPL testing surfaced both gaps. All 18 tests pass, incl. the live 20x schema-validation checkpoint. |
| 3 | FAQ / policy Q&A (RAG) | Done | 2026-08-24 | 16 policy docs chunked + embedded into Chroma (`agent/tools/policy_rag.py`); `search_policy` tool wired in with a relevance-threshold filter tuned against real data. Embedding backend is swappable (`EMBEDDING_BACKEND=local` default, no signup; `voyage` per PROJECT_PLAN.md when a key exists). All 27 tests pass, incl. the live hallucination checkpoint. |
| 4 | Ticket triage & escalation | Not started | | |
| 5 | Appointment / callback scheduling | Not started | | |
| 6 | Returns & refunds workflow | Not started | | |
| 7 | Voice I/O (local, no telephony) | Not started | | |
| 8 | Real-time pipeline (Pipecat) | Not started | | |
| 9 | Telephony (Twilio) | Not started | | |
| 10 | Guardrails & production hardening | Not started | | |

Status values: `Not started` → `In progress` → `Done`.
