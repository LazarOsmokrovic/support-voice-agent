"""REPL chat loop — Phase 1-6 interface.

This is the one file reused across phases 1-6 as more tools came online.
Since Phase 7, it's real I/O only — reading input, printing output — with
everything else (which tools exist, how a turn is processed, escalation,
session closing) delegated to agent/session.py, which transport/
voice_local.py now shares identically. See that module's docstring for why.

Run with: python -m transport.text_cli
"""

from __future__ import annotations

import asyncio

from agent.core import configure_logging
from agent.session import DEFAULT_CUSTOMER_ID, close_session, create_session, run_turn


async def main() -> None:
    configure_logging()
    customer_id = input(f"Customer ID [{DEFAULT_CUSTOMER_ID}]: ").strip() or DEFAULT_CUSTOMER_ID
    session = create_session(customer_id)

    print("\nSupport chat — type 'quit' or 'exit' to leave.\n")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit"}:
            break

        outcome = await run_turn(session, user_text)
        print(f"Agent: {outcome.reply}\n")
        for warning in outcome.warnings:
            print(f"({warning})\n")
        if outcome.notice:
            print(f"{outcome.notice}\n")
        if outcome.ended:
            break

    close_result = await close_session(session)
    if close_result.error:
        print(f"({close_result.error})")
    elif close_result.summary is not None:
        print(
            f"Session logged as ticket #{close_result.ticket_id} "
            f"(sentiment={close_result.summary.sentiment}, "
            f"follow_up_needed={close_result.summary.follow_up_needed})"
        )


if __name__ == "__main__":
    asyncio.run(main())
