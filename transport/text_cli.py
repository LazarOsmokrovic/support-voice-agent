"""REPL chat loop — Phase 1-6 interface.

This is the one file reused across phases 1-6 as more tools come online —
agent/core.py itself never changes (CLAUDE.md rule 5). TOOLS and
TOOL_HANDLERS below are the registry of what the agent can currently do;
extend both as later phases add tools, nothing else in this loop changes.

Run with: python -m transport.text_cli
"""

from __future__ import annotations

import asyncio

from agent.core import Agent, configure_logging
from agent.prompts import SYSTEM_PROMPT
from agent.tools import orders, policy_rag, summary

TOOLS = [orders.TOOL_SCHEMA, policy_rag.TOOL_SCHEMA, summary.END_CONVERSATION_SCHEMA]

TOOL_HANDLERS = {
    "get_order_status": orders.get_order_status,
    "search_policy": policy_rag.search_policy,
    "end_conversation": summary.end_conversation,
}

# No login/auth phase exists yet, so — same simplification as Phase 1's
# get_order_status — the REPL just asks for a customer ID up front. It must
# be one that already exists in the mock DB (e.g. CUST-1001..CUST-1005),
# since Phase 2's ticket logging has a foreign-key constraint on it.
DEFAULT_CUSTOMER_ID = "CUST-1001"


def dispatch_tool(tool_name: str, tool_input: dict) -> dict:
    """Route one tool_use call to the function that implements it."""
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        raise ValueError(f"unknown tool: {tool_name}")
    return handler(**tool_input)


def should_end_session(tool_calls: list[dict]) -> bool:
    """True if this turn's tool calls included the model deciding to sign off."""
    return any(call["name"] == "end_conversation" for call in tool_calls)


async def main() -> None:
    configure_logging()
    agent = Agent(system=SYSTEM_PROMPT, tools=TOOLS, tool_executor=dispatch_tool)

    customer_id = input(f"Customer ID [{DEFAULT_CUSTOMER_ID}]: ").strip() or DEFAULT_CUSTOMER_ID
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

        result = await agent.send(user_text)
        print(f"Agent: {result.reply}\n")

        if should_end_session(result.tool_calls):
            break

    # Phase 2: summarize and log the session as a ticket, once there was
    # actually a conversation to summarize. A failure here (e.g. no API
    # credit, or an unknown customer_id) shouldn't blow up the exit path
    # with a raw traceback — just tell the user and let the process end.
    if agent.messages:
        try:
            session_summary, ticket_id = await summary.close_session(customer_id, agent.messages)
            print(
                f"Session logged as ticket #{ticket_id} "
                f"(sentiment={session_summary.sentiment}, "
                f"follow_up_needed={session_summary.follow_up_needed})"
            )
        except Exception as exc:  # noqa: BLE001 — exit path must never crash on this
            print(f"(Could not log session summary: {exc})")


if __name__ == "__main__":
    asyncio.run(main())
