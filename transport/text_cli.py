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
from agent.tools import orders

TOOLS = [orders.TOOL_SCHEMA]

TOOL_HANDLERS = {
    "get_order_status": orders.get_order_status,
}


def dispatch_tool(tool_name: str, tool_input: dict) -> dict:
    """Route one tool_use call to the function that implements it."""
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        raise ValueError(f"unknown tool: {tool_name}")
    return handler(**tool_input)


async def main() -> None:
    configure_logging()
    agent = Agent(system=SYSTEM_PROMPT, tools=TOOLS, tool_executor=dispatch_tool)

    print("Support chat — type 'quit' or 'exit' to leave.\n")
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


if __name__ == "__main__":
    asyncio.run(main())
