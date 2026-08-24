"""REPL chat loop — Phase 1-6 interface.

This is the one file reused across phases 1-6 as more tools come online —
agent/core.py itself never changes (CLAUDE.md rule 5). TOOLS is the static
list of tool schemas the agent can currently use; extend it as later
phases add tools.

Starting Phase 5, tool *dispatch* can no longer be a static module-level
dict the way TOOL_HANDLERS used to be: book_appointment/cancel_appointment
(and, since Phase 6, issue_refund) need per-session state (a
pending-confirmation gate — see agent/confirmation.py) and to know which
customer is acting, unlike every earlier tool, which was a pure function
of its arguments. build_dispatch_tool() assembles a fresh dispatcher (and
its backing handler dict + SessionGates) once per session instead.

Run with: python -m transport.text_cli
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent.confirmation import PendingActionGate
from agent.core import Agent, configure_logging
from agent.prompts import SYSTEM_PROMPT
from agent.tools import escalation, orders, policy_rag, refunds, scheduling, summary

TOOLS = [
    orders.TOOL_SCHEMA,
    policy_rag.TOOL_SCHEMA,
    scheduling.FIND_SLOTS_SCHEMA,
    scheduling.BOOK_APPOINTMENT_SCHEMA,
    scheduling.CANCEL_APPOINTMENT_SCHEMA,
    refunds.TOOL_SCHEMA,
    summary.END_CONVERSATION_SCHEMA,
]

# No login/auth phase exists yet, so — same simplification as Phase 1's
# get_order_status — the REPL just asks for a customer ID up front. It must
# be one that already exists in the mock DB (e.g. CUST-1001..CUST-1005),
# since Phase 2's ticket logging has a foreign-key constraint on it.
DEFAULT_CUSTOMER_ID = "CUST-1001"


@dataclass
class SessionGates:
    """One PendingActionGate per gated-action family this session needs.
    Scheduling (booking/cancelling) and refunds each get their own — a
    pending refund shouldn't be clobbered by an unrelated pending booking,
    or vice versa.
    """

    scheduling: PendingActionGate = field(default_factory=PendingActionGate)
    refunds: PendingActionGate = field(default_factory=PendingActionGate)

    def advance_turn(self) -> None:
        self.scheduling.turn += 1
        self.refunds.turn += 1


def build_dispatch_tool(
    customer_id: str, gates: SessionGates | None = None
) -> tuple[Callable[[str, dict], Any], dict[str, Callable[..., Any]], SessionGates]:
    """Assemble one session's tool dispatcher.

    Returns (dispatch_tool, handlers, gates). `handlers` is returned too
    (not just the closure) so tests can stub an individual tool via
    monkeypatch.setitem — mutating the dict in place is visible to
    dispatch_tool since the closure captures it by reference, not by value.
    """
    gates = gates or SessionGates()
    handlers: dict[str, Callable[..., Any]] = {
        "get_order_status": orders.get_order_status,
        "search_policy": policy_rag.search_policy,
        "find_available_slots": scheduling.find_available_slots,
        "book_appointment": lambda **kw: scheduling.book_appointment(
            **kw, state=gates.scheduling, customer_id=customer_id
        ),
        "cancel_appointment": lambda **kw: scheduling.cancel_appointment(
            **kw, state=gates.scheduling, customer_id=customer_id
        ),
        "issue_refund": lambda **kw: refunds.issue_refund(**kw, state=gates.refunds, customer_id=customer_id),
        "end_conversation": summary.end_conversation,
    }

    def dispatch_tool(tool_name: str, tool_input: dict) -> Any:
        """Route one tool_use call to the function that implements it."""
        handler = handlers.get(tool_name)
        if handler is None:
            raise ValueError(f"unknown tool: {tool_name}")
        return handler(**tool_input)

    return dispatch_tool, handlers, gates


def should_end_session(tool_calls: list[dict]) -> bool:
    """True if this turn's tool calls included the model deciding to sign off."""
    return any(call["name"] == "end_conversation" for call in tool_calls)


async def main() -> None:
    configure_logging()
    customer_id = input(f"Customer ID [{DEFAULT_CUSTOMER_ID}]: ").strip() or DEFAULT_CUSTOMER_ID
    dispatch_tool, _handlers, gates = build_dispatch_tool(customer_id)
    agent = Agent(system=SYSTEM_PROMPT, tools=TOOLS, tool_executor=dispatch_tool)
    tracker = escalation.EscalationTracker()

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

        # Phase 5/6: advance both gates' turn counters before each real
        # exchange, so book_appointment/cancel_appointment/issue_refund can
        # tell "proposed this turn" from "confirmed in a later one" — see
        # agent/confirmation.py.
        gates.advance_turn()
        result = await agent.send(user_text)
        print(f"Agent: {result.reply}\n")

        # Phase 4: check escalation before should_end_session — a trigger
        # here always outranks the model deciding on its own the chat is
        # naturally over. A failure in the classifier call itself (e.g. no
        # API credit) shouldn't take the whole turn down with it.
        try:
            reason = await escalation.check_escalation(tracker, agent.messages, result.tool_calls)
        except Exception as exc:  # noqa: BLE001 — a classifier hiccup must not crash the chat
            reason = None
            print(f"(Could not run triage classification this turn: {exc})\n")

        if reason:
            try:
                packet = await escalation.create_handoff_packet(customer_id, agent.messages, reason)
                print(
                    f"I'm connecting you with a human agent — {reason}. "
                    f"(handoff #{packet['escalation_id']})\n"
                )
            except Exception as exc:  # noqa: BLE001 — exit path must never crash on this
                print(f"(Escalation triggered ({reason}) but the handoff packet couldn't be logged: {exc})\n")
            break

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
