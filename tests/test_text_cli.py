"""Phase 1/2/3/4/5 checkpoints for the REPL: tools are actually wired into
the Phase 0 tool-use loop, not just callable on their own, and the loop
knows to stop when the model signals the conversation is over.

Uses a mocked Claude client (no network, no API key) that scripts a
two-turn exchange: first Claude asks for a tool, then it replies with text
once the tool result comes back — exercising Agent.send end to end with a
real dispatch_tool built by transport/text_cli.py's build_dispatch_tool
(a factory since Phase 5 — see that module's docstring for why tool
dispatch can no longer be a static constant).

The live tests (gated on a real ANTHROPIC_API_KEY) are the full-pipeline
checkpoints: Phase 3's (a genuinely uncovered policy question, checking the
model doesn't invent an answer once retrieval correctly comes back empty),
Phase 4's (scripted conversations, checking escalation fires neither too
eagerly nor too late), and Phase 5's (a scripted booking/reschedule
conversation) — all run through the actual Agent + real Claude + real
tools (local embedding backend for search_policy — no Voyage key needed).
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core import Agent
from agent.prompts import SYSTEM_PROMPT
from agent.tools import escalation
from data import mock_db
from transport.text_cli import TOOLS, build_dispatch_tool, should_end_session


def _fresh_dispatch_tool(customer_id: str = "CUST-1001"):
    """Most tests here don't care about scheduling state or which customer
    is acting — this just saves repeating build_dispatch_tool's 3-tuple
    unpacking everywhere. Tests that DO care call build_dispatch_tool
    directly.
    """
    dispatch_tool, _handlers, _state = build_dispatch_tool(customer_id)
    return dispatch_tool


def _tool_use_response(name: str, tool_input: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.id = "toolu_test123"
    block.name = name
    block.input = tool_input
    response = MagicMock()
    response.content = [block]
    response.stop_reason = "tool_use"
    return response


def _text_response(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.stop_reason = "end_turn"
    return response


@pytest.mark.asyncio
async def test_order_status_tool_is_wired_into_the_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_wiring.db")
    mock_db.reset_and_seed()
    order_id = mock_db.ORDERS[0][0]

    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(
        side_effect=[
            _tool_use_response("get_order_status", {"order_id": order_id}),
            _text_response("Your order has shipped!"),
        ]
    )

    agent = Agent(client=fake_client, tools=TOOLS, tool_executor=_fresh_dispatch_tool())
    result = await agent.send(f"Where's my order {order_id}?")

    assert result.reply == "Your order has shipped!"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "get_order_status"
    assert result.tool_calls[0]["input"] == {"order_id": order_id}
    assert result.tool_calls[0]["output"]["found"] is True  # Phase 4 needs the raw output, not just name/input
    assert fake_client.messages.create.await_count == 2


@pytest.mark.asyncio
async def test_end_conversation_tool_is_wired_into_the_loop():
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(
        side_effect=[
            _tool_use_response("end_conversation", {}),
            _text_response("Glad that's sorted — have a great day!"),
        ]
    )

    agent = Agent(client=fake_client, tools=TOOLS, tool_executor=_fresh_dispatch_tool())
    result = await agent.send("Thanks, that's all, have a nice day!")

    assert result.reply == "Glad that's sorted — have a great day!"
    assert should_end_session(result.tool_calls) is True


def test_should_end_session_is_false_without_end_conversation_call():
    tool_calls = [{"name": "get_order_status", "input": {"order_id": "112-3487561-2938471"}}]
    assert should_end_session(tool_calls) is False


def test_should_end_session_is_false_with_no_tool_calls():
    assert should_end_session([]) is False


@pytest.mark.asyncio
async def test_search_policy_tool_is_wired_into_the_loop(monkeypatch):
    # Stub the handler rather than hit the real retrieval pipeline — this
    # test is about the wiring (does a search_policy tool_use call actually
    # reach transport.text_cli's dispatch_tool), not retrieval quality,
    # which tests/test_policy_rag.py already covers on its own.
    dispatch_tool, handlers, _state = build_dispatch_tool("CUST-1001")
    monkeypatch.setitem(
        handlers,
        "search_policy",
        lambda query: {
            "found": True,
            "results": [{"source": "returns_policy.md", "title": "Returns Policy", "text": "...", "distance": 0.1}],
        },
    )
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(
        side_effect=[
            _tool_use_response("search_policy", {"query": "How long do I have to return something?"}),
            _text_response("You have 30 days to return most items."),
        ]
    )

    agent = Agent(client=fake_client, tools=TOOLS, tool_executor=dispatch_tool)
    result = await agent.send("How long do I have to return something?")

    assert result.reply == "You have 30 days to return most items."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "search_policy"
    assert result.tool_calls[0]["input"] == {"query": "How long do I have to return something?"}
    assert result.tool_calls[0]["output"]["found"] is True


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="requires a real ANTHROPIC_API_KEY to hit the live Claude API",
)
@pytest.mark.asyncio
async def test_agent_does_not_invent_an_answer_for_an_uncovered_policy_question():
    """Phase 3 checkpoint, full pipeline: a real Claude call, asked a policy
    question none of the 16 real policy docs cover, correctly calls
    search_policy, sees found: False (verified separately and more directly
    in tests/test_policy_rag.py), and doesn't fabricate a confident answer
    instead. The keyword check below is a best-effort automated proxy for
    "didn't hallucinate," not a full substitute for reading the reply —
    this was also checked by hand in the live REPL, see the phase write-up.
    """
    agent = Agent(system=SYSTEM_PROMPT, tools=TOOLS, tool_executor=_fresh_dispatch_tool())

    result = await agent.send("Do you offer price matching with other stores?")

    search_calls = [c for c in result.tool_calls if c["name"] == "search_policy"]
    assert search_calls, "expected the model to call search_policy for a policy question"

    honest_signal = any(
        phrase in result.reply.lower()
        for phrase in [
            "don't have",
            "not sure",
            "check",
            "get back to",
            "no information",
            "not something",
            "don't currently",
            "unable to find",
        ]
    )
    assert honest_signal, f"reply doesn't read like an honest 'don't know' — got: {result.reply!r}"


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="requires a real ANTHROPIC_API_KEY to hit the live Claude API"
)
@pytest.mark.asyncio
async def test_escalation_fires_immediately_on_explicit_human_request():
    """Phase 4 checkpoint — not too late: an explicit ask for a human
    escalates on the very first turn, not after several more exchanges."""
    agent = Agent(system=SYSTEM_PROMPT, tools=TOOLS, tool_executor=_fresh_dispatch_tool())
    tracker = escalation.EscalationTracker()

    result = await agent.send("I don't want to talk to a bot, please connect me with a real person.")
    reason = await escalation.check_escalation(tracker, agent.messages, result.tool_calls)

    assert reason == "explicit request for a human"


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="requires a real ANTHROPIC_API_KEY to hit the live Claude API"
)
@pytest.mark.asyncio
async def test_escalation_fires_on_sustained_frustration_not_on_the_first_complaint():
    """Phase 4 checkpoint — neither too eager nor too late: one grumpy
    message shouldn't escalate on its own, but frustration sustained across
    turns should, by the second consecutive negative turn."""
    agent = Agent(system=SYSTEM_PROMPT, tools=TOOLS, tool_executor=_fresh_dispatch_tool())
    tracker = escalation.EscalationTracker()

    turns = [
        "My order is 3 days late, that's kind of annoying.",
        "This is ridiculous, it's been late every single time and nobody seems to care.",
    ]
    reasons = []
    for turn in turns:
        result = await agent.send(turn)
        reasons.append(await escalation.check_escalation(tracker, agent.messages, result.tool_calls))

    assert reasons[0] is None, "a single mildly annoyed message shouldn't escalate on its own"
    assert reasons[1] == "sustained negative sentiment across multiple turns"


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="requires a real ANTHROPIC_API_KEY to hit the live Claude API"
)
@pytest.mark.asyncio
async def test_escalation_never_fires_for_a_calm_satisfied_conversation():
    """Phase 4 checkpoint — not too eager: an ordinary, friendly
    conversation should never trip any escalation trigger."""
    agent = Agent(system=SYSTEM_PROMPT, tools=TOOLS, tool_executor=_fresh_dispatch_tool())
    tracker = escalation.EscalationTracker()

    turns = [
        "Hi! Can you tell me when order 112-3487561-2938471 will arrive?",
        "Great, thanks so much for checking!",
    ]
    for turn in turns:
        result = await agent.send(turn)
        reason = await escalation.check_escalation(tracker, agent.messages, result.tool_calls)
        assert reason is None


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="requires a real ANTHROPIC_API_KEY to hit the live Claude API"
)
@pytest.mark.asyncio
async def test_scheduling_book_then_reschedule_conversation(tmp_path, monkeypatch):
    """Phase 5 checkpoint: reschedule. A scripted conversation books a
    callback, then reschedules it to a different slot. Asserts on end
    state rather than each turn's exact wording — robust to minor
    variation in how the model phrases things, while still proving a real
    reschedule happened (not just one booking, or two bookings with
    nothing cancelled).
    """
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_scheduling_live.db")
    mock_db.reset_and_seed()
    customer_id = "CUST-1001"

    dispatch_tool, _handlers, gates = build_dispatch_tool(customer_id)
    agent = Agent(system=SYSTEM_PROMPT, tools=TOOLS, tool_executor=dispatch_tool)

    turns = [
        (
            "Can you check what appointment slots you have available in the next few days? "
            "I'd like to book a callback about a return."
        ),
        "Great, let's book the first slot you listed.",
        "Yes, please go ahead and confirm that.",
        (
            "Actually, I need to reschedule — could we move it to a later slot instead? "
            "Whatever's next available after that one is fine."
        ),
        "Yes, that works — please confirm the new time, and once that's booked, cancel the old one.",
        "Yes, please cancel the old one.",
    ]
    for turn in turns:
        gates.advance_turn()
        await agent.send(turn)

    with mock_db.get_connection() as conn:
        scheduled = conn.execute(
            "SELECT scheduled_time FROM appointments WHERE customer_id = ? AND status = 'scheduled'",
            (customer_id,),
        ).fetchall()
        cancelled = conn.execute(
            "SELECT scheduled_time FROM appointments WHERE customer_id = ? AND status = 'cancelled'",
            (customer_id,),
        ).fetchall()

    assert len(scheduled) == 1, f"expected exactly one scheduled appointment, got {scheduled}"
    assert len(cancelled) == 1, f"expected the original slot to end up cancelled, got {cancelled}"
    assert scheduled[0]["scheduled_time"] != cancelled[0]["scheduled_time"]


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="requires a real ANTHROPIC_API_KEY to hit the live Claude API"
)
@pytest.mark.asyncio
async def test_refund_conversation_proposes_then_confirms(tmp_path, monkeypatch):
    """Phase 6 checkpoint: a normal, low-value refund conversation actually
    writes a refund and updates the order.

    Note on calendar drift: issue_refund's tool schema doesn't expose `now`
    to the model (matching find_available_slots), so this live test checks
    eligibility against the REAL current date vs. the seeded order's fixed
    2026-08-13 delivery date. It's valid for the foreseeable future from
    when this was written (2026-08-25), but will eventually fall outside
    the 30-day window as real time passes — the deterministic tests in
    tests/test_refunds.py inject `now` explicitly and don't have this
    problem; they're what actually proves the window logic is correct.
    """
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_refund_live.db")
    mock_db.reset_and_seed()
    customer_id = "CUST-1001"
    order_id = "112-3487561-2938471"  # Echo Dot, $34.99, delivered 2026-08-13

    dispatch_tool, _handlers, gates = build_dispatch_tool(customer_id)
    agent = Agent(system=SYSTEM_PROMPT, tools=TOOLS, tool_executor=dispatch_tool)

    turns = [
        f"I'd like to return order {order_id} — I just changed my mind about it.",
        "Yes, please go ahead and refund it.",
    ]
    for turn in turns:
        gates.advance_turn()
        await agent.send(turn)

    with mock_db.get_connection() as conn:
        refund = conn.execute("SELECT * FROM refunds WHERE order_id = ?", (order_id,)).fetchone()
        order_status = conn.execute("SELECT status FROM orders WHERE order_id = ?", (order_id,)).fetchone()["status"]

    assert refund is not None, "expected a refund row to have been written"
    assert refund["amount"] == 34.99
    assert order_status == "Refunded"


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="requires a real ANTHROPIC_API_KEY to hit the live Claude API"
)
@pytest.mark.asyncio
async def test_high_value_refund_conversation_escalates_instead_of_confirming(tmp_path, monkeypatch):
    """Phase 6 checkpoint: auto-escalate above the $ threshold — proven live,
    not just at the tool level. Same calendar-drift caveat as the test
    above (valid from 2026-08-25 for the foreseeable future).
    """
    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_refund_escalate_live.db")
    mock_db.reset_and_seed()
    customer_id = "CUST-1005"
    order_id = "119-5647382-9182736"  # Sony WH-1000XM5, $349.99, delivered 2026-08-02

    dispatch_tool, _handlers, gates = build_dispatch_tool(customer_id)
    agent = Agent(system=SYSTEM_PROMPT, tools=TOOLS, tool_executor=dispatch_tool)
    tracker = escalation.EscalationTracker()

    gates.advance_turn()
    result = await agent.send(f"I'd like to return order {order_id} — I don't want them anymore.")

    refund_calls = [c for c in result.tool_calls if c["name"] == "issue_refund"]
    assert refund_calls, "expected the model to call issue_refund"
    assert any(c["output"].get("escalate") for c in refund_calls), "expected the high-value refund to signal escalate"

    with mock_db.get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM refunds").fetchone()[0] == 0, "should not have been issued"

    reason = await escalation.check_escalation(tracker, agent.messages, result.tool_calls)
    assert reason is not None, "the tracker should recognize the tool's escalate signal and escalate the session"
