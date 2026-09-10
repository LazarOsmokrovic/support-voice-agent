"""Phase 10f: the browser transport's WebSocket endpoint and, more
importantly, its session lifecycle.

What is tested here is what a demo actually depends on: pressing the button
starts a real session, pressing it again ends it and writes the post-call
summary, and pressing it a third time starts something that remembers
nothing. What is NOT tested here is audio — sample rates, buffering and
playback smoothness are only answerable by speaking into it, and the spec
says so plainly rather than pretending otherwise.

Note: the brief for this file also specifies a test asserting the page and
its static assets (app.js, style.css) are served. That test — and creating
the placeholder static/ files it depends on — is deliberately omitted here:
a separate, parallel task owns the entire static/ directory, and this task
was instructed not to create or touch anything under it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from transport import browser


@pytest.mark.asyncio
async def test_a_call_creates_a_session_and_closing_it_writes_the_summary(monkeypatch):
    """The hang-up path is the whole point of the button's second press: it
    must run the same teardown a hung-up phone call runs, not merely drop
    the socket."""
    closed = []

    async def _fake_close(session):
        closed.append(session)
        return type("R", (), {"error": None, "summary": None, "ticket_id": None})()

    monkeypatch.setattr(browser, "close_session", _fake_close)
    monkeypatch.setattr(browser, "build_pipeline", lambda transport, session: object())
    monkeypatch.setattr(browser, "_run_pipeline", AsyncMock(return_value=None))

    session = await browser.run_call(websocket=object())

    assert session is not None
    assert closed == [session], "hanging up must close exactly one session"


@pytest.mark.asyncio
async def test_the_session_closes_even_when_the_call_ends_mid_turn(monkeypatch):
    """A caller can press hang-up while the model is still generating. Phase
    8 proved cancelling mid-turn is safe; this proves it still writes the
    ticket rather than leaking the session."""
    closed = []

    async def _fake_close(session):
        closed.append(session)
        return type("R", (), {"error": None, "summary": None, "ticket_id": None})()

    monkeypatch.setattr(browser, "close_session", _fake_close)
    monkeypatch.setattr(browser, "build_pipeline", lambda transport, session: object())
    monkeypatch.setattr(
        browser, "_run_pipeline", AsyncMock(side_effect=RuntimeError("socket closed mid-turn"))
    )

    session = await browser.run_call(websocket=object())

    assert closed == [session], "an abrupt disconnect must still close the session"


@pytest.mark.asyncio
async def test_two_sequential_calls_share_nothing(monkeypatch):
    """The 'call again about a different order' requirement.

    A leaked session fails silently: the agent still answers, but with the
    previous caller's conversation history and their confirmation gates
    already open. That is worse than a crash, because a demo would look fine
    right up until it quoted the wrong order back.
    """
    async def _fake_close(session):
        return type("R", (), {"error": None, "summary": None, "ticket_id": None})()

    monkeypatch.setattr(browser, "close_session", _fake_close)
    monkeypatch.setattr(browser, "build_pipeline", lambda transport, session: object())
    monkeypatch.setattr(browser, "_run_pipeline", AsyncMock(return_value=None))

    first = await browser.run_call(websocket=object())
    first.agent.messages.append({"role": "user", "content": "where is my order"})

    second = await browser.run_call(websocket=object())

    assert second is not first
    assert second.session_id != first.session_id
    assert second.agent.messages == [], "a new call must not inherit conversation history"
    assert second.turn == 0, "a new call must start at turn zero"
