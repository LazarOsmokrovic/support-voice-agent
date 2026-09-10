"""Phase 10f: the browser transport's WebSocket endpoint and, more
importantly, its session lifecycle.

What is tested here is what a demo actually depends on: pressing the button
starts a real session, pressing it again ends it and writes the post-call
summary, and pressing it a third time starts something that remembers
nothing. What is NOT tested here is audio — sample rates, buffering and
playback smoothness are only answerable by speaking into it, and the spec
says so plainly rather than pretending otherwise.
"""

from __future__ import annotations

import contextlib
import threading
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from transport import browser
from transport.pcm_serializer import PCMFrameSerializer


def _close_result():
    return type("R", (), {"error": None, "summary": None, "ticket_id": None})()


def test_the_page_is_served_with_its_script_and_stylesheet():
    """A demo that 404s on its own assets is worse than no demo. This is the
    cheapest possible guard against a renamed file."""
    client = TestClient(browser.app)

    page = client.get("/")

    assert page.status_code == 200
    assert "app.js" in page.text
    assert "style.css" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


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


def test_a_real_websocket_connect_and_disconnect_runs_exactly_one_session(monkeypatch):
    """The one test that goes through the actual route.

    The three tests above call run_call(websocket=object()) — they prove the
    lifecycle *given* a socket, but they prove nothing about the wiring that
    produces one. The route registration, the accept(), the 16 kHz in/out
    sample rates and the PCMFrameSerializer are all things those tests would
    keep passing on if they were deleted. So this one opens a real WebSocket
    through TestClient, sends a real 20 ms frame, hangs up, and asserts what
    the server did with it.

    Only build_pipeline and _run_pipeline are stubbed, because those are the
    parts that would need Deepgram and Anthropic. The transport itself is the
    real FastAPIWebsocketTransport, constructed against the real socket.
    """
    monkeypatch.setenv("DEEPGRAM_API_KEY", "present-but-never-used-here")

    real_transport_cls = browser.FastAPIWebsocketTransport
    built: dict = {"frames": []}
    closed = []
    # The endpoint runs on the TestClient's own thread, so the assertions
    # need a happens-before edge rather than a hopeful sleep.
    finished = threading.Event()

    def _recording_transport(websocket, *, params):
        built["websocket"] = websocket
        built["params"] = params
        return real_transport_cls(websocket, params=params)

    def _capture_pipeline(transport, session):
        built["transport"] = transport
        built["session"] = session
        return object()

    async def _fake_run_pipeline(pipeline):
        # Stand in for the pipeline by doing the one thing a live call does
        # that a mocked socket cannot: hold the connection open, consume the
        # browser's audio, and return when the browser hangs up.
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            while True:
                built["frames"].append(await built["websocket"].receive_bytes())

    async def _fake_close(session):
        closed.append(session)
        finished.set()
        return _close_result()

    monkeypatch.setattr(browser, "FastAPIWebsocketTransport", _recording_transport)
    monkeypatch.setattr(browser, "build_pipeline", _capture_pipeline)
    monkeypatch.setattr(browser, "_run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr(browser, "close_session", _fake_close)

    # 320 Int16 samples: exactly one batch from static/capture-worklet.js.
    frame = b"\x11\x22" * 320
    client = TestClient(browser.app)
    with client.websocket_connect("/browser-stream") as websocket:
        websocket.send_bytes(frame)

    assert finished.wait(timeout=5), "the endpoint never finished the call"

    assert built["frames"] == [frame], "the route did not deliver the browser's audio"
    assert isinstance(built["transport"], real_transport_cls)
    assert built["params"].audio_in_enabled is True
    assert built["params"].audio_out_enabled is True
    # The rate the browser opens its AudioContext at. A mismatch here is the
    # failure mode that presents as "the agent never replies".
    assert built["params"].audio_in_sample_rate == browser.SAMPLE_RATE == 16000
    assert built["params"].audio_out_sample_rate == browser.SAMPLE_RATE == 16000
    assert isinstance(built["params"].serializer, PCMFrameSerializer)

    assert built["session"].transport == "browser"
    assert closed == [built["session"]], "a disconnect must close exactly one session"


def test_a_missing_deepgram_key_is_reported_rather_than_failing_silently(monkeypatch):
    """Without the pre-flight, a missing key produces a socket that is
    accepted and then dies on the first audio frame: the page flashes
    in-call -> ending -> idle with an empty status line, which is
    indistinguishable from a broken button. The spec calls that the worst
    outcome for a demo, so the server has to say what is wrong."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "")
    never_called = AsyncMock()
    monkeypatch.setattr(browser, "run_call", never_called)

    client = TestClient(browser.app)
    with client.websocket_connect("/browser-stream") as websocket:
        # A text frame, not binary — the page feeds binary straight into its
        # audio buffer, so an explanation has to arrive as something it can
        # tell apart from audio.
        message = websocket.receive_text()

    assert "DEEPGRAM_API_KEY" in message
    # No session may be created at all when the call cannot possibly work.
    never_called.assert_not_awaited()
