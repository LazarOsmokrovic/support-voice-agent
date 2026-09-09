"""Phase 9: transport/telephony.py's Twilio webhook and Media Streams
handshake — everything offline, no real Twilio account or network needed.

Covers: X-Twilio-Signature validation (valid/invalid/missing, using the same
HMAC-SHA1 scheme twilio.request_validator.RequestValidator implements — the
test signs requests by hand rather than importing RequestValidator itself,
so a bug in this project's *use* of it wouldn't be masked by reusing the
same code to both sign and check), the returned TwiML's shape, and
_read_start_event() parsing Twilio's actual WebSocket message shapes.

What this can't cover: a real phone call, a real Twilio Media Stream, or
the full Pipecat pipeline running end to end over one — that's this phase's
real checkpoint ("place a real call from your phone... run through
order-status, FAQ, and returns"), and has to be run by hand, same honest
limitation as every voice/telephony phase so far.
"""

from __future__ import annotations

import base64
import hmac
import json
import time
from hashlib import sha1

import httpx
import pytest

from agent.session import DEFAULT_CUSTOMER_ID, SessionCloseResult, create_session
from transport import telephony


def _sign(url: str, params: dict[str, str], auth_token: str) -> str:
    """Twilio's own signing scheme: the URL followed by every sorted
    param-name+value pair, HMAC-SHA1'd with the auth token, base64-encoded.
    """
    signed = url
    for key in sorted(params):
        signed += key + params[key]
    mac = hmac.new(auth_token.encode(), signed.encode(), sha1)
    return base64.b64encode(mac.digest()).decode()


async def _post_voice(form: dict[str, str], headers: dict[str, str]) -> httpx.Response:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=telephony.app), base_url="http://test") as client:
        return await client.post("/voice", data=form, headers=headers)


@pytest.mark.asyncio
async def test_voice_webhook_rejects_a_missing_signature(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-auth-token")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "abc123.ngrok-free.app")

    response = await _post_voice({"CallSid": "CA123"}, headers={})

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_voice_webhook_rejects_a_signature_signed_with_the_wrong_token(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "real-token")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "abc123.ngrok-free.app")
    form = {"CallSid": "CA123"}
    bad_signature = _sign("https://abc123.ngrok-free.app/voice", form, "wrong-token")

    response = await _post_voice(form, headers={"X-Twilio-Signature": bad_signature})

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_voice_webhook_accepts_a_validly_signed_request_and_returns_the_stream_twiml(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "real-token")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "abc123.ngrok-free.app")
    form = {"CallSid": "CA123", "From": "+15551234567"}
    signature = _sign("https://abc123.ngrok-free.app/voice", form, "real-token")

    response = await _post_voice(form, headers={"X-Twilio-Signature": signature})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    body = response.text
    assert "<Connect>" in body
    assert '<Stream url="wss://abc123.ngrok-free.app/media-stream"' in body


def _fake_receiver(messages: list[str]):
    iterator = iter(messages)

    async def receive() -> str:
        return next(iterator)

    return receive


@pytest.mark.asyncio
async def test_read_start_event_extracts_stream_call_and_account_sids():
    messages = [
        json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}),
        json.dumps(
            {
                "event": "start",
                "sequenceNumber": "1",
                "start": {
                    "accountSid": "AC1",
                    "streamSid": "MZ1",
                    "callSid": "CA1",
                    "tracks": ["inbound"],
                    "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
                },
                "streamSid": "MZ1",
            }
        ),
    ]

    stream_sid, call_sid, account_sid = await telephony._read_start_event(_fake_receiver(messages))

    assert stream_sid == "MZ1"
    assert call_sid == "CA1"
    assert account_sid == "AC1"


@pytest.mark.asyncio
async def test_read_start_event_skips_extra_connected_messages():
    """Defensive: only advance past `start` once it actually arrives, in case
    Twilio or a proxy ever sends more than one "connected" message.
    """
    messages = [
        json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}),
        json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}),
        json.dumps({"event": "start", "start": {"accountSid": "AC2", "streamSid": "MZ2", "callSid": "CA2"}}),
    ]

    stream_sid, call_sid, account_sid = await telephony._read_start_event(_fake_receiver(messages))

    assert stream_sid == "MZ2"
    assert call_sid == "CA2"
    assert account_sid == "AC2"


def test_render_whisper_briefs_the_human_from_the_packet():
    """The whisper is the entire point of a warm handoff — the human must
    hear who is waiting and why before the line opens."""
    packet = {
        "escalation_id": 42,
        "reason": "explicit request for a human",
        "customer_intent": "wants a refund outside the return window",
        "conversation_summary": "Asked about order status, then became frustrated.",
        "verified_account_info": "Maria Gonzalez, order 112-3487561-2938471",
        "actions_taken": "Looked up the order; explained the 30-day policy.",
        "sentiment": "negative",
    }
    whisper = telephony.render_whisper(packet)
    assert "42" in whisper
    assert "explicit request for a human" in whisper
    assert "refund outside the return window" in whisper
    assert "negative" in whisper


def test_render_whisper_survives_a_packet_missing_fields():
    """create_handoff_packet infers its fields from a model call, so a
    degraded packet is possible. A thin briefing beats a 500 that leaves
    the human hearing silence."""
    whisper = telephony.render_whisper({"escalation_id": 7})
    assert "7" in whisper
    assert whisper.strip()


def test_remember_transfer_stores_the_whisper_for_the_endpoint_to_read():
    """The /whisper endpoint runs in a SEPARATE HTTP request from the
    redirect, so the text has to outlive the call that built it. Stashing
    it here avoids re-reading the packet from SQLite, which would have
    meant adding a query to agent/ — forbidden this phase."""
    telephony.TRANSFERS.clear()
    packet = {"escalation_id": 9, "customer_intent": "billing question"}
    stored = telephony.remember_transfer("CA-test-sid", packet, "sess-1")
    assert telephony.TRANSFERS["CA-test-sid"] is stored
    assert stored.escalation_id == 9
    assert stored.session_id == "sess-1"
    assert "billing question" in stored.whisper


def test_build_transfer_twiml_dials_the_human_with_a_whisper_url(monkeypatch):
    """Asserted as parsed XML, not string matching — a test that greps for
    a substring passes on malformed TwiML that Twilio would reject."""
    import xml.etree.ElementTree as ET

    monkeypatch.setenv("PUBLIC_HOSTNAME", "example.ngrok.app")
    xml = telephony.build_transfer_twiml(42, "+15551234567", "+15559876543")
    root = ET.fromstring(xml)
    dial = root.find("Dial")
    assert dial is not None
    assert dial.get("timeout") == "20"
    assert "/transfer-status" in dial.get("action")
    number = dial.find("Number")
    assert number.text == "+15551234567"
    assert "/whisper" in number.get("url")
    assert "escalation_id=42" in number.get("url")


@pytest.mark.asyncio
async def test_transfer_to_human_issues_the_redirect(monkeypatch):
    monkeypatch.setenv("HUMAN_AGENT_NUMBER", "+15551234567")
    monkeypatch.setenv("TWILIO_CALLER_ID", "+15559876543")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "example.ngrok.app")
    telephony.TRANSFERS.clear()

    updated = {}

    class _FakeCalls:
        def __init__(self, sid):
            self.sid = sid

        def update(self, **kwargs):
            updated.update({"sid": self.sid, **kwargs})

    fake_client = type("C", (), {"calls": staticmethod(lambda sid: _FakeCalls(sid))})()

    ok = await telephony.transfer_to_human("CA-1", {"escalation_id": 42}, "sess-1", client=fake_client)

    assert ok is True
    assert updated["sid"] == "CA-1"
    assert "+15551234567" in updated["twiml"]
    assert telephony.TRANSFERS["CA-1"].escalation_id == 42


@pytest.mark.asyncio
async def test_transfer_to_human_is_a_no_op_without_a_configured_number(monkeypatch):
    """Optional-by-default, exactly like ESCALATION_WEBHOOK_URL. A developer
    with no human agent configured must still get a working agent."""
    monkeypatch.delenv("HUMAN_AGENT_NUMBER", raising=False)
    called = False

    def _boom(sid):
        nonlocal called
        called = True
        raise AssertionError("must not touch Twilio without a number configured")

    fake_client = type("C", (), {"calls": staticmethod(_boom)})()
    assert await telephony.transfer_to_human("CA-1", {}, "s", client=fake_client) is False
    assert called is False


@pytest.mark.asyncio
async def test_transfer_to_human_returns_false_when_twilio_rejects(monkeypatch):
    """A failed transfer must degrade to today's behaviour, never drop the
    call. The caller keeps the agent; it does not raise."""
    monkeypatch.setenv("HUMAN_AGENT_NUMBER", "+15551234567")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "example.ngrok.app")

    class _FailingCalls:
        def update(self, **kwargs):
            raise RuntimeError("call is no longer in-progress")

    fake_client = type("C", (), {"calls": staticmethod(lambda sid: _FailingCalls())})()
    assert await telephony.transfer_to_human("CA-1", {"escalation_id": 1}, "s", client=fake_client) is False


@pytest.mark.asyncio
async def test_transfer_to_human_fires_only_once_per_call(monkeypatch):
    """A model escalation immediately followed by a DTMF press must not
    redirect twice — the second would land on a call already dialling."""
    monkeypatch.setenv("HUMAN_AGENT_NUMBER", "+15551234567")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "example.ngrok.app")
    telephony.TRANSFERS.clear()
    calls = []

    class _Calls:
        def update(self, **kwargs):
            calls.append(kwargs)

    fake_client = type("C", (), {"calls": staticmethod(lambda sid: _Calls())})()

    first = await telephony.transfer_to_human("CA-1", {"escalation_id": 1}, "s", client=fake_client)
    second = await telephony.transfer_to_human("CA-1", {"escalation_id": 2}, "s", client=fake_client)

    assert first is True
    assert second is False
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_whisper_speaks_the_briefing_to_the_human(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "example.ngrok.app")
    telephony.TRANSFERS.clear()
    telephony.remember_transfer("CA-1", {"escalation_id": 42, "customer_intent": "refund dispute"}, "s1")

    url = "https://example.ngrok.app/whisper"
    params = {"CallSid": "CA-whisper-leg", "ParentCallSid": "CA-1"}
    signature = _sign(url, params, "test-token")

    transport_ = httpx.ASGITransport(app=telephony.app)
    async with httpx.AsyncClient(transport=transport_, base_url="https://example.ngrok.app") as client:
        response = await client.post("/whisper", data=params, headers={"X-Twilio-Signature": signature})

    assert response.status_code == 200
    assert "refund dispute" in response.text
    assert "<Say>" in response.text


@pytest.mark.asyncio
async def test_whisper_rejects_an_unsigned_request_and_leaks_nothing(monkeypatch):
    """This endpoint speaks a customer's handoff briefing aloud. The
    signature check is the ONLY access control on it — without it, anyone
    who guesses the URL can read intent, summary and account info."""
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "example.ngrok.app")
    telephony.TRANSFERS.clear()
    telephony.remember_transfer("CA-1", {"escalation_id": 42, "customer_intent": "refund dispute"}, "s1")

    transport_ = httpx.ASGITransport(app=telephony.app)
    async with httpx.AsyncClient(transport=transport_, base_url="https://example.ngrok.app") as client:
        response = await client.post(
            "/whisper", data={"ParentCallSid": "CA-1"}, headers={"X-Twilio-Signature": "wrong"}
        )

    assert response.status_code == 403
    assert "refund dispute" not in response.text


@pytest.mark.asyncio
async def test_whisper_accepts_a_signed_request_carrying_the_escalation_id_query_string(monkeypatch):
    """Twilio signs the FULL URL it requests, query string included, and
    build_transfer_twiml puts `?escalation_id=` on every /whisper callback.
    An earlier version of _validate_twilio_signature checked a reconstructed
    BARE PATH, which 403'd every real whisper request — and Twilio treats a
    whisper-URL error as "no whisper" and bridges the legs anyway, so the
    warm handoff silently degraded into a blind transfer on every call, with
    nothing an operator could see.

    This signs the URL WITH its query string, exactly as Twilio does, so it
    fails against the pre-fix code (confirmed by hand: `git stash` the fix,
    run this test, watch it 403; `git stash pop` to restore it) and only
    passes once the signature check validates the query string too.
    """
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "example.ngrok.app")
    telephony.TRANSFERS.clear()
    telephony.remember_transfer("CA-1", {"escalation_id": 42, "customer_intent": "refund dispute"}, "s1")

    url = "https://example.ngrok.app/whisper?escalation_id=42"
    params = {"CallSid": "CA-whisper-leg", "ParentCallSid": "CA-1"}
    signature = _sign(url, params, "test-token")

    transport_ = httpx.ASGITransport(app=telephony.app)
    async with httpx.AsyncClient(transport=transport_, base_url="https://example.ngrok.app") as client:
        response = await client.post(
            "/whisper?escalation_id=42", data=params, headers={"X-Twilio-Signature": signature}
        )

    assert response.status_code == 200
    assert "refund dispute" in response.text


@pytest.mark.asyncio
async def test_whisper_rejects_a_wrongly_signed_query_string_request_and_leaks_nothing(monkeypatch):
    """A signature computed for a DIFFERENT query string than the one
    actually requested — exactly what an attacker guessing escalation ids
    would send — must still 403, and the briefing must not leak into the
    error response."""
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "example.ngrok.app")
    telephony.TRANSFERS.clear()
    telephony.remember_transfer("CA-1", {"escalation_id": 42, "customer_intent": "refund dispute"}, "s1")

    params = {"CallSid": "CA-whisper-leg", "ParentCallSid": "CA-1"}
    wrong_signature = _sign("https://example.ngrok.app/whisper?escalation_id=99", params, "test-token")

    transport_ = httpx.ASGITransport(app=telephony.app)
    async with httpx.AsyncClient(transport=transport_, base_url="https://example.ngrok.app") as client:
        response = await client.post(
            "/whisper?escalation_id=42", data=params, headers={"X-Twilio-Signature": wrong_signature}
        )

    assert response.status_code == 403
    assert "refund dispute" not in response.text


@pytest.mark.asyncio
async def test_whisper_falls_back_when_the_transfer_is_unknown(monkeypatch):
    """A process restart between redirect and whisper loses the registry.
    The human should still get a usable call, not silence."""
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "example.ngrok.app")
    telephony.TRANSFERS.clear()

    url = "https://example.ngrok.app/whisper"
    params = {"ParentCallSid": "CA-unknown"}
    signature = _sign(url, params, "test-token")

    transport_ = httpx.ASGITransport(app=telephony.app)
    async with httpx.AsyncClient(transport=transport_, base_url="https://example.ngrok.app") as client:
        response = await client.post("/whisper", data=params, headers={"X-Twilio-Signature": signature})

    assert response.status_code == 200
    assert "<Say>" in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["busy", "no-answer", "failed", "canceled"])
async def test_transfer_status_returns_the_customer_to_the_agent(monkeypatch, status):
    """The human did not pick up. The customer has been holding — bringing
    them back to an agent that REMEMBERS the conversation is the whole point
    of passing the session id through."""
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "example.ngrok.app")
    telephony.TRANSFERS.clear()
    telephony.remember_transfer("CA-1", {"escalation_id": 42}, "sess-abc")

    url = "https://example.ngrok.app/transfer-status"
    params = {"CallSid": "CA-1", "DialCallStatus": status}
    signature = _sign(url, params, "test-token")

    transport_ = httpx.ASGITransport(app=telephony.app)
    async with httpx.AsyncClient(transport=transport_, base_url="https://example.ngrok.app") as client:
        response = await client.post("/transfer-status", data=params, headers={"X-Twilio-Signature": signature})

    assert response.status_code == 200
    assert "<Stream" in response.text
    assert "session=sess-abc" in response.text
    assert "CA-1" not in telephony.TRANSFERS


@pytest.mark.asyncio
async def test_transfer_status_hangs_up_after_a_completed_transfer(monkeypatch):
    """The human answered and the call is over. Reconnecting the AI here
    would drop a finished conversation back onto a bot."""
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "example.ngrok.app")
    telephony.TRANSFERS.clear()
    telephony.remember_transfer("CA-1", {"escalation_id": 42}, "sess-abc")

    url = "https://example.ngrok.app/transfer-status"
    params = {"CallSid": "CA-1", "DialCallStatus": "completed"}
    signature = _sign(url, params, "test-token")

    transport_ = httpx.ASGITransport(app=telephony.app)
    async with httpx.AsyncClient(transport=transport_, base_url="https://example.ngrok.app") as client:
        response = await client.post("/transfer-status", data=params, headers={"X-Twilio-Signature": signature})

    assert "<Hangup" in response.text
    assert "<Stream" not in response.text


@pytest.mark.asyncio
async def test_transfer_status_rejects_an_unsigned_request(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "example.ngrok.app")

    transport_ = httpx.ASGITransport(app=telephony.app)
    async with httpx.AsyncClient(transport=transport_, base_url="https://example.ngrok.app") as client:
        response = await client.post(
            "/transfer-status",
            data={"CallSid": "CA-1", "DialCallStatus": "no-answer"},
            headers={"X-Twilio-Signature": "wrong"},
        )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_transfer_status_accepts_a_signed_request_carrying_the_escalation_id_query_string(monkeypatch):
    """build_transfer_twiml also puts `?escalation_id=` on the <Dial action>
    URL, so /transfer-status needs the same query-string-aware signature
    check as /whisper. Signed here exactly as Twilio signs it — URL
    including the query string — so this would 403 against the pre-fix
    bare-path check."""
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "example.ngrok.app")
    telephony.TRANSFERS.clear()
    telephony.remember_transfer("CA-1", {"escalation_id": 42}, "sess-abc")

    url = "https://example.ngrok.app/transfer-status?escalation_id=42"
    params = {"CallSid": "CA-1", "DialCallStatus": "no-answer"}
    signature = _sign(url, params, "test-token")

    transport_ = httpx.ASGITransport(app=telephony.app)
    async with httpx.AsyncClient(transport=transport_, base_url="https://example.ngrok.app") as client:
        response = await client.post(
            "/transfer-status?escalation_id=42", data=params, headers={"X-Twilio-Signature": signature}
        )

    assert response.status_code == 200
    assert "<Stream" in response.text
    assert "session=sess-abc" in response.text


def test_session_registry_resumes_a_known_session():
    """After a failed transfer the customer comes back on a NEW Media Stream.
    Without this they would meet a brand-new session that has forgotten the
    entire conversation — worse than never attempting the transfer."""
    telephony.SESSIONS.clear()
    session = create_session(DEFAULT_CUSTOMER_ID, transport="telephony")
    telephony.SESSIONS[session.session_id] = session

    assert telephony.resolve_session(session.session_id) is session


def test_session_registry_falls_back_to_a_new_session_for_an_unknown_id():
    """A cold restart is worse than resuming, but far better than a 500 and
    a dropped call."""
    telephony.SESSIONS.clear()
    resumed = telephony.resolve_session("no-such-session")
    assert resumed is not None
    assert resumed.session_id != "no-such-session"


def test_session_registry_forgets_a_session_when_it_closes():
    """The registry is in-process and unbounded otherwise."""
    telephony.SESSIONS.clear()
    session = create_session(DEFAULT_CUSTOMER_ID, transport="telephony")
    telephony.SESSIONS[session.session_id] = session
    telephony.forget_session(session.session_id)
    assert session.session_id not in telephony.SESSIONS


def _fake_twilio_calls_client() -> object:
    """A Twilio REST client stub that accepts .calls(sid).update(twiml=...)
    without touching the network — what every transfer_to_human() call in
    these tests needs to drive the real redirect path."""

    class _Calls:
        def update(self, **kwargs):
            pass

    return type("C", (), {"calls": staticmethod(lambda sid: _Calls())})()


@pytest.mark.asyncio
async def test_transfer_to_human_detaches_the_session_so_media_stream_teardown_leaves_it_alone(monkeypatch):
    """The REST redirect transfer_to_human() issues is itself what ends the
    Media Stream — media_stream()'s teardown runs ~20 seconds before
    /transfer-status says whether the human answered. Before the fix that
    teardown closed the session outright, mid-transfer. Drives the real
    transfer_to_human() (not a hand-set DETACHED entry) and then reproduces
    media_stream()'s own teardown check against the real registries."""
    monkeypatch.setenv("HUMAN_AGENT_NUMBER", "+15551234567")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "example.ngrok.app")
    telephony.TRANSFERS.clear()
    telephony.SESSIONS.clear()
    telephony.DETACHED.clear()

    session = create_session(DEFAULT_CUSTOMER_ID, transport="telephony")
    telephony.SESSIONS[session.session_id] = session

    ok = await telephony.transfer_to_human(
        "CA-1", {"escalation_id": 1}, session.session_id, client=_fake_twilio_calls_client()
    )
    assert ok is True
    assert session.session_id in telephony.DETACHED

    # media_stream()'s own teardown, reproduced rather than driving a real
    # WebSocket: it must skip _close_and_forget for a detached session.
    if session.session_id not in telephony.DETACHED:
        await telephony._close_and_forget(session)

    assert session.session_id in telephony.SESSIONS


@pytest.mark.asyncio
async def test_transfer_status_failed_dial_keeps_session_alive_for_resolve_session_to_resume(monkeypatch):
    """Traces the real lifecycle end to end: transfer_to_human() (not a
    hand-set marker) detaches the session, then a failed-dial
    /transfer-status callback must keep it alive rather than closing it.
    Before the fix, media_stream()'s teardown would already have closed and
    forgotten the session ~20s earlier, so the reconnect found nothing and
    resolve_session() built a fresh one — the customer explained their
    problem, held through the ringing, got nobody, and was greeted from
    scratch. Asserts resolve_session() returns the SAME object (identity),
    conversation history intact."""
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("HUMAN_AGENT_NUMBER", "+15551234567")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "example.ngrok.app")
    telephony.TRANSFERS.clear()
    telephony.SESSIONS.clear()
    telephony.DETACHED.clear()

    session = create_session(DEFAULT_CUSTOMER_ID, transport="telephony")
    session.agent.messages.append({"role": "user", "content": "I need help with my order"})
    telephony.SESSIONS[session.session_id] = session

    ok = await telephony.transfer_to_human(
        "CA-1", {"escalation_id": 42}, session.session_id, client=_fake_twilio_calls_client()
    )
    assert ok is True

    url = "https://example.ngrok.app/transfer-status"
    params = {"CallSid": "CA-1", "DialCallStatus": "no-answer"}
    signature = _sign(url, params, "test-token")

    transport_ = httpx.ASGITransport(app=telephony.app)
    async with httpx.AsyncClient(transport=transport_, base_url="https://example.ngrok.app") as client:
        response = await client.post("/transfer-status", data=params, headers={"X-Twilio-Signature": signature})

    assert response.status_code == 200
    assert session.session_id in telephony.SESSIONS

    resumed = telephony.resolve_session(session.session_id)

    assert resumed is session
    assert resumed.agent.messages == [{"role": "user", "content": "I need help with my order"}]
    assert session.session_id not in telephony.DETACHED


@pytest.mark.asyncio
async def test_transfer_status_completed_dial_closes_the_session_exactly_once(monkeypatch):
    """The human took the call and it is over. media_stream()'s teardown
    already declined to close it (the session was detached), so
    /transfer-status on a completed dial is the only place left that can
    write the post-call summary ticket — and it must do so exactly once."""
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("HUMAN_AGENT_NUMBER", "+15551234567")
    monkeypatch.setenv("PUBLIC_HOSTNAME", "example.ngrok.app")
    telephony.TRANSFERS.clear()
    telephony.SESSIONS.clear()
    telephony.DETACHED.clear()

    close_calls = []

    async def _fake_close_session(session):
        close_calls.append(session.session_id)
        return SessionCloseResult(summary=None, ticket_id=1)

    monkeypatch.setattr(telephony, "close_session", _fake_close_session)

    session = create_session(DEFAULT_CUSTOMER_ID, transport="telephony")
    telephony.SESSIONS[session.session_id] = session

    ok = await telephony.transfer_to_human(
        "CA-1", {"escalation_id": 42}, session.session_id, client=_fake_twilio_calls_client()
    )
    assert ok is True

    url = "https://example.ngrok.app/transfer-status"
    params = {"CallSid": "CA-1", "DialCallStatus": "completed"}
    signature = _sign(url, params, "test-token")

    transport_ = httpx.ASGITransport(app=telephony.app)
    async with httpx.AsyncClient(transport=transport_, base_url="https://example.ngrok.app") as client:
        response = await client.post("/transfer-status", data=params, headers={"X-Twilio-Signature": signature})

    assert "<Hangup" in response.text
    assert close_calls == [session.session_id]
    assert session.session_id not in telephony.SESSIONS
    assert session.session_id not in telephony.DETACHED


@pytest.mark.asyncio
async def test_close_and_forget_is_idempotent_so_one_call_writes_only_one_ticket(monkeypatch):
    """Two paths can race to end the same transferred call (a dial that
    fails instantly can fire /transfer-status and the resumed Media
    Stream's own teardown close together) — this is the guard that stops
    one call producing two post-call summary tickets."""
    telephony.SESSIONS.clear()
    telephony.DETACHED.clear()

    close_calls = []

    async def _fake_close_session(session):
        close_calls.append(session.session_id)
        return SessionCloseResult(summary=None, ticket_id=1)

    monkeypatch.setattr(telephony, "close_session", _fake_close_session)

    session = create_session(DEFAULT_CUSTOMER_ID, transport="telephony")
    telephony.SESSIONS[session.session_id] = session
    telephony.DETACHED[session.session_id] = time.monotonic() + 3600

    await telephony._close_and_forget(session)
    await telephony._close_and_forget(session)

    assert close_calls == [session.session_id]
    assert session.session_id not in telephony.SESSIONS
    assert session.session_id not in telephony.DETACHED


@pytest.mark.asyncio
async def test_sweep_detached_sessions_abandons_expired_and_leaves_live_alone(monkeypatch):
    """Without this sweep, a /transfer-status callback that never arrives
    (Twilio can't reach us, the tunnel died) would pin its session in
    memory for the life of the process. A transfer still legitimately in
    flight (deadline not yet passed) must be left untouched."""
    telephony.SESSIONS.clear()
    telephony.DETACHED.clear()
    telephony.TRANSFERS.clear()

    close_calls = []

    async def _fake_close_session(session):
        close_calls.append(session.session_id)
        return SessionCloseResult(summary=None, ticket_id=1)

    monkeypatch.setattr(telephony, "close_session", _fake_close_session)

    expired = create_session(DEFAULT_CUSTOMER_ID, transport="telephony")
    live = create_session(DEFAULT_CUSTOMER_ID, transport="telephony")
    telephony.SESSIONS[expired.session_id] = expired
    telephony.SESSIONS[live.session_id] = live
    telephony.DETACHED[expired.session_id] = time.monotonic() - 1
    telephony.DETACHED[live.session_id] = time.monotonic() + 3600

    await telephony._sweep_detached_sessions()

    assert close_calls == [expired.session_id]
    assert expired.session_id not in telephony.SESSIONS
    assert expired.session_id not in telephony.DETACHED
    assert live.session_id in telephony.SESSIONS
    assert live.session_id in telephony.DETACHED
