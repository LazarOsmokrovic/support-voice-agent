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
from hashlib import sha1

import httpx
import pytest

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
