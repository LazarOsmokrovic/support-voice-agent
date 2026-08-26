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
