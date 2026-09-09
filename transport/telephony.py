"""Telephony transport — Phase 9 interface (Twilio Media Streams + Pipecat).

Routes a real phone call into the *same* Pipecat pipeline transport/
pipeline.py already runs over a local mic — transport/pipecat_processors.py's
build_pipeline() is shared between them unchanged, and everything downstream
of "raw audio in" (ClaudeTurnProcessor, TTS, LatencyLogger) is identical.
The only thing this file is responsible for is turning a real phone call
into a Pipecat `transport` object.

Twilio's call flow (confirmed against Twilio's actual docs and the
installed pipecat-ai/twilio source, not guessed):

  1. Twilio POSTs the incoming-call webhook (form-encoded, signed with an
     X-Twilio-Signature header) to POST /voice.
  2. This server validates that signature (twilio.request_validator.
     RequestValidator, the same HMAC-SHA1 scheme Twilio's own docs
     describe) and returns TwiML: <Connect><Stream url="wss://..."/>
     </Connect> — bidirectional (<Connect>, not the one-way <Start>) since
     the bot needs to talk back, not just listen.
  3. Twilio opens that WebSocket and sends "connected" then "start" (with
     streamSid/callSid/accountSid) — exactly what TwilioFrameSerializer
     needs at construction — followed by "media"/"dtmf" events, which
     Pipecat's own FastAPIWebsocketTransport takes over receiving from
     there. An EndFrame later triggers TwilioFrameSerializer's own
     auto_hang_up (a real Twilio REST call), and an InterruptionFrame
     becomes Twilio's own "clear" message — both already implemented in
     Pipecat, nothing to add here.

Every call defaults to DEFAULT_CUSTOMER_ID, same as every other transport —
there's no way to prompt for a customer ID over a phone call, and no
auth/caller-ID-lookup phase exists yet (a simplification already made
everywhere else in this project, stated plainly rather than silently).

Run with: python -m transport.telephony
Needs (in .env): TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, PUBLIC_HOSTNAME
(your ngrok hostname, no scheme, e.g. "abc123.ngrok-free.app").
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.workers.runner import WorkerRunner
from twilio.request_validator import RequestValidator
from twilio.twiml.voice_response import Connect, VoiceResponse

from agent.session import DEFAULT_CUSTOMER_ID, close_session, create_session
from transport.pipecat_processors import build_pipeline

PORT = int(os.getenv("PORT", "8765"))

app = FastAPI()


def _public_hostname() -> str:
    hostname = os.getenv("PUBLIC_HOSTNAME")
    if not hostname:
        raise RuntimeError(
            "PUBLIC_HOSTNAME is not set — needed to build the wss:// media-stream "
            "URL Twilio connects to (your ngrok hostname, no scheme)."
        )
    return hostname


# How long <Dial> rings the human before giving up. Twilio allows 5-600 and
# defaults to 30; 20 is long enough for a real pickup and short enough that a
# customer already waiting on hold is not abandoned to silence.
TRANSFER_TIMEOUT_SECONDS = 20


@dataclass(frozen=True)
class PendingTransfer:
    """One in-flight transfer, keyed by the customer's call SID.

    Exists because a transfer spans three separate HTTP interactions — the
    REST redirect, Twilio's request to /whisper, and its request to
    /transfer-status — and they need to share state the packet already has.
    """

    escalation_id: int | None
    whisper: str
    session_id: str


# Keyed by call_sid. In-process on purpose: this project runs one uvicorn
# worker (see __main__ at the foot of this file), and a distributed store
# would be machinery a mock project cannot justify. Entries are removed when
# the transfer resolves, so it cannot grow without bound.
TRANSFERS: dict[str, PendingTransfer] = {}


def render_whisper(packet: dict[str, Any]) -> str:
    """Turn a handoff packet into ~15 seconds of spoken briefing.

    Short on purpose: the human is holding a ringing phone and the customer
    is waiting on the other leg. Reason and intent come first because they
    are what decides how the human opens the conversation.

    The packet's free text was already redacted by guardrails/pii.py at
    create_handoff_packet (Phase 10a), so contact details arrive masked while
    the order ID survives — the right split, since the order ID is the thing
    that lets the human actually act.
    """
    parts = [f"Handoff {packet.get('escalation_id', 'unknown')}."]
    for label, key in (
        ("Reason", "reason"),
        ("Customer wants", "customer_intent"),
        ("Account", "verified_account_info"),
        ("Already done", "actions_taken"),
        ("Sentiment", "sentiment"),
    ):
        value = packet.get(key)
        if value:
            parts.append(f"{label}: {value}.")
    return " ".join(parts)


def remember_transfer(call_sid: str, packet: dict[str, Any], session_id: str) -> PendingTransfer:
    """Stash what /whisper and /transfer-status will need, before the
    redirect is issued."""
    pending = PendingTransfer(
        escalation_id=packet.get("escalation_id"),
        whisper=render_whisper(packet),
        session_id=session_id,
    )
    TRANSFERS[call_sid] = pending
    return pending


@app.post("/voice")
async def voice(request: Request) -> Response:
    """Twilio's incoming-call webhook. Validates the request signature
    before trusting anything in it, then returns TwiML connecting the call
    to a bidirectional Media Stream.
    """
    form = await request.form()
    signature = request.headers.get("X-Twilio-Signature", "")
    validator = RequestValidator(os.getenv("TWILIO_AUTH_TOKEN", ""))
    webhook_url = f"https://{_public_hostname()}/voice"
    if not validator.validate(webhook_url, dict(form), signature):
        raise HTTPException(status_code=403, detail="invalid Twilio request signature")

    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=f"wss://{_public_hostname()}/media-stream")
    response.append(connect)
    return Response(content=str(response), media_type="application/xml")


async def _read_start_event(receive_text: Callable[[], Awaitable[str]]) -> tuple[str, str, str]:
    """Drain Twilio's initial "connected" then "start" events, returning
    (stream_sid, call_sid, account_sid) — what TwilioFrameSerializer needs
    at construction, before Pipecat's own transport takes over receiving
    subsequent media/dtmf/stop events.

    `receive_text` is an async callable (`websocket.receive_text` in
    production; a canned sequence in tests) so this parsing logic is
    testable without a real WebSocket connection.
    """
    while True:
        message = json.loads(await receive_text())
        if message["event"] == "connected":
            continue
        if message["event"] == "start":
            start = message["start"]
            return start["streamSid"], start["callSid"], start["accountSid"]


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    stream_sid, call_sid, account_sid = await _read_start_event(websocket.receive_text)

    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=account_sid,
        auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
    )
    transport = FastAPIWebsocketTransport(
        websocket,
        params=FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True, serializer=serializer),
    )

    session = create_session(DEFAULT_CUSTOMER_ID, transport="telephony")
    pipeline = build_pipeline(transport, session)
    worker = PipelineWorker(pipeline, params=PipelineParams(enable_metrics=True))
    runner = WorkerRunner()
    await runner.add_workers(worker)

    await runner.run()

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
    uvicorn.run(app, host="0.0.0.0", port=PORT)
