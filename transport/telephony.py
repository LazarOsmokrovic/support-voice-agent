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
from twilio.rest import Client
from twilio.twiml.voice_response import Connect, Dial, VoiceResponse

from agent.session import DEFAULT_CUSTOMER_ID, Session, close_session, create_session
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


def build_transfer_twiml(escalation_id: int | None, human_number: str, caller_id: str | None) -> str:
    """The TwiML that replaces the Media Stream.

    <Number url=...> is Twilio's whisper: that TwiML runs on the CALLED
    party's end after they answer but before the two legs are bridged, so the
    human hears the briefing and the customer does not. It may not contain
    <Dial>.

    <Dial action=...> hands the parent call to /transfer-status when the dial
    ends, which is what makes the no-answer path possible — without it, the
    customer would simply be hung up on.

    The leading <Say> matters more than it looks: issuing the redirect cuts
    the Media Stream, which can truncate the agent's own spoken notice
    mid-word. This guarantees the customer hears something before ringing.
    """
    response = VoiceResponse()
    response.say("Connecting you now. Please hold.")
    dial = Dial(
        action=f"https://{_public_hostname()}/transfer-status",
        timeout=TRANSFER_TIMEOUT_SECONDS,
        caller_id=caller_id,
    )
    dial.number(human_number, url=f"https://{_public_hostname()}/whisper?escalation_id={escalation_id}")
    response.append(dial)
    return str(response)


async def transfer_to_human(
    call_sid: str,
    packet: dict[str, Any],
    session_id: str,
    *,
    client: Any | None = None,
) -> bool:
    """Redirect the customer's live call into a whispered <Dial>.

    Returns True if the redirect was issued, False otherwise. NEVER raises:
    every caller treats False as "carry on as before", so a broken transfer
    costs the customer a handoff, not the call.
    """
    human_number = os.getenv("HUMAN_AGENT_NUMBER")
    if not human_number:
        print("(no HUMAN_AGENT_NUMBER configured — skipping transfer)")
        return False

    if call_sid in TRANSFERS:
        print(f"(transfer already in flight for {call_sid} — ignoring duplicate)")
        return False

    remember_transfer(call_sid, packet, session_id)
    try:
        rest = client or Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
        twiml = build_transfer_twiml(packet.get("escalation_id"), human_number, os.getenv("TWILIO_CALLER_ID"))
        rest.calls(call_sid).update(twiml=twiml)
    except Exception as exc:  # noqa: BLE001 — a failed transfer must never drop the call
        TRANSFERS.pop(call_sid, None)
        print(f"(transfer to {human_number} failed: {exc})")
        return False
    print(f"(transferring {call_sid} to {human_number})")
    return True


async def _validate_twilio_signature(request: Request, path: str) -> dict[str, str]:
    """Shared by every Twilio-facing endpoint. Extracted rather than repeated
    because /whisper and /transfer-status must not drift from /voice's
    checking — a weaker check on the endpoint that SPEAKS a customer's
    briefing would be the worst place to have one.
    """
    form = await request.form()
    signature = request.headers.get("X-Twilio-Signature", "")
    validator = RequestValidator(os.getenv("TWILIO_AUTH_TOKEN", ""))
    if not validator.validate(f"https://{_public_hostname()}{path}", dict(form), signature):
        raise HTTPException(status_code=403, detail="invalid Twilio request signature")
    return dict(form)


@app.post("/voice")
async def voice(request: Request) -> Response:
    """Twilio's incoming-call webhook. Validates the request signature
    before trusting anything in it, then returns TwiML connecting the call
    to a bidirectional Media Stream.
    """
    await _validate_twilio_signature(request, "/voice")

    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=f"wss://{_public_hostname()}/media-stream")
    response.append(connect)
    return Response(content=str(response), media_type="application/xml")


@app.post("/whisper")
async def whisper(request: Request) -> Response:
    """Spoken to the human agent only, after they answer and before the two
    legs are bridged. Twilio requests this via the `url` attribute on
    <Number>; the customer never hears it.
    """
    form = await _validate_twilio_signature(request, "/whisper")
    pending = TRANSFERS.get(form.get("ParentCallSid", ""))
    text = pending.whisper if pending else "A customer is waiting. No context is available for this transfer."
    response = VoiceResponse()
    response.say(text)
    return Response(content=str(response), media_type="application/xml")


# Every DialCallStatus that means the human did NOT take the call. Twilio's
# full set is completed/answered/busy/no-answer/failed/canceled; the first two
# mean the bridge happened and the conversation is over.
_DIAL_FAILED = frozenset({"busy", "no-answer", "failed", "canceled"})


@app.post("/transfer-status")
async def transfer_status(request: Request) -> Response:
    """Twilio requests this when <Dial> ends, and from here the action URL —
    not the original TwiML — controls the parent call.

    On a failed dial the customer is still on the line, having waited through
    the ringing. Reconnecting the Media Stream with the ORIGINAL session id
    means the agent resumes with full history and can apologise and offer a
    callback, rather than greeting them from scratch as a stranger.
    """
    form = await _validate_twilio_signature(request, "/transfer-status")
    call_sid = form.get("CallSid", "")
    pending = TRANSFERS.pop(call_sid, None)
    status = form.get("DialCallStatus", "")

    response = VoiceResponse()
    if status in _DIAL_FAILED and pending is not None:
        print(f"(transfer for {call_sid} ended as {status!r} — returning the caller to the agent)")
        connect = Connect()
        connect.stream(url=f"wss://{_public_hostname()}/media-stream?session={pending.session_id}")
        response.append(connect)
    else:
        response.hangup()
    return Response(content=str(response), media_type="application/xml")


# Live sessions keyed by session_id, so a call that comes back from a failed
# transfer can resume the SAME conversation. In-process on purpose (one
# uvicorn worker); entries are removed by forget_session when the call ends.
SESSIONS: dict[str, Session] = {}


def resolve_session(session_id: str | None) -> Session:
    """Resume a session by id, or start a fresh one.

    An unknown id is not an error worth failing a live call over — the caller
    is on the phone right now. A cold start loses history; a 500 loses the
    customer.
    """
    if session_id and session_id in SESSIONS:
        print(f"(resuming session {session_id} after a failed transfer)")
        return SESSIONS[session_id]
    session = create_session(DEFAULT_CUSTOMER_ID, transport="telephony")
    SESSIONS[session.session_id] = session
    return session


def forget_session(session_id: str) -> None:
    SESSIONS.pop(session_id, None)


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

    session = resolve_session(websocket.query_params.get("session"))

    async def _on_escalation(packet: dict[str, Any] | None) -> bool:
        if packet is None:
            return False
        return await transfer_to_human(call_sid, packet, session.session_id)

    pipeline = build_pipeline(transport, session, on_escalation=_on_escalation)
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
    forget_session(session.session_id)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
