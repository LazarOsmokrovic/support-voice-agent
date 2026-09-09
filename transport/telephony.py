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
import time
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

    Both callback URLs carry `?escalation_id=`, as the design spec's TwiML
    does. It is deliberately not load-bearing — /whisper keys off
    ParentCallSid and /transfer-status off CallSid, both of which Twilio
    posts in the form body — but it makes Twilio's own request log say which
    handoff a callback belongs to, which is the only place an operator can
    look when a transfer misbehaves on a live call. That is only safe
    because _signed_url() now validates the URL *including* its query
    string; validating a reconstructed bare path 403'd every one of these.
    """
    query = "" if escalation_id is None else f"?escalation_id={escalation_id}"
    response = VoiceResponse()
    response.say("Connecting you now. Please hold.")
    dial = Dial(
        action=f"https://{_public_hostname()}/transfer-status{query}",
        timeout=TRANSFER_TIMEOUT_SECONDS,
        caller_id=caller_id,
    )
    dial.number(human_number, url=f"https://{_public_hostname()}/whisper{query}")
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
    # The redirect that just succeeded is itself what ends the Media Stream,
    # so media_stream()'s own teardown is about to run — ~20 seconds before
    # /transfer-status says whether the customer is coming back. Marking the
    # session detached here (with no await in between, so the teardown cannot
    # interleave) is what stops that teardown closing a session the reconnect
    # still needs. Done only on success: a failed redirect leaves the Media
    # Stream up and the call ends normally.
    detach_session(session_id)
    print(f"(transferring {call_sid} to {human_number})")
    return True


def _signed_url(request: Request) -> str:
    """Rebuild the exact URL Twilio signed for this request.

    Twilio signs the FULL URL it requested, query string included. An earlier
    version of this function reconstructed a bare path, which 403'd every
    signed request carrying a query string — and Twilio treats a whisper-URL
    error as "no whisper" and bridges the legs anyway, so the warm handoff
    would have degraded silently into the blind transfer this phase exists to
    prevent, with nothing an operator could see.

    Scheme and host come from PUBLIC_HOSTNAME rather than from the incoming
    request on purpose: behind ngrok (or any TLS-terminating proxy) the
    request arrives as plain `http` on an internal hostname, while Twilio
    signed the public `https` URL. Trusting request.url there would
    reintroduce the same bug in a subtler form.

    Path and query come from the request, so this covers /voice, /whisper and
    /transfer-status alike — including a query string added to any of them
    later, with nobody having to remember this function exists.
    """
    query = request.url.query
    return f"https://{_public_hostname()}{request.url.path}" + (f"?{query}" if query else "")


async def _validate_twilio_signature(request: Request) -> dict[str, str]:
    """Shared by every Twilio-facing endpoint. Extracted rather than repeated
    because /whisper and /transfer-status must not drift from /voice's
    checking — a weaker check on the endpoint that SPEAKS a customer's
    briefing would be the worst place to have one.

    It takes no path argument: the URL it checks is derived entirely from the
    request, so an endpoint can never be registered with the wrong one.
    """
    form = await request.form()
    signature = request.headers.get("X-Twilio-Signature", "")
    validator = RequestValidator(os.getenv("TWILIO_AUTH_TOKEN", ""))
    if not validator.validate(_signed_url(request), dict(form), signature):
        raise HTTPException(status_code=403, detail="invalid Twilio request signature")
    return dict(form)


@app.post("/voice")
async def voice(request: Request) -> Response:
    """Twilio's incoming-call webhook. Validates the request signature
    before trusting anything in it, then returns TwiML connecting the call
    to a bidirectional Media Stream.
    """
    await _validate_twilio_signature(request)

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
    form = await _validate_twilio_signature(request)
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

    This is also where a transferred call's session lifecycle is decided,
    because media_stream()'s teardown deliberately does nothing while a
    transfer is in flight (see transfer_to_human): a completed dial is the
    real end of the call and closes the session here, while a failed dial
    keeps it alive for the reconnected stream to close when it ends.
    """
    form = await _validate_twilio_signature(request)
    call_sid = form.get("CallSid", "")
    pending = TRANSFERS.pop(call_sid, None)
    status = form.get("DialCallStatus", "")

    response = VoiceResponse()
    if status in _DIAL_FAILED and pending is not None:
        print(f"(transfer for {call_sid} ended as {status!r} — returning the caller to the agent)")
        # Refresh the detached marker rather than clearing it: this callback
        # can arrive before the Media Stream's own teardown (a dial that fails
        # instantly), and that teardown must still leave the session alone.
        # resolve_session() clears it when the customer actually comes back.
        detach_session(pending.session_id)
        connect = Connect()
        connect.stream(url=f"wss://{_public_hostname()}/media-stream?session={pending.session_id}")
        response.append(connect)
    else:
        if pending is not None:
            # The human took the call and it is now over. The Media Stream
            # ended at transfer time without closing anything, so this is the
            # only place the post-call summary ticket can be written.
            print(f"(transfer for {call_sid} ended as {status!r} — closing the session)")
            session = SESSIONS.get(pending.session_id)
            if session is not None:
                await _close_and_forget(session)
        elif status in _DIAL_FAILED:
            # Not the normal failure path — that one has a registry entry.
            # Reaching here means the process restarted between the redirect
            # and this callback, so there is no session id to reconnect to.
            # Logged because the customer is hung up on mid-problem and an
            # operator otherwise cannot tell this apart from a clean goodbye.
            print(
                f"(transfer for {call_sid} ended as {status!r} but no transfer is on record — "
                "the process likely restarted mid-transfer; hanging up rather than reconnecting "
                "the caller to a session that no longer exists)"
            )
        response.hangup()
    return Response(content=str(response), media_type="application/xml")


# Live sessions keyed by session_id, so a call that comes back from a failed
# transfer can resume the SAME conversation. In-process on purpose (one
# uvicorn worker); entries are removed by forget_session when the call ends.
SESSIONS: dict[str, Session] = {}


# Sessions that are alive but have no Media Stream attached, mapped to the
# monotonic deadline past which they are abandoned. A session lands here for
# the length of a transfer: the REST redirect ends the Media Stream long
# before Twilio says whether the human answered, so "the WebSocket closed"
# stops meaning "the call is over" and this is what tells the two apart.
DETACHED: dict[str, float] = {}

# A detached session outlives its Media Stream for as long as the human is
# talking to the customer, which can be a long conversation — so this bound
# is deliberately far longer than TRANSFER_TIMEOUT_SECONDS. It exists only so
# a /transfer-status callback that never arrives (Twilio cannot reach us, the
# tunnel died) cannot pin a session in memory for the life of the process.
DETACHED_SESSION_MAX_SECONDS = 3600.0


def detach_session(session_id: str) -> None:
    """Mark a session as deliberately outliving its Media Stream."""
    DETACHED[session_id] = time.monotonic() + DETACHED_SESSION_MAX_SECONDS


def resolve_session(session_id: str | None) -> Session:
    """Resume a session by id, or start a fresh one.

    An unknown id is not an error worth failing a live call over — the caller
    is on the phone right now. A cold start loses history; a 500 loses the
    customer.
    """
    if session_id and session_id in SESSIONS:
        print(f"(resuming session {session_id} after a failed transfer)")
        # It has a Media Stream again, so the ordinary teardown owns it once
        # more and the abandonment sweep must leave it alone.
        DETACHED.pop(session_id, None)
        return SESSIONS[session_id]
    session = create_session(DEFAULT_CUSTOMER_ID, transport="telephony")
    SESSIONS[session.session_id] = session
    return session


def forget_session(session_id: str) -> None:
    SESSIONS.pop(session_id, None)
    DETACHED.pop(session_id, None)


async def _close_and_forget(session: Session) -> None:
    """Close a session exactly once, whichever path ends its call.

    Three paths can be the end of one call — the Media Stream closing, a
    completed dial reported to /transfer-status, and the abandonment sweep —
    and two of them can even race (a dial that fails instantly). Removing the
    registry entry FIRST and closing only if it was still there makes this
    idempotent, so one call can never write two post-call summary tickets.
    """
    DETACHED.pop(session.session_id, None)
    if SESSIONS.pop(session.session_id, None) is None:
        return

    close_result = await close_session(session)
    if close_result.error:
        print(f"({close_result.error})")
    elif close_result.summary is not None:
        print(
            f"Session logged as ticket #{close_result.ticket_id} "
            f"(sentiment={close_result.summary.sentiment}, "
            f"follow_up_needed={close_result.summary.follow_up_needed})"
        )


async def _sweep_detached_sessions() -> None:
    """Close and drop detached sessions whose transfer never resolved.

    Without this, a /transfer-status callback that never arrives would leave
    its session in SESSIONS for the life of the process — the unbounded growth
    the registry's own comment promises cannot happen. Run when a new call
    arrives, which is both the moment memory starts mattering again and the
    only regularly-scheduled event this single-process app has.
    """
    now = time.monotonic()
    for session_id, deadline in list(DETACHED.items()):
        if deadline > now:
            continue
        print(f"(abandoning detached session {session_id} — its transfer never reported a status)")
        for call_sid, pending in list(TRANSFERS.items()):
            if pending.session_id == session_id:
                TRANSFERS.pop(call_sid, None)
        session = SESSIONS.get(session_id)
        if session is None:
            DETACHED.pop(session_id, None)
        else:
            await _close_and_forget(session)


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
    await _sweep_detached_sessions()
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

    # Reaching here does NOT necessarily mean the call is over. The REST
    # redirect a transfer issues is itself what ends this Media Stream, so on
    # a transferred call this runs ~20 seconds BEFORE /transfer-status says
    # whether the human answered. Closing here would write a post-call summary
    # ticket mid-call and delete the very session the reconnect resumes — the
    # customer would explain their problem, hold through the ringing, get
    # nobody, and then be greeted from scratch by an agent that forgot them.
    # /transfer-status owns the close in that case: on a completed dial, and
    # on a failed one after the resumed conversation finally ends here.
    if session.session_id in DETACHED:
        print(f"(media stream for {call_sid} ended with a transfer in flight — keeping the session alive)")
        return

    await _close_and_forget(session)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
