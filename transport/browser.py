"""Browser voice console — Phase 10f interface.

The fourth transport to sit behind an unchanged build_pipeline, after the
text CLI (Phase 1), the local microphone (Phase 7) and Twilio (Phase 9).
Nothing in agent/ changes, and neither does transport/pipecat_processors.py:
it already accepts any BaseTransport, and this is the proof.

Why a WebSocket rather than WebRTC: Pipecat ships SmallWebRTCTransport,
which would talk to a browser directly, but it needs aiortc and its native
dependencies, which this project does not have. FastAPIWebsocketTransport is
already installed and already carries this project's Twilio traffic — Twilio
simply wraps it in a different serializer. Reusing it costs no new dependency
and reuses a transport already proven against real calls.

A pleasant consequence: no tunnel. Twilio had to reach back into this machine,
so Phase 9 needed ngrok and a PUBLIC_HOSTNAME. A browser dials outward to
localhost, so this demo runs entirely on one laptop with nothing exposed.

Run with: python -m transport.browser, then open http://localhost:8080.
Requires DEEPGRAM_API_KEY (and ANTHROPIC_API_KEY) in .env. No Twilio.
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.workers.runner import WorkerRunner

from agent.core import configure_logging
from agent.session import DEFAULT_CUSTOMER_ID, Session, close_session, create_session
from transport.pcm_serializer import PCMFrameSerializer
from transport.pipecat_processors import build_pipeline

# Deliberately not telephony's PORT (8765), so both transports can run at
# once without a collision during a demo.
BROWSER_PORT = int(os.getenv("BROWSER_PORT", "8080"))

# The browser opens its AudioContext at exactly this rate, so nothing
# resamples anywhere: not the browser, not the serializer, not the pipeline.
SAMPLE_RATE = 16000

_STATIC = Path(__file__).resolve().parent.parent / "static"

app = FastAPI()
# static/ now ships real files (index.html, app.js, style.css, the capture
# worklet), so the mount is held to the normal StaticFiles default: raise at
# import time if the directory is missing. A silently-tolerated missing
# directory would otherwise turn a broken deployment into a 404 at request
# time instead of a clear failure at startup.
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


async def _run_pipeline(pipeline) -> None:
    """Run one call's pipeline to completion.

    Same worker/runner shape transport/telephony.py already uses, rather than
    the PipelineTask API Pipecat also exposes — one convention per project.

    Separated from run_call so the lifecycle around it can be tested without
    standing up a real WebSocket and a real Pipecat run.
    """
    worker = PipelineWorker(pipeline, params=PipelineParams(enable_metrics=True))
    runner = WorkerRunner()
    await runner.add_workers(worker)
    await runner.run()


async def run_call(websocket: WebSocket) -> Session:
    """One button press to the next: a whole call.

    Every call gets a NEW session, deliberately. There is no resume here and
    no carried-over history — pressing the button again is a new customer
    ringing, not the previous one coming back. (Phase 10d's session resume
    exists for the opposite reason: surviving a failed transfer.)

    close_session runs in a finally block because the socket can close at any
    moment, including mid-turn while the model is still generating. Ending a
    call must always write its post-call summary; a hang-up is not an excuse
    to lose the ticket.
    """
    session = create_session(DEFAULT_CUSTOMER_ID, transport="browser")
    transport = FastAPIWebsocketTransport(
        websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=SAMPLE_RATE,
            audio_out_sample_rate=SAMPLE_RATE,
            serializer=PCMFrameSerializer(),
        ),
    )
    try:
        await _run_pipeline(build_pipeline(transport, session))
    except Exception as exc:  # noqa: BLE001 — a dropped socket is normal, not exceptional
        print(f"(call ended: {exc})")
    finally:
        close_result = await close_session(session)
        if close_result.error:
            print(f"({close_result.error})")
        elif close_result.summary is not None:
            print(
                f"Session logged as ticket #{close_result.ticket_id} "
                f"(sentiment={close_result.summary.sentiment}, "
                f"follow_up_needed={close_result.summary.follow_up_needed})"
            )
    return session


@app.websocket("/browser-stream")
async def browser_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    await run_call(websocket)


if __name__ == "__main__":
    configure_logging()
    print(f"Browser console on http://localhost:{BROWSER_PORT}")
    uvicorn.run(app, host="127.0.0.1", port=BROWSER_PORT)
