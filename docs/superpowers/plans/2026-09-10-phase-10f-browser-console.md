# Phase 10f — Browser Voice Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One browser page with one circle button: press to hold a real spoken conversation with the agent, press again to hang up, press again for a genuinely fresh conversation.

**Architecture:** The browser streams 16 kHz Int16 microphone PCM over a plain WebSocket to FastAPI, which feeds it into the *unchanged* Pipecat pipeline via `FastAPIWebsocketTransport` and a small raw-PCM serializer. This is the same transport class that already carries this project's Twilio traffic — Twilio merely wraps it in a different serializer — so browser becomes the fourth transport behind an unchanged `build_pipeline`, with no new Python dependency and no tunnel.

**Tech Stack:** Python 3.12+, FastAPI, Pipecat (`FastAPIWebsocketTransport`, `FrameSerializer`), Deepgram Flux + Aura-2, pytest + pytest-asyncio. Browser side: vanilla JS, Web Audio API (`AudioContext`, `AudioWorklet`), no framework, no build step, no npm.

**Spec:** `docs/superpowers/specs/2026-09-10-phase-10f-browser-console-design.md`

## Global Constraints

- **ZERO changes under `agent/`.** If a task seems to need one, stop and report — it means the seam is wrong. `transport/pipecat_processors.py` must not change either: it already accepts any `BaseTransport`, which is the point being proven.
- **NO new Python dependencies.** `aiortc` is deliberately not used; verified absent and rejected in the spec. If a task appears to need a new package, stop and report.
- **Audio format is 16 kHz, mono, Int16 little-endian, both directions.** The browser opens its `AudioContext` at 16000 Hz so no resampling is needed anywhere.
- **Every call is a fresh session.** Each WebSocket connection calls `create_session()`; two sequential calls must share no history, no `PendingActionGate` state, and no `session_id`. This is the opposite of Phase 10d's resume behaviour.
- **Hanging up runs the real teardown.** Disconnect ends the pipeline and calls `close_session`, writing the post-call summary ticket, exactly as a hung-up phone call does.
- `transport/browser.py` uses `BROWSER_PORT` (default `8080`), never telephony's `PORT` (default `8765`), so both can run at once.
- **No ngrok, no `PUBLIC_HOSTNAME`.** The browser connects to localhost.
- Never hard-code a seeded literal (order ID, customer ID, email, phone, tracking number) in a test assertion — read it from `data/mock_db.py`. Two Criticals in this project came from exactly that.
- House style: `from __future__ import annotations`, docstrings explaining WHY. `observability/turn_log.py` is the reference.
- **The only permitted test command**, empty values and never unset — `agent/core.py` calls `load_dotenv()` at import, which repopulates a *missing* variable and would fire real paid API calls:

      ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest -q

- Baseline: **325 passed, 3 skipped.**

## File structure map

| File | Responsibility |
|---|---|
| `transport/pcm_serializer.py` (create) | `PCMFrameSerializer`: raw Int16 PCM bytes ↔ Pipecat audio frames. One job, so the wire format lives somewhere readable. |
| `transport/browser.py` (create) | FastAPI app: serves `static/`, hosts `WS /browser-stream`, owns per-call session lifecycle. Mirrors `transport/telephony.py`'s shape including its own `__main__`. |
| `static/index.html` (create) | The page: one circle, one button. |
| `static/app.js` (create) | Mic capture, playback, button state machine, animation. |
| `static/capture-worklet.js` (create) | AudioWorklet processor: Float32 → Int16, posted to the main thread. |
| `static/style.css` (create) | The visual design. |
| `tests/test_pcm_serializer.py` (create) | Task 1. |
| `tests/test_browser.py` (create) | Tasks 2, 3, 6. |
| `.env.example`, `README.md`, `PROGRESS.md` (modify) | Task 6. |

---

### Task 1: The PCM frame serializer

**Files:**
- Create: `transport/pcm_serializer.py`
- Test: `tests/test_pcm_serializer.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `PCMFrameSerializer` (subclass of `pipecat.serializers.base_serializer.FrameSerializer`), with `PCMFrameSerializer.InputParams(sample_rate: int | None = None)`.

**Why this exists rather than passing `serializer=None`:** `FastAPIWebsocketTransport` drops frames entirely when no serializer is set (`fastapi.py:372` and `:563` both guard on it). A serializer is required, not optional.

**Why it is so much smaller than `TwilioFrameSerializer`:** Twilio's has to base64-encode, wrap in a JSON envelope with a stream SID, and resample between 8 kHz μ-law and the pipeline rate. A browser can send exactly the format the pipeline wants, so this one moves bytes.

- [ ] **Step 1: Write the failing tests**

```python
"""Phase 10f: the raw-PCM wire format between a browser and the pipeline.

Deliberately tiny. Everything Twilio's serializer does — base64, JSON
envelopes, mu-law, resampling — exists because a telephony provider dictates
the format. A browser can send precisely what the pipeline already wants, so
the correct implementation is almost a passthrough, and the tests exist to
keep it that way.
"""

from __future__ import annotations

import pytest
from pipecat.frames.frames import InputAudioRawFrame, StartFrame, TTSAudioRawFrame, TextFrame

from transport.pcm_serializer import PCMFrameSerializer

SAMPLE_RATE = 16000


async def _ready() -> PCMFrameSerializer:
    """A serializer that has seen its StartFrame, as the transport always
    sends before any audio."""
    serializer = PCMFrameSerializer()
    await serializer.setup(
        StartFrame(audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE)
    )
    return serializer


@pytest.mark.asyncio
async def test_incoming_bytes_become_an_input_audio_frame():
    serializer = await _ready()
    payload = b"\x01\x00\x02\x00\x03\x00"  # three Int16 samples

    frame = await serializer.deserialize(payload)

    assert isinstance(frame, InputAudioRawFrame)
    assert frame.audio == payload
    assert frame.sample_rate == SAMPLE_RATE
    assert frame.num_channels == 1


@pytest.mark.asyncio
async def test_outgoing_audio_serialises_back_to_identical_bytes():
    """Round-trip fidelity: what the pipeline speaks is what the browser
    plays, byte for byte. Any transformation here would be a bug, because
    both ends already agree on the format."""
    serializer = await _ready()
    payload = b"\x10\x00\x20\x00\x30\x00"

    out = await serializer.serialize(
        TTSAudioRawFrame(audio=payload, sample_rate=SAMPLE_RATE, num_channels=1)
    )

    assert out == payload


@pytest.mark.asyncio
async def test_a_non_audio_frame_serialises_to_nothing():
    """The pipeline pushes plenty of non-audio frames. Returning None tells
    the transport to send nothing, rather than putting control data on a
    socket the browser will try to play as sound."""
    serializer = await _ready()
    assert await serializer.serialize(TextFrame(text="hello")) is None


@pytest.mark.asyncio
async def test_an_empty_payload_is_ignored_rather_than_raising():
    """A browser can send a zero-length frame on a slow first buffer. That
    must not kill a live call."""
    serializer = await _ready()
    assert await serializer.deserialize(b"") is None


@pytest.mark.asyncio
async def test_an_odd_length_payload_is_ignored_rather_than_raising():
    """Int16 samples are two bytes. A truncated frame means a dropped packet
    or a browser bug; dropping it is right, crashing the call is not."""
    serializer = await _ready()
    assert await serializer.deserialize(b"\x01\x00\x02") is None


@pytest.mark.asyncio
async def test_text_payloads_are_ignored():
    """The transport hands str for text frames. Nothing in this protocol
    sends text, so anything that arrives as text is not ours."""
    serializer = await _ready()
    assert await serializer.deserialize("not audio") is None
```

- [ ] **Step 2: Run them and watch them fail**

Run: `ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest tests/test_pcm_serializer.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'transport.pcm_serializer'`

- [ ] **Step 3: Implement**

```python
"""Raw PCM over a WebSocket — Phase 10f.

The wire format between a browser and the Pipecat pipeline: 16 kHz, mono,
Int16 little-endian, in both directions, with no envelope at all.

Why this is so much smaller than TwilioFrameSerializer: everything that one
does — base64, a JSON envelope carrying a stream SID, mu-law codec work,
resampling between 8 kHz and the pipeline rate — exists because a telephony
provider dictates the format and the pipeline must adapt. A browser has no
such opinion. It opens its AudioContext at exactly the pipeline's rate and
sends exactly the bytes the pipeline wants, so the honest implementation is
a passthrough with guards.

The guards are the interesting part. A browser can deliver a zero-length
first buffer, or a frame truncated by a dropped packet, and neither may be
allowed to raise: the caller is mid-conversation and a serializer exception
would tear down a live call over one bad packet.
"""

from __future__ import annotations

from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer

BYTES_PER_SAMPLE = 2  # Int16
NUM_CHANNELS = 1


class PCMFrameSerializer(FrameSerializer):
    """Passthrough serializer for raw Int16 PCM."""

    class InputParams(FrameSerializer.InputParams):
        """Optional override for the pipeline's input sample rate. Left None
        in normal use, where the StartFrame supplies it."""

        sample_rate: int | None = None

    def __init__(self, params: InputParams | None = None, **kwargs):
        super().__init__(params=params or PCMFrameSerializer.InputParams(), **kwargs)
        self._sample_rate = 0

    async def setup(self, frame: StartFrame) -> None:
        """Called once by the transport before any audio flows."""
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Audio out to the browser; everything else is not ours to send.

        Returning None for non-audio frames matters: the transport would
        otherwise put control data on a socket the browser feeds straight
        into an audio buffer.
        """
        if isinstance(frame, AudioRawFrame):
            return frame.audio
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Browser microphone audio in.

        Every rejection path returns None rather than raising. A serializer
        exception during a live call tears the call down, and none of these
        conditions is worth a customer's conversation.
        """
        if not isinstance(data, bytes):
            return None
        if not data or len(data) % BYTES_PER_SAMPLE:
            return None
        return InputAudioRawFrame(
            audio=data,
            sample_rate=self._sample_rate,
            num_channels=NUM_CHANNELS,
        )
```

- [ ] **Step 4: Run and watch them pass**

Run: `ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest tests/test_pcm_serializer.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add transport/pcm_serializer.py tests/test_pcm_serializer.py
git commit -m "Phase 10f Task 1: raw PCM frame serializer for browser audio"
```

---

### Task 2: The WebSocket endpoint and session lifecycle

**Files:**
- Create: `transport/browser.py`
- Test: `tests/test_browser.py`

**Interfaces:**
- Consumes: `PCMFrameSerializer` (Task 1); `build_pipeline(transport, session)` from `transport/pipecat_processors.py`; `create_session`, `close_session`, `DEFAULT_CUSTOMER_ID` from `agent/session.py`.
- Produces: `app` (FastAPI), `BROWSER_PORT`, `SAMPLE_RATE`, `run_call(websocket)`.

**Read `transport/telephony.py` first.** This file mirrors its shape deliberately: accept the socket, build a transport, build the pipeline, run it, close the session on the way out. The differences are that there is no Twilio handshake to drain, no signature to validate, and no `PUBLIC_HOSTNAME`.

**Structure the connection handler so the pipeline run is separable from the socket accept**, because a test can drive `run_call` with a fake transport far more cheaply than it can drive a real WebSocket through Pipecat.

- [ ] **Step 1: Write the failing tests**

```python
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

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from transport import browser


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
```

- [ ] **Step 2: Run them and watch them fail**

Run: `ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest tests/test_browser.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'transport.browser'`

- [ ] **Step 3: Implement**

```python
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
```

Create `static/` with placeholder `index.html`, `app.js` and `style.css` so the mount and the asset test resolve; Tasks 4 and 5 fill them in.

- [ ] **Step 4: Run and watch them pass**

Run: `ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest tests/test_browser.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add transport/browser.py tests/test_browser.py static/
git commit -m "Phase 10f Task 2: browser WebSocket endpoint and per-call session lifecycle"
```

---

### Task 3: Fresh session per call

**Files:**
- Modify: `tests/test_browser.py` (append)

**Interfaces:**
- Consumes: `browser.run_call` (Task 2).
- Produces: nothing importable.

**Why this gets its own task and its own reviewer.** "Press again and ask about a different order" is a stated requirement, and the failure mode is silent: a leaked session would still answer, just with the previous caller's history and a confirmation gate already primed. That is exactly the kind of defect that looks fine in a demo until it does not, and Phase 10d shipped a session-lifecycle bug that every task review missed.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run it**

Run: `ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest tests/test_browser.py -q -k share_nothing`
Expected: PASS if Task 2 called `create_session` per call as specified. **If it FAILS, that is a real defect in Task 2 — fix `run_call`, not the test.**

- [ ] **Step 3: Commit**

```bash
git add tests/test_browser.py
git commit -m "Phase 10f Task 3: prove each call starts a genuinely fresh session"
```

---

### Task 4: The page — capture, playback, and the button

**Files:**
- Create/replace: `static/index.html`, `static/app.js`, `static/capture-worklet.js`
- Test: manual (browser); no automated coverage — see below

**Interfaces:**
- Consumes: `WS /browser-stream` (Task 2), `SAMPLE_RATE = 16000`.
- Produces: a `data-state` attribute on the button element (`idle`, `connecting`, `in-call`, `ending`) and a global `window.audioLevel` object `{mic: number, agent: number}` in the range 0–1, which Task 5's animation reads.

**No automated tests, stated deliberately.** This is browser audio; the honest coverage is Task 6's manual checkpoint. Do not add a headless-browser dependency to fake it — an assertion that a fake microphone produced a fake buffer proves nothing about whether a real conversation sounds right.

- [ ] **Step 1: Write the capture worklet**

`static/capture-worklet.js`:

```js
// Runs on the audio thread. Converts Float32 samples to the Int16 PCM the
// pipeline expects and posts them to the main thread, which owns the socket.
// An AudioWorklet rather than the deprecated ScriptProcessorNode: capture
// must not compete with the animation for the main thread, or the ripples
// stutter exactly when someone is speaking.
//
// Audio is BATCHED to ~20 ms before being posted. An AudioWorklet's render
// quantum is 128 frames, which at 16 kHz is 8 ms — posting every quantum would
// send 125 WebSocket messages a second, each carrying 256 bytes of audio
// under a full frame's worth of overhead. Batching to 320 samples cuts that
// to 50 messages a second and matches the chunk size telephony transports
// use. Sending unbatched is a plausible cause of choppy audio on its own,
// so this is built in rather than discovered later.
const BATCH_SAMPLES = 320; // 20 ms at 16 kHz

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Int16Array(BATCH_SAMPLES);
    this._filled = 0;
    this._energy = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const samples = input[0];

    for (let i = 0; i < samples.length; i++) {
      const clamped = Math.max(-1, Math.min(1, samples[i]));
      this._buffer[this._filled++] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
      this._energy += clamped * clamped;

      if (this._filled === BATCH_SAMPLES) {
        // RMS travels with the audio so the animation never re-measures it.
        const level = Math.sqrt(this._energy / BATCH_SAMPLES);
        const batch = this._buffer.slice();
        this.port.postMessage({ pcm: batch, level }, [batch.buffer]);
        this._filled = 0;
        this._energy = 0;
      }
    }
    return true;
  }
}
registerProcessor("capture-processor", CaptureProcessor);
```

- [ ] **Step 2: Write the page**

`static/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Support Voice Agent</title>
    <link rel="stylesheet" href="/static/style.css" />
  </head>
  <body>
    <main>
      <canvas id="ripples"></canvas>
      <button id="call" data-state="idle" aria-label="Start call">
        <span id="label">Call</span>
      </button>
      <p id="status" role="status"></p>
    </main>
    <script src="/static/app.js" type="module"></script>
  </body>
</html>
```

- [ ] **Step 3: Write the client**

`static/app.js`:

```js
// One button, one call. Press to talk to the agent, press again to hang up,
// press again for a conversation that remembers nothing of the last one —
// the server creates a new session per socket, so a fresh press is a fresh
// caller.
const SAMPLE_RATE = 16000;

// How far ahead of the audio clock each chunk is scheduled. See playChunk.
const PLAYBACK_LEAD_SECONDS = 0.1;

const button = document.getElementById("call");
const label = document.getElementById("label");
const status = document.getElementById("status");

// Read by the animation (style.css / Task 5). Kept on window rather than
// passed around because the animation loop runs independently of call state.
window.audioLevel = { mic: 0, agent: 0 };

let ctx = null;
let socket = null;
let stream = null;
let playHead = 0;

function setState(state, message = "") {
  button.dataset.state = state;
  label.textContent = state === "in-call" ? "Hang up" : "Call";
  status.textContent = message;
}

function playChunk(bytes) {
  // Int16 back to Float32, scheduled end-to-end so consecutive chunks play
  // as continuous speech rather than overlapping or gapping.
  const pcm = new Int16Array(bytes);
  const buffer = ctx.createBuffer(1, pcm.length, SAMPLE_RATE);
  const channel = buffer.getChannelData(0);
  let sum = 0;
  for (let i = 0; i < pcm.length; i++) {
    channel[i] = pcm[i] / 0x8000;
    sum += channel[i] * channel[i];
  }
  window.audioLevel.agent = Math.sqrt(sum / pcm.length);

  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  // Schedule a jitter buffer ahead of the clock rather than at it. Starting
  // a chunk at exactly ctx.currentTime means any scheduling jitter lands it
  // late, and a late chunk is an audible gap mid-word. A ~100 ms lead costs
  // a tenth of a second of latency nobody notices and removes the most
  // likely source of stutter.
  playHead = Math.max(playHead, ctx.currentTime + PLAYBACK_LEAD_SECONDS);
  source.start(playHead);
  playHead += buffer.duration;
}

async function startCall() {
  setState("connecting", "Connecting…");
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      // Browser-side echo cancellation, so the agent's own voice coming out
      // of the speakers is not fed back in as the caller interrupting — the
      // self-echo problem Phase 8 hit with a local microphone.
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
    });
  } catch (err) {
    setState("idle", "Microphone permission denied.");
    return;
  }

  // Opening the context at the pipeline's rate means nothing resamples,
  // anywhere.
  ctx = new AudioContext({ sampleRate: SAMPLE_RATE });

  // sampleRate is a HINT, not a guarantee. A browser that ignores it and
  // hands back 48 kHz breaks everything downstream by a factor of three:
  // Deepgram receives what sounds like nonsense, never fires end-of-turn,
  // and the agent simply never replies. That symptom points nowhere near
  // its cause, so fail loudly here instead of debugging silence later.
  if (ctx.sampleRate !== SAMPLE_RATE) {
    setState("idle", `Browser gave ${ctx.sampleRate} Hz, not ${SAMPLE_RATE} Hz. Try Chrome.`);
    await ctx.close();
    stream.getTracks().forEach((track) => track.stop());
    ctx = null;
    stream = null;
    return;
  }

  await ctx.audioWorklet.addModule("/static/capture-worklet.js");

  socket = new WebSocket(`ws://${location.host}/browser-stream`);
  socket.binaryType = "arraybuffer";

  socket.onmessage = (event) => playChunk(event.data);
  socket.onclose = () => endCall();
  socket.onerror = () => setState("idle", "Connection failed.");

  socket.onopen = () => {
    const source = ctx.createMediaStreamSource(stream);
    const capture = new AudioWorkletNode(ctx, "capture-processor");
    capture.port.onmessage = ({ data }) => {
      window.audioLevel.mic = data.level;
      if (socket && socket.readyState === WebSocket.OPEN) socket.send(data.pcm.buffer);
    };
    source.connect(capture);
    playHead = ctx.currentTime;
    setState("in-call", "");
  };
}

function endCall() {
  if (button.dataset.state === "idle") return;
  setState("ending", "");
  if (socket && socket.readyState === WebSocket.OPEN) socket.close();
  if (stream) stream.getTracks().forEach((track) => track.stop());
  if (ctx) ctx.close();
  socket = null;
  stream = null;
  ctx = null;
  window.audioLevel = { mic: 0, agent: 0 };
  setState("idle", "");
}

button.addEventListener("click", () => {
  if (button.dataset.state === "idle") startCall();
  else endCall();
});
```

- [ ] **Step 4: Confirm the suite still passes**

Run: `ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest -q`
Expected: no regressions; Task 2's asset test now serves real files.

- [ ] **Step 5: Commit**

```bash
git add static/
git commit -m "Phase 10f Task 4: browser capture, playback and the call button"
```

---

### Task 5: The visual design

**Files:**
- Create/replace: `static/style.css`
- Modify: `static/app.js` (append the animation loop only)

**Interfaces:**
- Consumes: `button[data-state]` and `window.audioLevel` (Task 4), `<canvas id="ripples">`.
- Produces: nothing importable.

**Run the `design-consultation` skill for this task** rather than inventing a palette. The states are now settled, which is what it needs: `idle`, `connecting`, `in-call`, `ending`.

**The animation requirement, precisely, because it is easy to get subtly wrong:**

Ripples are concentric rings spawning at the circle's edge and expanding outward, fading as they travel. Spawn rate and travel distance scale with the current audio level.

**Each ring takes its colour at spawn time and keeps it for its whole life.** It does not re-tint as it expands. This is the difference between an effect that works and one that looks generic: because every ring carries the loudness it was born with, the expanding field becomes a visible history of the last few seconds of speech — a loud syllable sends a bright ring travelling outward while quieter rings follow behind it. Re-tinting every ring each frame collapses that into one flat pulsing colour and throws the history away.

Colour is interpolated along a continuous scale, so a rising voice sweeps rather than steps. The caller and the agent occupy distinguishable ends of that scale, so a viewer watching the recording can tell who is speaking without sound.

- [ ] **Step 1: Invoke the design skill**

Run the `design-consultation` skill with the four button states, the ripple behaviour above, and the context that this is recorded for a job-interview demo video, likely shown split-screen beside a Slack notification. Take its palette, typography and motion easing.

- [ ] **Step 2: Write the animation loop**

Append to `static/app.js` a `requestAnimationFrame` loop that maintains an array of rings. Each frame: spawn a ring when the elapsed time since the last spawn exceeds an interval derived from `window.audioLevel`, assigning it a colour sampled from the scale at that instant and recording whether `mic` or `agent` was louder; advance every ring's radius; drop rings past the canvas bounds; and redraw. Never recompute an existing ring's colour.

- [ ] **Step 3: Write the stylesheet**

`static/style.css` per the design consultation's system: the circle, its per-state appearance, the canvas positioning behind it, typography, and the page background. Honour `prefers-reduced-motion` by holding the rings still rather than removing the button.

- [ ] **Step 4: Confirm the suite still passes**

Run: `ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest -q`
Expected: no regressions.

- [ ] **Step 5: Commit**

```bash
git add static/
git commit -m "Phase 10f Task 5: circle, audio-reactive ripples and the visual system"
```

---

### Task 6: Documentation and the honest end state

**Files:**
- Modify: `.env.example`, `README.md`, `PROGRESS.md`
- Test: none new — verification is the full suite

- [ ] **Step 1: Add the environment note**

In `.env.example`, document `BROWSER_PORT` (default `8080`) and state that the browser console needs **no Twilio credentials and no PUBLIC_HOSTNAME** — only `DEEPGRAM_API_KEY` and `ANTHROPIC_API_KEY`.

- [ ] **Step 2: Write the README section**

Cover: how to run it (`python -m transport.browser`, open `localhost:8080`); that it is the fourth transport behind an unchanged `build_pipeline` and nothing in `agent/` changed; why WebSocket rather than WebRTC (`aiortc` absent, and `FastAPIWebsocketTransport` already carries the Twilio traffic); that every press is a fresh session; that escalation still fires the n8n webhook so a split-screen Slack recording works; and **that audio quality is unverified until someone speaks into it**, with sample-rate conversion, buffering and playback smoothness named as the specific unknowns.

- [ ] **Step 3: Update PROGRESS.md**

Add the 10f row matching the 10a–10d format and tone. State the automated result and that the manual checkpoint has or has not been run — do not claim it if it has not. This project has corrected two phases for exactly that.

- [ ] **Step 4: Run the full suite**

Run: `ANTHROPIC_API_KEY= DEEPGRAM_API_KEY= .venv/bin/python -m pytest -q`
Expected: 325 baseline plus roughly 10 new tests, 3 skipped, no regressions. Report the actual numbers.

- [ ] **Step 5: Commit**

```bash
git add .env.example README.md PROGRESS.md
git commit -m "Phase 10f Task 6: document the browser console and its unverified audio"
```

---

## Self-review

**Spec coverage.** PCM serializer → Task 1. WebSocket endpoint and session lifecycle → Task 2. Fresh session per call → Task 3. Button state machine, capture, playback, permission handling → Task 4. Animation and colour-follows-ripple → Task 5. Docs and honest end state → Task 6. Error handling: microphone denial (Task 4), unexpected socket close (Tasks 2 and 4), mid-turn hang-up (Task 2), malformed audio payloads (Task 1). No gaps.

**Placeholders.** None: every code step carries real code; the one step that legitimately defers content (Task 5 Step 1) defers it to a named skill, not to the implementer's imagination.

**Type consistency.** `PCMFrameSerializer` and `SAMPLE_RATE = 16000` are spelled identically in Tasks 1, 2 and 4. `run_call(websocket) -> Session` matches its Task 3 call sites. `data-state` values (`idle`, `connecting`, `in-call`, `ending`) match between Tasks 4 and 5. `window.audioLevel` is `{mic, agent}` in both.

**Two risks I want the executor to see rather than discover.**

The `_run_pipeline` seam exists so Task 2's lifecycle can be tested without a real WebSocket. If Pipecat's runner API differs from what is written here, fix the implementation and keep the seam — the tests depend on the boundary, not on the runner's exact call shape.

Self-review already caught one instance of exactly that: this plan originally used `PipelineTask`/`PipelineRunner`, which Pipecat 1.7.0 does export, but `transport/telephony.py` uses `PipelineWorker` + `WorkerRunner` (`pipecat.pipeline.worker` and `pipecat.workers.runner`). Both work; the project having two conventions for the same thing would not. Corrected to match telephony.

Audio is genuinely unverified. Every test in this plan passes with silence. Sample-rate agreement, buffer sizing, and whether playback sounds continuous rather than choppy are settled only by Task 6's manual checkpoint, and one round of tuning afterwards should be expected rather than treated as failure.

**Three mitigations are built in rather than left to that tuning round**, because each is cheap to do correctly the first time and expensive to diagnose from its symptom:

1. **20 ms capture batching** (Task 4, worklet). Unbatched, an AudioWorklet posts every 128-frame quantum — 125 WebSocket messages a second at 16 kHz, each 256 bytes of audio wrapped in a full frame's overhead. A plausible cause of choppiness by itself.
2. **A sample-rate assertion** (Task 4, `startCall`). `AudioContext({sampleRate})` is a hint. A browser that returns 48 kHz instead breaks everything downstream by a factor of three, and the visible symptom is that the agent never replies — which points nowhere near the cause.
3. **A 100 ms playback lead** (Task 4, `playChunk`). Scheduling at exactly `currentTime` means any jitter lands a chunk late, and a late chunk is an audible gap mid-word.

Three further mitigations were considered and deliberately left out, to be added only if the symptoms actually appear: resetting `playHead` when drift exceeds 500 ms, fades at chunk edges to kill clicks, and server-side RMS logging to distinguish "capture is broken" from "the model is quiet". Building all six up front would ship untested defensive code against problems that may never occur.
