# Phase 10f — Browser voice console

Date: 2026-09-10
Status: approved, not yet implemented
Depends on: Phase 8 (Pipecat pipeline). Independent of 10d/10e.

## Why this phase exists

Every voice demo this project can currently give requires something the owner does not have.
Phase 9 and 10d need a Twilio account, and Twilio does not sell numbers in Serbia; a trial
account cannot be created there either. Phase 7's `transport/voice_local.py` works with only a
microphone and a Deepgram key, but it is a terminal program — nothing to look at, nothing that
records well.

This phase gives the project a demo it can actually run: one browser page, one button, a real
spoken conversation with the agent, and an animation that makes the exchange legible on screen.

It is also the fourth transport to sit behind an unchanged `build_pipeline`. Phases 7, 8 and 9
swapped text CLI → local microphone → Pipecat → Twilio without touching `agent/`. Doing it a
fourth time, for a completely different medium, is the strongest available evidence that the
decoupling in CLAUDE.md rule 5 is real rather than aspirational.

## Decisions taken by the project owner

Fixed inputs, not open questions:

1. **One page, one button.** No separate agent console. Press to talk to the agent, press again
   to hang up, press again to start a fresh conversation.
2. **Hanging up behaves like a real phone call** — the session ends, the post-call summary is
   written, and the next press starts something genuinely new.
3. **The animation is the point.** A single large circle with ripples expanding outward in
   circular motion, reacting to live audio, with colour that shifts smoothly and travels
   outward with each ripple.
4. **The agent console, the whisper and two-way bridging are out of scope**, along with the
   n8n relinking, which the owner will do separately.

## The constraint that chose the architecture

`SmallWebRTCTransport` is Pipecat's browser-direct WebRTC transport and would have been the
obvious choice, but it requires `aiortc`, which is not installed and brings native
dependencies with it. Verified: importing it fails with `No module named 'aiortc'` and
Pipecat's own message directing you to `pipecat-ai[webrtc]`.

The alternative needs **no new dependency at all**. `FastAPIWebsocketTransport` is already
installed and already carries this project's production telephony traffic. Twilio merely wraps
it in a `TwilioFrameSerializer`. Passing the same transport with a small raw-PCM serializer
gives a browser the same pipeline.

That is also the lower-risk option in a second way: it reuses a transport this project has
already run against real calls, rather than introducing an unfamiliar one.

**A welcome consequence: no ngrok, no `PUBLIC_HOSTNAME`, no tunnel.** Twilio needed to reach
back into the machine, so Phase 9 required a public hostname. A browser connects outward to
`localhost`, so this demo runs entirely offline on one laptop.

## Architecture

```
browser microphone
  -> getUserMedia (16 kHz, echoCancellation on)
  -> AudioWorklet: Float32 -> Int16 PCM
  -> binary WebSocket frames
      -> WS /browser-stream
      -> PCMFrameSerializer -> InputAudioRawFrame
      -> build_pipeline(transport, session)   [UNCHANGED from Phase 8]
           DeepgramFluxSTTService -> ClaudeTurnProcessor -> TTS -> LatencyLogger
      -> TTSAudioRawFrame -> PCMFrameSerializer -> binary WebSocket frames
  -> Web Audio playback
```

Nothing in `agent/` changes. Nothing in `transport/pipecat_processors.py` changes — it already
accepts any `BaseTransport`, which is the whole point.

## Components

| File | Responsibility |
|---|---|
| `transport/browser.py` (create) | FastAPI app: serves the page, hosts `WS /browser-stream`, owns the per-call session lifecycle. Mirrors `transport/telephony.py`'s shape, including its own `__main__`. |
| `transport/pcm_serializer.py` (create) | `PCMFrameSerializer`: raw Int16 PCM bytes ↔ Pipecat audio frames. Small and single-purpose, so the audio format lives in one readable place rather than inside the transport. |
| `static/index.html` (create) | The page. One circle, one button, nothing else. |
| `static/app.js` (create) | Microphone capture, playback, the button state machine, the animation. |
| `static/style.css` (create) | The visual design. |

`transport/browser.py` uses its own port (`BROWSER_PORT`, default `8080`) rather than
telephony's `PORT` (default `8765`), so both can run at once without a collision.

## The button state machine

```
idle  --press-->  connecting  --socket open-->  in call
in call  --press-->  ending  --socket closed-->  idle
```

- **idle** — the circle sits still, waiting.
- **connecting** — the WebSocket is opening and the microphone permission may be pending.
- **in call** — audio flows both ways; the greeting plays immediately, because
  `ClaudeTurnProcessor` already speaks `GREETING` on `StartFrame` and needs no model call to
  do it.
- **ending** — the browser closes the socket. The server sees the disconnect, the pipeline
  stops, and `close_session` writes the post-call summary ticket.

**Every press of the button in `idle` calls `create_session()`.** A second call shares nothing
with the first: no conversation history, no `PendingActionGate` state, a new `session_id`. This
is deliberate and is the opposite of Phase 10d's session resume, which existed solely so a
customer could survive a failed transfer. Here a new call is a new customer.

## The animation

One large circle, centred. Ripples are concentric rings that spawn at the circle's edge and
expand outward, fading as they go.

**Audio-reactive.** The page measures RMS level from two sources: the local microphone
(you speaking) and the incoming audio stream (the agent speaking). The louder the current
source, the faster ripples spawn and the further they travel.

**Colour follows the ripple.** Each ring is assigned its colour **at spawn time**, from the
audio level in that instant, and keeps that colour for its whole life as it expands. It does
not re-tint as it travels.

This is the detail that makes the effect work rather than look generic. Because each ring
carries the loudness it was born with, the expanding field becomes a visible history of the
last few seconds of speech — a loud syllable sends a bright ring travelling outward while
quieter rings follow behind it. Re-tinting every ring on each frame would produce a field that
pulses as one flat colour and throws that history away.

**Smooth transitions.** Colour is interpolated along a continuous scale rather than snapped
between fixed values, so a rising voice sweeps through the scale instead of stepping.

**Who is speaking is visible.** The caller and the agent occupy distinguishable ends of the
colour scale, so a viewer watching the recording can tell them apart without hearing it.

The specific palette, easing, ripple rate and circle proportions are for the design
consultation, which runs against these states once the spec is approved.

## Escalation still works, and is worth filming

Nothing here touches escalation. `create_handoff_packet` lives in `agent/tools/` and is
transport-agnostic, so asking for a human during a browser call still classifies the turn,
writes the `escalations` row, and POSTs the redacted handoff packet to `ESCALATION_WEBHOOK_URL`
(Phase 11's n8n webhook) — verified at `agent/tools/escalation.py:299`.

So a split-screen recording showing the browser page beside a Slack message works with no
additional code. The owner will relink n8n separately.

What does *not* happen is a transfer: there is no agent console in this phase, so the agent
speaks its handoff notice and the call ends.

## Error handling

- **Microphone permission denied** — the button returns to `idle` with a visible message. No
  silent failure; a dead-looking button is the worst outcome for a demo.
- **WebSocket closes unexpectedly mid-call** — the page returns to `idle` and the server closes
  the session as it would for a normal hang-up, so no session leaks.
- **The browser hangs up mid-turn**, while the model is generating — the pipeline task is
  cancelled and `close_session` still runs. Phase 8 already proved cancellation mid-turn is
  safe; this must not regress it.
- **`DEEPGRAM_API_KEY` missing** — the page loads and the button reports the failure rather
  than opening a socket that will die on the first frame.
- **A second browser tab connects** — each connection gets its own session and its own
  pipeline. They do not interfere. Not a supported demo mode, but it must not corrupt state.

## Testing

Everything offline; no browser automation, no API keys.

- `PCMFrameSerializer` round-trips: bytes in produce an `InputAudioRawFrame` with the right
  sample rate and channel count; a `TTSAudioRawFrame` serialises back to the identical bytes.
- Odd-length and empty payloads are handled rather than raising — a truncated frame from a
  browser must not kill the call.
- Connecting to `WS /browser-stream` creates exactly one session; disconnecting closes exactly
  one and writes one summary.
- **Two sequential connections produce two different `session_id`s and share no conversation
  history** — the "call again with a different order" requirement, asserted rather than assumed.
- A disconnect mid-turn still closes the session.
- The page is served and references the script and stylesheet that actually exist.

Values come from `data/mock_db.py` at runtime, never hard-coded.

**What testing cannot cover, stated plainly:** audio quality. Sample-rate conversion, buffer
sizes, playback smoothness and whether the ripples feel connected to the voice are only
answerable by speaking into it. One round of hands-on tuning after the suite is green should be
expected, not treated as a surprise.

## Checkpoint

**Automated:** all new tests pass offline with no API keys; no regressions in the existing
suite.

**Manual (needs only a Deepgram key — no Twilio, no tunnel):** run
`python -m transport.browser`, open `localhost:8080`, press the circle. Confirm the greeting
plays, a spoken order-status question is answered, the ripples track your voice and the agent's
distinguishably, pressing again ends the call and writes a summary ticket, and pressing once
more starts a conversation that remembers nothing of the first.

## Out of scope

- The agent console, the whisper, and two-way bridging between two humans.
- WebRTC. Revisit only if WebSocket audio quality proves inadequate in practice.
- Relinking n8n — the owner's, separately.
- Authentication. This binds to localhost for a local demo; it is not exposed.
- Mobile layout. It is a desktop screen recording.
