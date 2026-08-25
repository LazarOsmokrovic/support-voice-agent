"""Real-time streaming voice loop — Phase 8 interface (Pipecat).

Rebuilds transport/voice_local.py's sequential loop as a Pipecat pipeline of
frame processors, adding real barge-in (stop the bot the instant the caller
starts talking) and partial transcripts. This file is the *only* place
Pipecat and this project's brain touch — agent/core.py and agent/session.py
are completely unchanged, exactly what CLAUDE.md rule 5 requires. The whole
integration point is one custom FrameProcessor:

    transport.input() -> DeepgramFluxSTTService -> ClaudeTurnProcessor
        -> TTS service -> LatencyLogger -> transport.output()

Verified against the actually-installed pipecat-ai (1.7.0) source before
writing this, not against its web docs — those describe an older API
(PipelineTask/PipelineRunner, a separate StartInterruptionFrame) that this
version has already deprecated in favor of PipelineWorker/WorkerRunner and a
single consolidated InterruptionFrame. Same "read real source, don't guess"
principle Phase 7 applied to the Deepgram SDK.

Why no separate VAD: DeepgramFluxSTTService broadcasts UserStartedSpeaking/
UserStoppedSpeakingFrame directly from Flux's own turn detection, and (via
its should_interrupt=True default) triggers the pipeline's interruption
itself — continuing Phase 7's reasoning that Flux's built-in end-of-turn
detection is exactly what avoids hand-rolling VAD.

Why TTS is a Pipecat service, not transport/tts.py: real barge-in needs the
framework to cancel in-flight synthesis and drop queued audio the instant an
InterruptionFrame arrives. Pipecat's WebSocket TTS services participate in
that; transport/tts.py's one-shot REST call (correct for Phase 7's strictly
sequential loop) doesn't. transport/tts.py and transport/voice_local.py are
untouched — this is an additive, new transport.

Run with: python -m transport.pipeline
"""

from __future__ import annotations

import asyncio
import os
import time

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)
from pipecat.workers.runner import WorkerRunner

from agent.core import configure_logging
from agent.session import (
    DEFAULT_CUSTOMER_ID,
    Session,
    close_session,
    create_session,
    run_turn,
)
from transport.tts import (
    DEFAULT_CARTESIA_MODEL,
    DEFAULT_CARTESIA_VOICE,
    DEFAULT_DEEPGRAM_VOICE,
)


class ClaudeTurnProcessor(FrameProcessor):
    """The one integration point between Pipecat and this project's brain.

    On a final TranscriptionFrame, runs the turn through agent/session.py's
    run_turn() — the exact same function transport/text_cli.py and
    transport/voice_local.py already use — and pushes the reply as a plain
    TextFrame for the TTS service downstream. Brackets it with
    LLMFullResponseStart/EndFrame, matching what a real Pipecat LLM service
    would emit (TTSService explicitly keys its per-turn audio-context
    tracking off these two frames).

    Barge-in needs no special handling *here*: FrameProcessor's base
    process_frame() already cancels this processor's own in-flight
    process_frame() call when an InterruptionFrame arrives (it cancels and
    recreates the per-processor task actually running this coroutine) — so a
    mid-turn interruption simply aborts the `await run_turn(...)` call
    already in flight. Nothing past that point ever executes, so no stale
    reply gets pushed after the interruption.

    The raw TranscriptionFrame itself is deliberately not forwarded
    downstream — it's fully consumed here, not "passed along," since
    downstream (TTS) has no use for the caller's own words.
    """

    def __init__(self, *, session: Session, **kwargs):
        super().__init__(**kwargs)
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            if frame.text.strip():
                await self._handle_final_transcript(frame.text)
            return

        await self.push_frame(frame, direction)

    async def _handle_final_transcript(self, text: str) -> None:
        outcome = await run_turn(self._session, text)
        print(f"[latency] LLM turn: {outcome.llm_latency_seconds * 1000:.0f}ms")
        print(f"[reply] {len(outcome.reply)} chars: {outcome.reply!r}")
        for warning in outcome.warnings:
            print(f"({warning})")

        await self.push_frame(LLMFullResponseStartFrame())
        if outcome.reply.strip():
            await self.push_frame(TextFrame(text=outcome.reply))
        else:
            # Nothing to synthesize — pushing an empty TextFrame would ask
            # DeepgramTTSService to open a TTS context that produces zero
            # audio, which is exactly what its own 3s pause-watchdog logs as
            # "no BotStartedSpeakingFrame ... force-resuming". Skipping it
            # keeps that (rare, model-side) edge case from ever reaching TTS.
            print("(model returned an empty reply this turn — nothing to speak)")
        if outcome.notice:
            print(outcome.notice)
            await self.push_frame(TextFrame(text=outcome.notice))
        await self.push_frame(LLMFullResponseEndFrame())

        if outcome.ended:
            # EndFrame is a ControlFrame (ordered, not high-priority) and
            # UninterruptibleFrame — it queues in after the reply above and
            # survives a stray interruption, and the output transport drains
            # already-queued audio before actually stopping. So this reply
            # still gets spoken in full before the call ends; no abrupt cut.
            await self.push_frame(EndFrame())


class LatencyLogger(FrameProcessor):
    """Sits right before transport.output(). Every processor forwards frames
    it doesn't act on, so both UserStoppedSpeakingFrame (end of the caller's
    turn) and the reply's first TTSAudioRawFrame propagate all the way down
    to this position — diffing their timestamps gives the actual round-trip
    latency PROJECT_PLAN.md's checkpoint asks about ("~1s"), not a synthetic
    one measured some other way.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._turn_ended_at: float | None = None
        self._logged_this_turn = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStoppedSpeakingFrame):
            self._turn_ended_at = time.perf_counter()
            self._logged_this_turn = False
        elif (
            isinstance(frame, TTSAudioRawFrame)
            and self._turn_ended_at is not None
            and not self._logged_this_turn
        ):
            round_trip = time.perf_counter() - self._turn_ended_at
            print(f"[latency] round-trip (end-of-turn -> first bot audio): {round_trip * 1000:.0f}ms")
            self._logged_this_turn = True

        await self.push_frame(frame, direction)


def get_pipecat_tts_service():
    """TTS_BACKEND env var: "deepgram" (default) or "cartesia" — same switch
    and same default voice/model constants as transport/tts.py's
    get_tts_backend(), just backed by Pipecat's own streaming TTS services
    (which participate in interruption) instead of a one-shot REST call.
    """
    backend_name = os.getenv("TTS_BACKEND", "deepgram").lower()
    if backend_name == "cartesia":
        return CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            settings=CartesiaTTSService.Settings(
                voice=os.getenv("CARTESIA_TTS_VOICE", DEFAULT_CARTESIA_VOICE),
                model=os.getenv("CARTESIA_TTS_MODEL", DEFAULT_CARTESIA_MODEL),
            ),
        )
    if backend_name != "deepgram":
        raise ValueError(f"unknown TTS_BACKEND: {backend_name!r} (expected 'deepgram' or 'cartesia')")
    return DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramTTSService.Settings(voice=os.getenv("DEEPGRAM_TTS_VOICE", DEFAULT_DEEPGRAM_VOICE)),
    )


async def main() -> None:
    configure_logging()
    customer_id = input(f"Customer ID [{DEFAULT_CUSTOMER_ID}]: ").strip() or DEFAULT_CUSTOMER_ID
    session = create_session(customer_id)

    transport = LocalAudioTransport(LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True))
    stt = DeepgramFluxSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = get_pipecat_tts_service()

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            ClaudeTurnProcessor(session=session),
            tts,
            LatencyLogger(),
            transport.output(),
        ]
    )
    worker = PipelineWorker(pipeline, params=PipelineParams(enable_metrics=True))
    runner = WorkerRunner()
    await runner.add_workers(worker)

    print(
        f"\nVoice session started for {customer_id} (real-time). Speak naturally — "
        "you can interrupt the bot at any time by talking over it. Ctrl+C to abort.\n"
    )

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
    asyncio.run(main())
