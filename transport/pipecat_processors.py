"""Pipecat-specific glue shared by transport/pipeline.py (local mic) and
transport/telephony.py (Twilio) — Phase 9.

Extracted out of transport/pipeline.py on Twilio's arrival as a second real
use case, for the same reason agent/session.py was extracted out of
transport/text_cli.py in Phase 7: a transport importing from another
transport module would be backwards, and duplicating this logic risks the
two drifting apart. It can't live in agent/ either — it imports
pipecat.frames/pipecat.processors directly, and agent/ has to stay
completely decoupled from whatever I/O layer is driving it (CLAUDE.md
rule 5). This is the framework-facing half of that boundary; agent/core.py
and agent/session.py themselves are untouched by Phase 9, same as Phase 8.

Nothing here changed behavior from Phase 8 except one addition: a new
deterministic DTMF branch in ClaudeTurnProcessor ("press 0 for a human",
PROJECT_PLAN.md's Phase 9 safety net) — an *additional* layer on top of the
model-driven escalation agent/tools/escalation.py already does automatically
via run_turn() (an explicit spoken request, repeated failures, or a
policy-restricted topic all already escalate with zero button presses, since
Phase 4). This is a fallback for when that detection doesn't fire — the
model misjudges the request or the pipeline misbehaves — not a replacement
for it. Telephony-only in practice (a local mic never produces an
InputDTMFFrame), but harmless to share since it's just one more isinstance()
branch next to the existing model-driven one.

Entirely provider-agnostic: nothing here imports Twilio (or any other
telephony provider) directly — that lives only in transport/telephony.py,
which supplies the `transport` object build_pipeline() wires in.
"""

from __future__ import annotations

import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    InputDTMFFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartFrame,
    STTMuteFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.transports.base_transport import BaseTransport

from agent.prompts import GREETING
from agent.session import Session, run_turn
from agent.tools import escalation
from observability.turn_log import TurnRecord, log_turn
from transport.tts import (
    DEFAULT_CARTESIA_MODEL,
    DEFAULT_CARTESIA_VOICE,
    DEFAULT_DEEPGRAM_VOICE,
)


EscalationHook = Callable[[dict[str, Any] | None], Awaitable[bool]]


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

    Phase 9 adds one more branch: InputDTMFFrame. Pressing 0 is
    PROJECT_PLAN.md's "safety net independent of the AI" — an *additional*
    layer on top of the model-driven escalation run_turn() already performs
    every turn (an explicit spoken request, repeated failures, or a
    policy-restricted topic already escalate automatically, since Phase 4).
    The decision to act on a 0 press is a plain isinstance() check, never
    classify_turn() or any other model judgment, matching how
    EscalationTracker's deterministic counters already sit next to
    classify_turn's model-driven one in agent/tools/escalation.py (CLAUDE.md
    rule 7). Since InputDTMFFrame is a SystemFrame it can arrive and be
    handled on a separate, higher-priority task while a
    TranscriptionFrame-triggered turn is still in flight on this same
    processor — acceptable here since the fallback is meant to preempt
    whatever the AI is doing, not queue politely behind it.
    """

    def __init__(self, *, session: Session, on_escalation: EscalationHook | None = None, **kwargs):
        super().__init__(**kwargs)
        self._session = session
        # Injected by transport/telephony.py, absent for transport/pipeline.py.
        # This is how a real call transfer happens without this shared module
        # importing Twilio — the local-mic pipeline has no phone call to
        # transfer, and CLAUDE.md rule 5 keeps provider code out of here.
        self._on_escalation = on_escalation

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            # Forward StartFrame FIRST, so the TTS service downstream is
            # initialized by the time the greeting text reaches it, then
            # speak. Without this the caller hears nothing until they talk
            # first, which on a phone line reads as a dead connection.
            await self.push_frame(frame, direction)
            await self._speak_greeting()
            return

        if isinstance(frame, InputDTMFFrame):
            if frame.button == KeypadEntry.ZERO:
                await self._handle_dtmf_escalation()
            return

        if isinstance(frame, TranscriptionFrame):
            if frame.text.strip():
                await self._handle_final_transcript(frame.text)
            return

        await self.push_frame(frame, direction)

    async def _speak_greeting(self) -> None:
        """Say hello the moment the pipeline starts, before the caller has
        said anything.

        Bracketed with LLMFullResponseStart/EndFrame for the same reason
        every reply is: the TTS service keys its per-turn audio-context
        tracking off that pair, so an unbracketed TextFrame would arrive
        outside any context. GREETING is a constant (agent/prompts.py) —
        no model call, so this adds no latency to answering the call.
        """
        print(f"[greeting] {GREETING}")
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(TextFrame(text=GREETING))
        await self.push_frame(LLMFullResponseEndFrame())

    async def _fire_escalation_hook(self, packet: dict[str, Any] | None) -> bool:
        """Call the transport's transfer hook, swallowing anything it raises.

        A telephony failure must never crash the pipeline: the customer is
        mid-call and the notice still has to reach them.
        """
        if self._on_escalation is None:
            return False
        try:
            return await self._on_escalation(packet)
        except Exception as exc:  # noqa: BLE001 — a broken transfer must not drop the call
            print(f"(escalation hook failed: {exc})")
            return False

    async def _handle_dtmf_escalation(self) -> None:
        escalation_id: int | None = None
        packet: dict[str, Any] | None = None
        try:
            packet = await escalation.create_handoff_packet(
                self._session.customer_id, self._session.agent.messages, "caller pressed 0 for a human"
            )
            escalation_id = packet["escalation_id"]
            notice = f"Connecting you with a human agent. (handoff #{escalation_id})"
        except Exception as exc:  # noqa: BLE001 — the fallback must never crash the call
            notice = "Connecting you with a human agent."
            print(f"(DTMF escalation triggered, but the handoff packet couldn't be logged: {exc})")

        # This path bypasses run_turn entirely, so it advances the session's
        # own turn counter itself (agent/session.py's Session.turn) — without
        # this, its record would carry the SAME turn number as the spoken
        # turn immediately before it (or turn=0 if 0 is pressed before anyone
        # speaks), colliding at the one event this phase exists to cover.
        self._session.turn += 1

        # Pressing 0 bypasses run_turn entirely (Phase 9's deterministic safety
        # net), so it would otherwise leave no trace in the turn log — a hole at
        # exactly the event most worth recording. Same schema as a spoken turn,
        # so keypress and spoken escalations read alike in one file.
        try:
            log_turn(
                TurnRecord(
                    session_id=self._session.session_id,
                    customer_id=self._session.customer_id,
                    transport=self._session.transport,
                    turn=self._session.turn,
                    user_text="[DTMF] 0",
                    reply=notice,
                    original_reply=None,
                    grounding_flagged=False,
                    hedge_spoken=False,
                    tool_calls=[],
                    llm_latency_seconds=0.0,
                    warnings=[],
                    escalated=True,
                    escalation_reason="caller pressed 0 for a human",
                    escalation_id=escalation_id,
                    ended=True,
                    end_reason="escalated",
                )
            )
        except Exception as exc:  # noqa: BLE001 — telemetry must never break the fallback
            print(f"(turn log write failed for the DTMF escalation: {exc})")

        await self._fire_escalation_hook(packet if escalation_id is not None else None)

        print(notice)
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(TextFrame(text=notice))
        await self.push_frame(LLMFullResponseEndFrame())
        await self.push_frame(EndFrame())

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
            if outcome.end_reason == "escalated":
                await self._fire_escalation_hook(outcome.escalation_packet)
            # EndFrame is a ControlFrame (ordered, not high-priority) and
            # UninterruptibleFrame — it queues in after the reply above and
            # survives a stray interruption, and the output transport drains
            # already-queued audio before actually stopping. So this reply
            # still gets spoken in full before the call ends; no abrupt cut.
            await self.push_frame(EndFrame())


class MicMuteGate(FrameProcessor):
    """Mutes the STT service for the duration the bot is speaking, so the
    local mic never sends the bot's own voice back to Deepgram as if it were
    the caller talking over it.

    This is the fix PROGRESS.md's Phase 8 entry considered and deliberately
    did *not* apply, because it trades away genuine barge-in along with the
    self-echo it stops: there's no way to tell "the speaker playing the bot's
    voice" apart from "the caller actually interrupting" without real
    acoustic echo cancellation, so muting stops both for as long as the bot
    is talking. Revisited and accepted as a conscious tradeoff for local
    mic/speaker testing without headphones.

    Sits between transport.input() and the STT service. BotStartedSpeaking/
    BotStoppedSpeakingFrame are emitted by BaseOutputTransport and travel
    upstream all the way back from transport.output() to transport.input(),
    passing through this processor on the way — so no extra wiring is needed
    to observe them. On each one, this pushes an STTMuteFrame *downstream*
    (STTService.process_frame() handles it regardless of the direction it
    arrives from) to actually toggle the STT service's own `_muted` flag,
    which makes it drop incoming audio without transcribing it — the mic
    hardware itself stays on, only what reaches Deepgram is gated.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            await self.push_frame(STTMuteFrame(mute=True), FrameDirection.DOWNSTREAM)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self.push_frame(STTMuteFrame(mute=False), FrameDirection.DOWNSTREAM)

        await self.push_frame(frame, direction)


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


def build_pipeline(
    transport: BaseTransport,
    session: Session,
    *,
    mute_mic_during_tts: bool = False,
    on_escalation: EscalationHook | None = None,
) -> Pipeline:
    """Assemble the one pipeline shape both transports share:

        transport.input() -> [MicMuteGate] -> DeepgramFluxSTTService
            -> ClaudeTurnProcessor -> TTS service -> LatencyLogger
            -> transport.output()

    `transport` is the only thing that differs between transport/pipeline.py
    (LocalAudioTransport) and transport/telephony.py (FastAPIWebsocketTransport
    + TwilioFrameSerializer) — everything downstream of "raw audio in" is
    identical, which is the whole point of extracting it here.

    `mute_mic_during_tts` is opt-in and defaults off, so telephony.py's call
    (which doesn't pass it) is unaffected: a real caller's phone has no local
    speaker feeding back into a local mic, so there's no self-echo problem to
    fix there, and muting would only cost a real caller their barge-in for no
    benefit. transport/pipeline.py (local mic/speaker) opts in — see
    MicMuteGate's own docstring for the barge-in tradeoff that comes with it.
    """
    stt = DeepgramFluxSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = get_pipecat_tts_service()
    stages = [transport.input()]
    if mute_mic_during_tts:
        stages.append(MicMuteGate())
    stages += [
        stt,
        ClaudeTurnProcessor(session=session, on_escalation=on_escalation),
        tts,
        LatencyLogger(),
        transport.output(),
    ]
    return Pipeline(stages)
