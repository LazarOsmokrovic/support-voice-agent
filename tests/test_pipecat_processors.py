"""Phases 8-9: transport/pipecat_processors.py's custom FrameProcessors,
shared by transport/pipeline.py (local mic) and transport/telephony.py
(Twilio). Moved here from tests/test_pipeline.py when the processors moved
out of transport/pipeline.py into this shared module on Twilio's arrival as
a second real use case — see that module's docstring for why.

Everything else in either pipeline (the transports, DeepgramFluxSTTService,
the TTS services) is Pipecat's own, already-shipped, already-tested code —
this file only tests the code this project actually wrote:
ClaudeTurnProcessor (the one place Pipecat and agent/session.py touch) and
LatencyLogger.

Testing approach: FrameProcessor(enable_direct_mode=True) is Pipecat's own
documented mechanism for processing frames synchronously with no internal
queue/task machinery — call process_frame() directly and inspect what a
linked capturing "sink" processor received via push_frame(). A processor
must first receive a StartFrame (as a real pipeline would send) before
push_frame() will actually deliver anything; skipping that step silently
drops every frame, which is itself a useful sanity check the tests below
rely on implicitly.

What this can't cover: real audio, a real Flux/TTS/Twilio connection, or the
framework's own interruption-delivery plumbing (InterruptionFrame routing,
the per-processor task cancellation it triggers) — that's Pipecat's own
tested code, not this project's. The stale-reply-after-cancellation test
below validates the same guarantee at the level this project controls:
cancelling the asyncio task actually running ClaudeTurnProcessor.process_frame()
(exactly what the framework's interruption handling does under the hood)
must not let a reply get pushed afterward. The actual "stress-test with
rapid interruptions and overlapping speech" (Phase 8) and "place a real
call... press 0" (Phase 9) are checkpoints that have to be run by hand —
same honest limitation as every voice checkpoint so far.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    InputDTMFFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    STTMuteFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from agent.prompts import GREETING
from agent.session import create_session
from agent.tools import escalation
from agent.tools.escalation import TurnClassification
from transport import pipecat_processors as processors
from transport.pipecat_processors import ClaudeTurnProcessor, LatencyLogger, MicMuteGate


class _CapturingSink(FrameProcessor):
    """A minimal processor that just records every frame pushed into it."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frames: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self.frames.append(frame)


async def _started(processor: FrameProcessor) -> _CapturingSink:
    """Link a capturing sink and send the StartFrame every processor needs
    before push_frame() will actually deliver anything. ClaudeTurnProcessor
    forwards the StartFrame itself (it's not a TranscriptionFrame or
    InputDTMFFrame, so it falls through to the pass-through branch) —
    cleared here so tests only see frames pushed by the turn under test, not
    this handshake frame.
    """
    sink = _CapturingSink(enable_direct_mode=True)
    processor.link(sink)
    await processor.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
    sink.frames.clear()
    return sink


async def _started_both_ways(processor: FrameProcessor) -> tuple[_CapturingSink, _CapturingSink]:
    """Like _started(), but also links a sink upstream (processor._prev) so
    tests can inspect what a processor pushes with FrameDirection.UPSTREAM —
    MicMuteGate needs this since it forwards Bot*SpeakingFrame upstream (back
    toward transport.input()) while pushing STTMuteFrame downstream (toward
    the STT service). Returns (upstream_sink, downstream_sink).
    """
    upstream_sink = _CapturingSink(enable_direct_mode=True)
    downstream_sink = _CapturingSink(enable_direct_mode=True)
    upstream_sink.link(processor)
    processor.link(downstream_sink)
    await processor.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
    downstream_sink.frames.clear()
    return upstream_sink, downstream_sink


def _transcript(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="caller", timestamp="")


def _text_response(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.stop_reason = "end_turn"
    return response


def _tool_use_response(name: str, tool_input: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.id = "toolu_test123"
    block.name = name
    block.input = tool_input
    response = MagicMock()
    response.content = [block]
    response.stop_reason = "tool_use"
    return response


def _all_thinking_phrases() -> set[str]:
    """Every filler the agent might speak while the model is thinking, so a
    test can assert "it said one of these" without pinning which."""
    from agent.prompts import THINKING_PHRASES

    return {phrase for phrases in THINKING_PHRASES.values() for phrase in phrases}


def _calm_classification():
    return TurnClassification(intent="chitchat", sentiment="neutral", policy_restricted=False)


@pytest.mark.asyncio
async def test_claude_turn_processor_greets_when_the_pipeline_starts():
    """The bot speaks first. On a phone call the alternative is the caller
    hearing silence until they say something, which reads as a dead line.

    Deliberately does NOT use _started(), since that helper clears exactly
    the frames under test here.
    """
    session = create_session("CUST-1001")
    processor = ClaudeTurnProcessor(session=session, enable_direct_mode=True)
    sink = _CapturingSink(enable_direct_mode=True)
    processor.link(sink)

    await processor.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)

    assert any(isinstance(f, TextFrame) and f.text == GREETING for f in sink.frames)
    # StartFrame must still reach the services downstream — they need it to
    # initialize, and it has to arrive before the greeting text does.
    assert isinstance(sink.frames[0], StartFrame)
    assert [type(f) for f in sink.frames] == [
        StartFrame,
        LLMFullResponseStartFrame,
        TextFrame,
        LLMFullResponseEndFrame,
    ]


@pytest.mark.asyncio
async def test_greeting_needs_no_model_call():
    """Answering a call must not wait on an API round-trip — the whole reason
    GREETING is a constant rather than a generated line."""
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=_text_response("should never be called"))
    session = create_session("CUST-1001", client=fake_client)
    processor = ClaudeTurnProcessor(session=session, enable_direct_mode=True)
    sink = _CapturingSink(enable_direct_mode=True)
    processor.link(sink)

    await processor.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)

    fake_client.messages.create.assert_not_called()


@pytest.mark.asyncio
async def test_claude_turn_processor_pushes_reply_and_absorbs_the_transcript(monkeypatch):
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=_text_response("Happy to help!"))
    monkeypatch.setattr(escalation, "classify_turn", AsyncMock(return_value=_calm_classification()))
    session = create_session("CUST-1001", client=fake_client)
    processor = ClaudeTurnProcessor(session=session, enable_direct_mode=True)
    sink = await _started(processor)

    await processor.process_frame(_transcript("Hi there"), FrameDirection.DOWNSTREAM)

    assert not any(isinstance(f, TranscriptionFrame) for f in sink.frames)
    assert [type(f) for f in sink.frames] == [
        LLMFullResponseStartFrame,
        TextFrame,
        LLMFullResponseEndFrame,
    ]
    # No thinking filler here, deliberately: the fake client answers instantly,
    # and Phase 10f only speaks a filler once a turn has taken longer than
    # THINKING_FILLER_DELAY_SECONDS. Padding a fast reply with "let me check
    # that" sounds worse than the silence it was meant to cover.
    assert sink.frames[1].text == "Happy to help!"


@pytest.mark.asyncio
async def test_claude_turn_processor_skips_tts_for_an_empty_reply(monkeypatch):
    """A reply that's empty/whitespace-only must never reach TTS: DeepgramTTSService
    would open a context, synthesize nothing, and log its own 3s pause-watchdog
    warning ("no BotStartedSpeakingFrame ... force-resuming") — reproduced live
    against the real pipeline, this locks in the fix.
    """
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=_text_response("   "))
    monkeypatch.setattr(escalation, "classify_turn", AsyncMock(return_value=_calm_classification()))
    session = create_session("CUST-1001", client=fake_client)
    processor = ClaudeTurnProcessor(session=session, enable_direct_mode=True)
    sink = await _started(processor)

    await processor.process_frame(_transcript("Hi there"), FrameDirection.DOWNSTREAM)

    assert not any(isinstance(f, TextFrame) for f in sink.frames)
    assert [type(f) for f in sink.frames] == [LLMFullResponseStartFrame, LLMFullResponseEndFrame]


@pytest.mark.asyncio
async def test_claude_turn_processor_ignores_an_empty_transcript(monkeypatch):
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=_text_response("should not be called"))
    monkeypatch.setattr(escalation, "classify_turn", AsyncMock(return_value=_calm_classification()))
    session = create_session("CUST-1001", client=fake_client)
    processor = ClaudeTurnProcessor(session=session, enable_direct_mode=True)
    sink = await _started(processor)

    await processor.process_frame(_transcript("   "), FrameDirection.DOWNSTREAM)

    assert sink.frames == []
    fake_client.messages.create.assert_not_called()


@pytest.mark.asyncio
async def test_claude_turn_processor_pushes_end_frame_when_model_ends_conversation(monkeypatch):
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(
        side_effect=[
            _tool_use_response("end_conversation", {}),
            _text_response("Take care!"),
        ]
    )
    monkeypatch.setattr(escalation, "classify_turn", AsyncMock(return_value=_calm_classification()))
    session = create_session("CUST-1001", client=fake_client)
    processor = ClaudeTurnProcessor(session=session, enable_direct_mode=True)
    sink = await _started(processor)

    await processor.process_frame(_transcript("Thanks, that's all!"), FrameDirection.DOWNSTREAM)

    assert isinstance(sink.frames[-1], EndFrame)
    assert any(isinstance(f, TextFrame) and f.text == "Take care!" for f in sink.frames)


@pytest.mark.asyncio
async def test_claude_turn_processor_pushes_notice_and_ends_on_escalation(monkeypatch, tmp_path):
    from data import mock_db

    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_pipecat_processors.db")
    mock_db.reset_and_seed()

    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=_text_response("Sure, one moment."))
    monkeypatch.setattr(
        escalation,
        "classify_turn",
        AsyncMock(return_value=TurnClassification(intent="request_human", sentiment="neutral", policy_restricted=False)),
    )
    monkeypatch.setattr(
        escalation,
        "create_handoff_packet",
        AsyncMock(return_value={"escalation_id": 42, "reason": "explicit request for a human"}),
    )
    session = create_session("CUST-1001", client=fake_client)
    processor = ClaudeTurnProcessor(session=session, enable_direct_mode=True)
    sink = await _started(processor)

    await processor.process_frame(_transcript("I want to talk to a human"), FrameDirection.DOWNSTREAM)

    # Matches on the customer-facing wording, not the internal handoff id —
    # which is deliberately no longer spoken. See test_session.py's
    # escalation-notice test for the reasoning.
    notice_frames = [f for f in sink.frames if isinstance(f, TextFrame) and "call you back" in f.text]
    assert len(notice_frames) == 1
    assert isinstance(sink.frames[-1], EndFrame)


@pytest.mark.asyncio
async def test_claude_turn_processor_drops_a_stale_reply_when_cancelled_mid_turn(monkeypatch):
    """Mirrors what the framework's real interruption handling does under
    the hood (cancel the task actually running process_frame()) without
    needing Pipecat's own InterruptionFrame plumbing running in the test.
    """
    started_call = asyncio.Event()
    release_call = asyncio.Event()

    async def _slow_create(**_kwargs):
        started_call.set()
        await release_call.wait()
        return _text_response("too late")

    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(side_effect=_slow_create)
    monkeypatch.setattr(escalation, "classify_turn", AsyncMock(return_value=_calm_classification()))
    session = create_session("CUST-1001", client=fake_client)
    processor = ClaudeTurnProcessor(session=session, enable_direct_mode=True)
    sink = await _started(processor)

    task = asyncio.create_task(processor.process_frame(_transcript("Hi"), FrameDirection.DOWNSTREAM))
    await started_call.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The thinking filler is spoken before run_turn is even called, so it has
    # legitimately already left by the time the cancellation lands. What must
    # NOT escape is the reply — stale audio arriving after a barge-in is the
    # bug this test exists for, and that guarantee is unchanged.
    assert all(
        not isinstance(f, TextFrame) or f.text in _all_thinking_phrases() for f in sink.frames
    ), "a stale reply escaped after the turn was cancelled"


@pytest.mark.asyncio
async def test_claude_turn_processor_escalates_on_dtmf_zero_independent_of_the_model(monkeypatch, tmp_path):
    """The DTMF safety net must work even if the model/classifier is doing
    nothing at all — no run_turn(), no classify_turn(), just the digit.
    """
    from data import mock_db

    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_dtmf.db")
    mock_db.reset_and_seed()

    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=_text_response("should never be called"))
    monkeypatch.setattr(
        escalation,
        "create_handoff_packet",
        AsyncMock(return_value={"escalation_id": 7, "reason": "caller pressed 0 for a human"}),
    )
    session = create_session("CUST-1001", client=fake_client)
    processor = ClaudeTurnProcessor(session=session, enable_direct_mode=True)
    sink = await _started(processor)

    await processor.process_frame(InputDTMFFrame(KeypadEntry.ZERO), FrameDirection.DOWNSTREAM)

    fake_client.messages.create.assert_not_called()
    assert [type(f) for f in sink.frames] == [LLMFullResponseStartFrame, TextFrame, LLMFullResponseEndFrame, EndFrame]
    assert "handoff #7" in sink.frames[1].text


@pytest.mark.asyncio
async def test_claude_turn_processor_ignores_dtmf_digits_other_than_zero(monkeypatch):
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=_text_response("should never be called"))
    session = create_session("CUST-1001", client=fake_client)
    processor = ClaudeTurnProcessor(session=session, enable_direct_mode=True)
    sink = await _started(processor)

    await processor.process_frame(InputDTMFFrame(KeypadEntry.FIVE), FrameDirection.DOWNSTREAM)

    assert sink.frames == []
    fake_client.messages.create.assert_not_called()


@pytest.mark.asyncio
async def test_claude_turn_processor_dtmf_escalation_survives_a_handoff_packet_failure(monkeypatch):
    """The fallback must still end the call gracefully even if logging the
    handoff packet itself fails — this path exists specifically so a broken
    dependency (e.g. no DB, no API credit) can't strand the caller.
    """
    session = create_session("CUST-1001")
    monkeypatch.setattr(
        escalation, "create_handoff_packet", AsyncMock(side_effect=RuntimeError("db unavailable"))
    )
    processor = ClaudeTurnProcessor(session=session, enable_direct_mode=True)
    sink = await _started(processor)

    await processor.process_frame(InputDTMFFrame(KeypadEntry.ZERO), FrameDirection.DOWNSTREAM)

    assert isinstance(sink.frames[-1], EndFrame)
    assert any(isinstance(f, TextFrame) and "human agent" in f.text for f in sink.frames)


@pytest.mark.asyncio
async def test_dtmf_escalation_emits_a_turn_record(tmp_path, monkeypatch):
    """Pressing 0 bypasses run_turn by design, so it needs its own record —
    otherwise the single most important escalation leaves no trace."""
    import json

    from data import mock_db

    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_dtmf_log.db")
    mock_db.reset_and_seed()
    path = tmp_path / "turns.jsonl"
    monkeypatch.setenv("TURN_LOG_PATH", str(path))
    monkeypatch.setattr(
        escalation, "create_handoff_packet",
        AsyncMock(return_value={"escalation_id": 7, "reason": "caller pressed 0 for a human"}),
    )
    session = create_session("CUST-1001", transport="telephony")
    processor = ClaudeTurnProcessor(session=session, enable_direct_mode=True)
    await _started(processor)

    await processor.process_frame(InputDTMFFrame(KeypadEntry.ZERO), FrameDirection.DOWNSTREAM)

    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(records) == 1
    assert records[0]["escalated"] is True
    assert records[0]["transport"] == "telephony"
    assert "DTMF" in records[0]["user_text"]
    assert records[0]["grounding_flagged"] is False
    assert records[0]["hedge_spoken"] is False
    assert records[0]["escalation_id"] == 7
    assert records[0]["turn"] == 1


@pytest.mark.asyncio
async def test_dtmf_escalation_does_not_duplicate_the_preceding_spoken_turn_number(monkeypatch, tmp_path):
    """The DTMF handler bypasses run_turn entirely, so it advances
    session.turn itself — without that, its record would carry the SAME
    turn number as the spoken turn immediately before it, colliding at the
    one event this phase went out of its way to cover."""
    import json

    from data import mock_db

    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_dtmf_dedup.db")
    mock_db.reset_and_seed()
    path = tmp_path / "turns.jsonl"
    monkeypatch.setenv("TURN_LOG_PATH", str(path))
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=_text_response("Happy to help!"))
    monkeypatch.setattr(escalation, "classify_turn", AsyncMock(return_value=_calm_classification()))
    monkeypatch.setattr(
        escalation, "create_handoff_packet",
        AsyncMock(return_value={"escalation_id": 7, "reason": "caller pressed 0 for a human"}),
    )
    session = create_session("CUST-1001", client=fake_client, transport="telephony")
    processor = ClaudeTurnProcessor(session=session, enable_direct_mode=True)
    await _started(processor)

    await processor.process_frame(_transcript("Hi there"), FrameDirection.DOWNSTREAM)
    await processor.process_frame(InputDTMFFrame(KeypadEntry.ZERO), FrameDirection.DOWNSTREAM)

    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(records) == 2
    spoken_turn, dtmf_turn = records[0]["turn"], records[1]["turn"]
    assert dtmf_turn != spoken_turn
    assert dtmf_turn > spoken_turn


@pytest.mark.asyncio
async def test_latency_logger_logs_the_round_trip_once_per_turn(capsys):
    processor = LatencyLogger(enable_direct_mode=True)
    await _started(processor)

    await processor.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(
        TTSAudioRawFrame(audio=b"\x00\x00", sample_rate=24000, num_channels=1), FrameDirection.DOWNSTREAM
    )
    await processor.process_frame(
        TTSAudioRawFrame(audio=b"\x00\x00", sample_rate=24000, num_channels=1), FrameDirection.DOWNSTREAM
    )

    output = capsys.readouterr().out
    assert output.count("[latency] round-trip") == 1


@pytest.mark.asyncio
async def test_latency_logger_forwards_every_frame(capsys):
    processor = LatencyLogger(enable_direct_mode=True)
    sink = await _started(processor)

    await processor.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    audio_frame = TTSAudioRawFrame(audio=b"\x00\x00", sample_rate=24000, num_channels=1)
    await processor.process_frame(audio_frame, FrameDirection.DOWNSTREAM)

    assert audio_frame in sink.frames
    assert any(isinstance(f, UserStoppedSpeakingFrame) for f in sink.frames)


@pytest.mark.asyncio
async def test_mic_mute_gate_mutes_stt_when_the_bot_starts_speaking():
    processor = MicMuteGate(enable_direct_mode=True)
    upstream_sink, downstream_sink = await _started_both_ways(processor)

    frame = BotStartedSpeakingFrame()
    await processor.process_frame(frame, FrameDirection.UPSTREAM)

    # STTMuteFrame goes downstream (toward the STT service sitting right
    # after this gate in the real pipeline); the original frame keeps
    # travelling upstream (toward transport.input()) unmodified.
    assert [type(f) for f in downstream_sink.frames] == [STTMuteFrame]
    assert downstream_sink.frames[0].mute is True
    assert upstream_sink.frames == [frame]


@pytest.mark.asyncio
async def test_mic_mute_gate_unmutes_stt_when_the_bot_stops_speaking():
    processor = MicMuteGate(enable_direct_mode=True)
    upstream_sink, downstream_sink = await _started_both_ways(processor)

    frame = BotStoppedSpeakingFrame()
    await processor.process_frame(frame, FrameDirection.UPSTREAM)

    assert [type(f) for f in downstream_sink.frames] == [STTMuteFrame]
    assert downstream_sink.frames[0].mute is False
    assert upstream_sink.frames == [frame]


@pytest.mark.asyncio
async def test_mic_mute_gate_passes_through_unrelated_frames_untouched():
    processor = MicMuteGate(enable_direct_mode=True)
    upstream_sink, downstream_sink = await _started_both_ways(processor)

    downstream_frame = TextFrame(text="irrelevant")
    await processor.process_frame(downstream_frame, FrameDirection.DOWNSTREAM)

    # No STTMuteFrame synthesized, and the frame keeps going the direction it
    # was already travelling.
    assert downstream_sink.frames == [downstream_frame]
    assert upstream_sink.frames == []


@pytest.mark.asyncio
async def test_dtmf_escalation_fires_the_transfer_callback(monkeypatch):
    """Press 0 is the safety net for when the AI is already failing — it MUST
    transfer, not just announce a transfer."""
    fired = []

    async def _on_escalation(packet):
        fired.append(packet)
        return True

    session = create_session("CUST-1001")
    monkeypatch.setattr(
        escalation, "create_handoff_packet", AsyncMock(return_value={"escalation_id": 5})
    )
    processor = ClaudeTurnProcessor(session=session, on_escalation=_on_escalation, enable_direct_mode=True)
    await _started(processor)

    await processor.process_frame(InputDTMFFrame(button=KeypadEntry.ZERO), FrameDirection.DOWNSTREAM)

    assert len(fired) == 1
    assert fired[0]["escalation_id"] == 5


@pytest.mark.asyncio
async def test_model_driven_escalation_fires_the_transfer_callback(monkeypatch):
    """The other path: run_turn decided to escalate."""
    fired = []

    async def _on_escalation(packet):
        fired.append(packet)
        return True

    session = create_session("CUST-1001")
    outcome = SimpleNamespace(
        reply="Connecting you with a human agent.",
        notice=None,
        warnings=[],
        llm_latency_seconds=0.0,
        ended=True,
        end_reason="escalated",
        escalation_packet={"escalation_id": 8},
    )
    monkeypatch.setattr(
        "transport.pipecat_processors.run_turn", AsyncMock(return_value=outcome)
    )
    processor = ClaudeTurnProcessor(session=session, on_escalation=_on_escalation, enable_direct_mode=True)
    await _started(processor)

    await processor.process_frame(_transcript("get me a human"), FrameDirection.DOWNSTREAM)

    assert len(fired) == 1


@pytest.mark.asyncio
async def test_no_callback_means_todays_behaviour_is_unchanged(monkeypatch):
    """transport/pipeline.py (local mic) passes no callback. It must behave
    exactly as it did before this phase."""
    session = create_session("CUST-1001")
    monkeypatch.setattr(
        escalation, "create_handoff_packet", AsyncMock(return_value={"escalation_id": 5})
    )
    processor = ClaudeTurnProcessor(session=session, enable_direct_mode=True)
    sink = await _started(processor)

    await processor.process_frame(InputDTMFFrame(button=KeypadEntry.ZERO), FrameDirection.DOWNSTREAM)

    assert any(isinstance(f, TextFrame) and "human agent" in f.text for f in sink.frames)


@pytest.mark.asyncio
async def test_a_raising_callback_never_breaks_the_call(monkeypatch):
    """Telephony failures must not crash the pipeline — the customer still
    hears the notice."""
    async def _on_escalation(packet):
        raise RuntimeError("twilio exploded")

    session = create_session("CUST-1001")
    monkeypatch.setattr(
        escalation, "create_handoff_packet", AsyncMock(return_value={"escalation_id": 5})
    )
    processor = ClaudeTurnProcessor(session=session, on_escalation=_on_escalation, enable_direct_mode=True)
    sink = await _started(processor)

    await processor.process_frame(InputDTMFFrame(button=KeypadEntry.ZERO), FrameDirection.DOWNSTREAM)

    assert any(isinstance(f, TextFrame) for f in sink.frames)


@pytest.mark.asyncio
async def test_a_slow_turn_is_covered_by_a_thinking_filler(monkeypatch):
    """The other half of the filler design: a turn that actually takes time.

    Silence is the worst thing a voice agent can do — on a phone line a
    multi-second gap reads as a dropped call, and the caller starts saying
    "hello? are you there?" over the reply as it arrives. So a slow turn gets
    an acknowledgement, chosen deterministically from the caller's own words
    so it costs no extra model round-trip.

    Crucially the model call is already in flight while the filler plays:
    pushing a TextFrame only queues it downstream, so synthesis overlaps the
    rest of the turn rather than delaying it. This test proves the filler is
    spoken BEFORE the reply, which is only possible if the two overlap.
    """
    slow = asyncio.Event()

    async def _slow_run_turn(session, text):
        await asyncio.sleep(processors.THINKING_FILLER_DELAY_SECONDS + 0.2)
        slow.set()
        return SimpleNamespace(
            reply="Your order is out for delivery.",
            notice=None,
            warnings=[],
            llm_latency_seconds=0.0,
            ended=False,
            end_reason=None,
            escalation_packet=None,
        )

    monkeypatch.setattr(processors, "run_turn", _slow_run_turn)
    session = create_session("CUST-1001")
    processor = ClaudeTurnProcessor(session=session, enable_direct_mode=True)
    sink = await _started(processor)

    # A POLICY question, deliberately: search_policy needs nothing from the
    # caller, so the agent really can start work and the filler is honest.
    # An order question with no ID in play is now suppressed instead.
    await processor.process_frame(
        _transcript("how long do I have to return something"), FrameDirection.DOWNSTREAM
    )

    spoken = [f.text for f in sink.frames if isinstance(f, TextFrame)]
    assert len(spoken) == 2, f"expected a filler then the reply, got {spoken}"
    assert spoken[0] in _all_thinking_phrases()
    assert spoken[1] == "Your order is out for delivery."
    assert slow.is_set()


@pytest.mark.asyncio
async def test_a_greeting_gets_no_thinking_filler(monkeypatch):
    """The first thing a real caller noticed.

    Saying "hello" to the agent produced "let me check that for you" and
    THEN "hi, how can I help?". Nothing was being checked — the caller had
    not asked for anything. A person says hello back.

    Uses a deliberately slow turn so the filler WOULD fire on a substantive
    question; the point is that a social turn suppresses it regardless of
    how long the model takes.
    """
    async def _slow_run_turn(session, text):
        await asyncio.sleep(processors.THINKING_FILLER_DELAY_SECONDS + 0.2)
        return SimpleNamespace(
            reply="Hi there! How can I help?",
            notice=None,
            warnings=[],
            llm_latency_seconds=0.0,
            ended=False,
            end_reason=None,
            escalation_packet=None,
        )

    monkeypatch.setattr(processors, "run_turn", _slow_run_turn)
    session = create_session("CUST-1001")
    processor = ClaudeTurnProcessor(session=session, enable_direct_mode=True)
    sink = await _started(processor)

    await processor.process_frame(_transcript("Hello"), FrameDirection.DOWNSTREAM)

    spoken = [f.text for f in sink.frames if isinstance(f, TextFrame)]
    assert spoken == ["Hi there! How can I help?"], f"a greeting must not be filled: {spoken}"


@pytest.mark.asyncio
async def test_a_goodbye_gets_no_thinking_filler(monkeypatch):
    """The same mistake at the other end of the call: "let me check
    that... goodbye" checks nothing. The call is finishing."""
    async def _slow_run_turn(session, text):
        await asyncio.sleep(processors.THINKING_FILLER_DELAY_SECONDS + 0.2)
        return SimpleNamespace(
            reply="Glad I could help — take care!",
            notice=None,
            warnings=[],
            llm_latency_seconds=0.0,
            ended=True,
            end_reason="model_ended",
            escalation_packet=None,
        )

    monkeypatch.setattr(processors, "run_turn", _slow_run_turn)
    session = create_session("CUST-1001")
    processor = ClaudeTurnProcessor(session=session, enable_direct_mode=True)
    sink = await _started(processor)

    await processor.process_frame(_transcript("No that's everything, thanks"), FrameDirection.DOWNSTREAM)

    spoken = [f.text for f in sink.frames if isinstance(f, TextFrame)]
    assert spoken == ["Glad I could help — take care!"], f"a goodbye must not be filled: {spoken}"


@pytest.mark.asyncio
async def test_an_order_question_with_no_id_yet_gets_no_filler(monkeypatch):
    """Noticed live: "can you help me with my order" produced "sure, let me
    look into that" and then "actually, I need the order ID".

    The agent cannot begin an order lookup without a number, so a filler
    there promises work that has not started — worse than silence, because
    it claims to be doing something impossible. The honest reply is simply
    "I can do that, what's the order number?".

    Uses a deliberately slow turn, so the filler would fire on a question
    the agent COULD act on. The suppression is about capability, not speed.
    """
    async def _slow_run_turn(session, text):
        await asyncio.sleep(processors.THINKING_FILLER_DELAY_SECONDS + 0.2)
        return SimpleNamespace(
            reply="Of course — what's the order number?",
            notice=None,
            warnings=[],
            llm_latency_seconds=0.0,
            ended=False,
            end_reason=None,
            escalation_packet=None,
        )

    monkeypatch.setattr(processors, "run_turn", _slow_run_turn)
    session = create_session("CUST-1001")
    processor = ClaudeTurnProcessor(session=session, enable_direct_mode=True)
    sink = await _started(processor)

    await processor.process_frame(
        _transcript("Hi, can you help me with my order?"), FrameDirection.DOWNSTREAM
    )

    spoken = [f.text for f in sink.frames if isinstance(f, TextFrame)]
    assert spoken == ["Of course — what's the order number?"], (
        f"an order question with no ID must not be filled: {spoken}"
    )


@pytest.mark.asyncio
async def test_an_order_question_is_filled_once_the_id_is_known(monkeypatch):
    """The other half: once an order ID is in the conversation the agent CAN
    look something up, so covering the wait is honest again.

    The ID is read from the transcript rather than tracked as state, because
    that is where it actually arrives — via a tool call the model composes.
    """
    from data import mock_db

    order_id = mock_db.ORDERS[0][0]

    async def _slow_run_turn(session, text):
        await asyncio.sleep(processors.THINKING_FILLER_DELAY_SECONDS + 0.2)
        return SimpleNamespace(
            reply="It's out for delivery.",
            notice=None,
            warnings=[],
            llm_latency_seconds=0.0,
            ended=False,
            end_reason=None,
            escalation_packet=None,
        )

    monkeypatch.setattr(processors, "run_turn", _slow_run_turn)
    session = create_session("CUST-1001")
    session.agent.messages.append({"role": "user", "content": f"my order is {order_id}"})
    processor = ClaudeTurnProcessor(session=session, enable_direct_mode=True)
    sink = await _started(processor)

    await processor.process_frame(_transcript("where is my order"), FrameDirection.DOWNSTREAM)

    spoken = [f.text for f in sink.frames if isinstance(f, TextFrame)]
    assert len(spoken) == 2, f"expected a filler then the reply, got {spoken}"
    assert spoken[0] in _all_thinking_phrases()
    assert spoken[1] == "It's out for delivery."
