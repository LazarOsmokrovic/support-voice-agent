"""Phase 8: transport/pipeline.py's two custom FrameProcessors.

Everything else in the pipeline (LocalAudioTransport, DeepgramFluxSTTService,
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

What this can't cover: real audio, a real Flux/TTS connection, or the
framework's own interruption-delivery plumbing (InterruptionFrame routing,
the per-processor task cancellation it triggers) — that's Pipecat's own
tested code, not this project's. The stale-reply-after-cancellation test
below validates the same guarantee at the level this project controls:
cancelling the asyncio task actually running ClaudeTurnProcessor.process_frame()
(exactly what the framework's interruption handling does under the hood)
must not let a reply get pushed afterward. The actual "stress-test with
rapid interruptions and overlapping speech" is this phase's real checkpoint
and has to be run by hand — python -m transport.pipeline — same honest
limitation as every voice checkpoint so far.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from agent.session import create_session
from agent.tools import escalation
from agent.tools.escalation import TurnClassification
from transport.pipeline import ClaudeTurnProcessor, LatencyLogger


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
    forwards the StartFrame itself (it's not a TranscriptionFrame, so it
    falls through to the pass-through branch) — cleared here so tests only
    see frames pushed by the turn under test, not this handshake frame.
    """
    sink = _CapturingSink(enable_direct_mode=True)
    processor.link(sink)
    await processor.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
    sink.frames.clear()
    return sink


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


def _calm_classification():
    return TurnClassification(intent="chitchat", sentiment="neutral", policy_restricted=False)


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

    monkeypatch.setattr(mock_db, "DB_PATH", tmp_path / "test_pipeline.db")
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

    notice_frames = [f for f in sink.frames if isinstance(f, TextFrame) and "handoff #42" in f.text]
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

    assert sink.frames == []


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
