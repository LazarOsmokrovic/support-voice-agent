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
