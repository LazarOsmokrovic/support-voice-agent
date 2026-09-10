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
