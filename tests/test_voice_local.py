"""Phase 7: transport/voice_local.py.

Two tiers, same convention as every prior phase's live tests:

- Mocked-connection tests (always run): the Flux control-flow logic
  (listen_and_transcribe stopping exactly at EndOfTurn, surfacing a fatal
  error) and WAV decoding/playback, all against fakes — no real audio
  device, no network. sounddevice's RawInputStream/play/wait are
  monkeypatched so these never touch real hardware.
- A live round-trip test, gated on DEEPGRAM_API_KEY: synthesizes a known
  phrase with the real TTS backend and feeds the audio straight into a
  real Flux connection, asserting the transcript reasonably matches. This
  proves the STT<->TTS integration end-to-end with zero human voice
  needed.

What none of this can cover: an actual person speaking into an actual
microphone. That's this phase's real checkpoint, and it has to be run by
hand — `python -m transport.voice_local` — see PROJECT_PLAN.md and README.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from transport.tts import get_tts_backend
from transport.voice_local import FLUX_MODEL, _decode_wav, listen_and_transcribe, speak


def _make_wav_bytes(samples: list[int], sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(np.array(samples, dtype=np.int16).tobytes())
    return buf.getvalue()


class _FakeSocket:
    """Stands in for AsyncV2SocketClient: async-iterable over canned
    messages, send_media is a no-op AsyncMock.
    """

    def __init__(self, messages: list[SimpleNamespace]) -> None:
        self._messages = messages
        self.send_media = AsyncMock()

    async def __aiter__(self):
        for message in self._messages:
            yield message


class _FakeConnect:
    def __init__(self, socket: _FakeSocket) -> None:
        self._socket = socket

    async def __aenter__(self) -> _FakeSocket:
        return self._socket

    async def __aexit__(self, *_exc) -> bool:
        return False


def _fake_deepgram_client(messages: list[SimpleNamespace]) -> MagicMock:
    """A MagicMock shaped like AsyncDeepgramClient: .listen.v2.connect(**kw)
    returns an async context manager yielding a fake socket.
    """
    socket = _FakeSocket(messages)
    client = MagicMock()
    client.listen.v2.connect = MagicMock(return_value=_FakeConnect(socket))
    return client


class _FakeRawInputStream:
    """Stands in for sounddevice.RawInputStream — never opens a real mic.
    The callback is simply never invoked, so the sender task's queue stays
    empty and it blocks on audio_queue.get() until cancelled, exactly like
    a real stream would if no test ever fed it audio.
    """

    def __init__(self, **_kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> bool:
        return False


@pytest.mark.asyncio
async def test_listen_and_transcribe_stops_at_end_of_turn(monkeypatch):
    from transport import voice_local

    monkeypatch.setattr(voice_local.sd, "RawInputStream", _FakeRawInputStream)
    messages = [
        SimpleNamespace(type="Connected"),
        SimpleNamespace(type="TurnInfo", event="StartOfTurn", transcript=""),
        SimpleNamespace(type="TurnInfo", event="Update", transcript="What is"),
        SimpleNamespace(type="TurnInfo", event="EndOfTurn", transcript="What is my order status"),
    ]
    client = _fake_deepgram_client(messages)

    transcript, latency = await listen_and_transcribe(client)

    assert transcript == "What is my order status"
    assert latency >= 0
    client.listen.v2.connect.assert_called_once_with(model=FLUX_MODEL, encoding="linear16", sample_rate=16000)


@pytest.mark.asyncio
async def test_listen_and_transcribe_raises_on_a_fatal_error(monkeypatch):
    from transport import voice_local

    monkeypatch.setattr(voice_local.sd, "RawInputStream", _FakeRawInputStream)
    messages = [SimpleNamespace(type="Error", code="INTERNAL_SERVER_ERROR", description="socket died")]
    client = _fake_deepgram_client(messages)

    with pytest.raises(RuntimeError, match="socket died"):
        await listen_and_transcribe(client)


def test_decode_wav_round_trips_pcm_samples():
    samples = [0, 1000, -1000, 32767, -32768]
    wav_bytes = _make_wav_bytes(samples, sample_rate=24000)

    audio, sample_rate = _decode_wav(wav_bytes)

    assert sample_rate == 24000
    assert audio.tolist() == samples


@pytest.mark.asyncio
async def test_speak_plays_the_decoded_audio(monkeypatch):
    from transport import voice_local

    play_mock = MagicMock()
    monkeypatch.setattr(voice_local.sd, "play", play_mock)
    monkeypatch.setattr(voice_local.sd, "wait", MagicMock())
    fake_backend = MagicMock()
    fake_backend.synthesize = AsyncMock(return_value=_make_wav_bytes([1, 2, 3], sample_rate=24000))

    latency = await speak("Hello there", fake_backend)

    assert latency >= 0
    fake_backend.synthesize.assert_awaited_once_with("Hello there")
    played_audio, played_kwargs = play_mock.call_args
    assert played_audio[0].tolist() == [1, 2, 3]
    assert played_kwargs["samplerate"] == 24000


@pytest.mark.skipif(not os.getenv("DEEPGRAM_API_KEY"), reason="requires a real DEEPGRAM_API_KEY")
@pytest.mark.asyncio
async def test_tts_to_flux_round_trip_recognizes_synthesized_speech():
    """Live, automated STT<->TTS integration check: no human mic needed,
    but it does spend real API credit against both services (Deepgram
    always; whichever TTS_BACKEND is configured).
    """
    from deepgram import AsyncDeepgramClient

    backend = get_tts_backend()
    audio_bytes = await backend.synthesize("I would like to check my order status please")
    audio, sample_rate = _decode_wav(audio_bytes)
    raw_pcm = audio.astype(np.int16).tobytes()

    client = AsyncDeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))
    transcript = ""
    chunk_size = 3200

    async with client.listen.v2.connect(model=FLUX_MODEL, encoding="linear16", sample_rate=sample_rate) as socket:

        async def _send() -> None:
            for i in range(0, len(raw_pcm), chunk_size):
                await socket.send_media(raw_pcm[i : i + chunk_size])
                await asyncio.sleep(0.05)
            # Trailing silence so Flux's end-of-turn timer actually fires —
            # otherwise it just keeps waiting for the speaker to continue.
            silence = b"\x00" * chunk_size
            for _ in range(20):
                await socket.send_media(silence)
                await asyncio.sleep(0.05)

        sender = asyncio.create_task(_send())
        try:
            async for message in socket:
                if getattr(message, "type", None) == "TurnInfo" and message.event == "EndOfTurn":
                    transcript = message.transcript
                    break
        finally:
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender

    assert "order" in transcript.lower()
