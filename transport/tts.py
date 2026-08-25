"""Swappable text-to-speech backend: Deepgram Aura-2 (default) or Cartesia
Sonic, selected via the TTS_BACKEND env var. Mirrors
agent/tools/policy_rag.py's EmbeddingBackend pattern from Phase 3 —
built that way for the identical reason: Deepgram is mandatory anyway
(Flux, for STT, has no alternative), so defaulting TTS to it too means no
second signup; Cartesia stays available as a one-env-var swap for whoever
wants its independently-benchmarked edge on voice naturalness.

Both are one-shot REST calls, not the streaming/WebSocket APIs each
provider also offers — Phase 7 is a simple sequential local loop, not
real-time streaming (that's Phase 8's job, via Pipecat).
"""

from __future__ import annotations

import os
from typing import Protocol

import httpx

DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak"
CARTESIA_TTS_URL = "https://api.cartesia.ai/tts/bytes"
CARTESIA_API_VERSION = "2026-08-14"  # required header; bump if Cartesia deprecates this version

DEFAULT_DEEPGRAM_VOICE = "aura-2-thalia-en"
DEFAULT_CARTESIA_MODEL = "sonic-3.5"
# A real voice ID from Cartesia's own documented example — a placeholder
# until you pick one from your own voice library once actually using this
# backend (see CARTESIA_TTS_VOICE below).
DEFAULT_CARTESIA_VOICE = "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"


class TTSBackend(Protocol):
    async def synthesize(self, text: str) -> bytes: ...


class DeepgramTTSBackend:
    """One-shot REST call to Deepgram's /v1/speak — Aura-2 by default."""

    def __init__(self, voice: str | None = None, api_key: str | None = None) -> None:
        self.voice = voice or os.getenv("DEEPGRAM_TTS_VOICE", DEFAULT_DEEPGRAM_VOICE)
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")

    async def synthesize(self, text: str) -> bytes:
        """Returns raw audio bytes (WAV, linear16). Raises httpx.HTTPStatusError
        on failure.

        `encoding`/`container` are explicit, not defaults — Deepgram's
        default response is MP3, which the stdlib `wave` module used for
        playback in transport/voice_local.py cannot decode.
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                DEEPGRAM_TTS_URL,
                params={"model": self.voice, "encoding": "linear16", "container": "wav"},
                json={"text": text},
                headers={"Authorization": f"Token {self.api_key}"},
                timeout=30.0,
            )
            response.raise_for_status()
            return response.content


class CartesiaTTSBackend:
    """One-shot REST call to Cartesia's /tts/bytes — Sonic by default."""

    def __init__(self, model: str | None = None, voice: str | None = None, api_key: str | None = None) -> None:
        self.model = model or os.getenv("CARTESIA_TTS_MODEL", DEFAULT_CARTESIA_MODEL)
        self.voice = voice or os.getenv("CARTESIA_TTS_VOICE", DEFAULT_CARTESIA_VOICE)
        self.api_key = api_key or os.getenv("CARTESIA_API_KEY")

    async def synthesize(self, text: str) -> bytes:
        """Returns raw audio bytes (WAV, pcm_s16le @ 24kHz). Raises
        httpx.HTTPStatusError on failure.
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                CARTESIA_TTS_URL,
                json={
                    "model_id": self.model,
                    "transcript": text,
                    "voice": self.voice,
                    "output_format": {"container": "wav", "encoding": "pcm_s16le", "sample_rate": 24000},
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Cartesia-Version": CARTESIA_API_VERSION,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            return response.content


def get_tts_backend() -> TTSBackend:
    """TTS_BACKEND env var: "deepgram" (default) or "cartesia"."""
    backend_name = os.getenv("TTS_BACKEND", "deepgram").lower()
    if backend_name == "cartesia":
        return CartesiaTTSBackend()
    if backend_name != "deepgram":
        raise ValueError(f"unknown TTS_BACKEND: {backend_name!r} (expected 'deepgram' or 'cartesia')")
    return DeepgramTTSBackend()
