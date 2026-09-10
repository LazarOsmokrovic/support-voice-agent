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
import re
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


# Markdown a speech synthesiser would otherwise read out loud. Ordered so
# the longer markers are consumed before their shorter prefixes — **bold**
# before *italic*, __bold__ before _italic_ — since matching the short form
# first would leave a stray marker behind.
_MARKDOWN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # A REAL newline after the fence, not an optional one. With \n? the
    # language-tag class [\w-]* happily swallowed an order ID — which is
    # entirely word characters and hyphens — leaving the capture empty and
    # DELETING it: "order ```112-3487561-2938471``` shipped" became "order
    # shipped". A reply that was nothing but a fenced identifier stripped to
    # empty, tripping the empty-reply guard so the caller heard nothing at
    # all. Exactly the defect class this function exists to prevent, in a
    # new place.
    (re.compile(r"```[\w-]*\n(.*?)```", re.DOTALL), r"\1"),   # fenced code block
    (re.compile(r"`([^`]+)`"), r"\1"),                         # inline code
    (re.compile(r"\*\*\*(.+?)\*\*\*", re.DOTALL), r"\1"),      # ***both***
    (re.compile(r"\*\*(.+?)\*\*", re.DOTALL), r"\1"),          # **bold**
    (re.compile(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", re.DOTALL), r"\1"),  # *italic*
    (re.compile(r"___(.+?)___", re.DOTALL), r"\1"),            # ___both___
    (re.compile(r"__(.+?)__", re.DOTALL), r"\1"),              # __bold__
    (re.compile(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)", re.DOTALL), r"\1"),    # _italic_
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),             # [text](url)
    (re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE), ""),      # # heading
    (re.compile(r"^\s{0,3}[-*+]\s+", re.MULTILINE), ""),       # - bullet
    (re.compile(r"^\s{0,3}>\s?", re.MULTILINE), ""),           # > quote
    # Anything backtick-shaped still standing after the rules above — an
    # unmatched fence, a stray pair around a same-line identifier. A
    # synthesiser pronounces them; nothing is lost by removing them, and
    # unlike the capturing rules this cannot delete what sits between.
    (re.compile(r"`+"), ""),
)


def speakable(text: str) -> str:
    """Strip markdown so a speech synthesiser does not read it aloud.

    A model writing "**wait for delivery**" is doing something reasonable —
    it is emphasising a phrase the way it would in text. But Deepgram and
    Cartesia both pronounce the asterisks, so the customer hears "star star
    wait for delivery star star" and the agent sounds broken. The same goes
    for backticks, bullet markers and heading hashes.

    The real fix is the system prompt, which now tells the model everything
    it says is spoken. This is the safety net: prompts are probabilistic and
    this failure is audible on every single slip, so it is worth catching
    deterministically at the one boundary every spoken word passes through.

    Deliberately NOT applied to transport/text_cli.py, where markdown is
    harmless and stripping it would only remove information a reader can see.
    """
    for pattern, replacement in _MARKDOWN_PATTERNS:
        text = pattern.sub(replacement, text)
    return text.strip()


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
