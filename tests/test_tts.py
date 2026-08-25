"""Phase 7: the swappable TTS backend, tested against mocked HTTP responses
(pytest-httpx) — no real network, no API key, mirroring how
tests/test_policy_rag.py tests the swappable embedding backend.
"""

from __future__ import annotations

import httpx
import pytest

from transport.tts import (
    CARTESIA_TTS_URL,
    DEEPGRAM_TTS_URL,
    CartesiaTTSBackend,
    DeepgramTTSBackend,
    get_tts_backend,
)


@pytest.mark.asyncio
async def test_deepgram_backend_posts_to_the_speak_endpoint(httpx_mock):
    httpx_mock.add_response(
        url=httpx.URL(
            DEEPGRAM_TTS_URL, params={"model": "aura-2-thalia-en", "encoding": "linear16", "container": "wav"}
        ),
        content=b"FAKEWAV",
    )
    backend = DeepgramTTSBackend(voice="aura-2-thalia-en", api_key="test-key")

    audio = await backend.synthesize("Hello there")

    assert audio == b"FAKEWAV"
    request = httpx_mock.get_requests()[0]
    assert request.headers["authorization"] == "Token test-key"
    assert b"Hello there" in request.content


@pytest.mark.asyncio
async def test_cartesia_backend_posts_to_the_bytes_endpoint(httpx_mock):
    httpx_mock.add_response(url=CARTESIA_TTS_URL, content=b"FAKEWAV2")
    backend = CartesiaTTSBackend(model="sonic-3.5", voice="some-voice-id", api_key="test-key")

    audio = await backend.synthesize("Hello there")

    assert audio == b"FAKEWAV2"
    request = httpx_mock.get_requests()[0]
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["cartesia-version"]
    assert b"sonic-3.5" in request.content
    assert b"some-voice-id" in request.content


@pytest.mark.asyncio
async def test_backend_raises_on_an_http_error(httpx_mock):
    httpx_mock.add_response(
        url=httpx.URL(
            DEEPGRAM_TTS_URL, params={"model": "aura-2-thalia-en", "encoding": "linear16", "container": "wav"}
        ),
        status_code=401,
    )
    backend = DeepgramTTSBackend(voice="aura-2-thalia-en", api_key="bad-key")

    with pytest.raises(httpx.HTTPStatusError):
        await backend.synthesize("Hello there")


def test_get_tts_backend_defaults_to_deepgram(monkeypatch):
    monkeypatch.delenv("TTS_BACKEND", raising=False)
    assert isinstance(get_tts_backend(), DeepgramTTSBackend)


def test_get_tts_backend_respects_the_env_var(monkeypatch):
    monkeypatch.setenv("TTS_BACKEND", "cartesia")
    assert isinstance(get_tts_backend(), CartesiaTTSBackend)


def test_get_tts_backend_rejects_an_unknown_value(monkeypatch):
    monkeypatch.setenv("TTS_BACKEND", "not-a-real-backend")
    with pytest.raises(ValueError, match="unknown TTS_BACKEND"):
        get_tts_backend()
