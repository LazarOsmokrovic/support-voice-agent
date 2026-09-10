"""Local mic/speaker voice loop — Phase 7 interface.

Wires a real microphone and speakers to the *exact same* orchestration
transport/text_cli.py uses: agent/session.py's create_session/run_turn/
close_session. Deepgram Flux (live STT) replaces input(); a swappable TTS
backend (transport/tts.py) plus sounddevice playback replaces print(). No
business logic lives here — see CLAUDE.md rule 5 and agent/session.py's
docstring for why that split exists.

STT: Deepgram Flux is a live WebSocket model, not a one-shot REST call —
its whole value is built-in end-of-turn detection, so this loop streams mic
audio in and waits for Flux's own "EndOfTurn" event rather than hand-rolling
silence detection. Strictly sequential, same as the plan calls for: the
agent finishes speaking, *then* starts listening for the next turn. No
barge-in, no partial-transcript handling — that's Phase 8 (Pipecat).

Run with: python -m transport.voice_local
Requires DEEPGRAM_API_KEY in .env (and CARTESIA_API_KEY too, only if
TTS_BACKEND=cartesia).
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import time
import wave

import numpy as np
import sounddevice as sd
from deepgram import AsyncDeepgramClient

from agent.core import configure_logging
from agent.prompts import GREETING
from agent.session import DEFAULT_CUSTOMER_ID, close_session, create_session, run_turn
from transport.tts import TTSBackend, get_tts_backend, speakable

# Flux's documented model for English; matches transport/tts.py's
# DEEPGRAM_TTS_URL sibling constants in spirit — the one place these
# concerns live.
FLUX_MODEL = "flux-general-en"
SAMPLE_RATE = 16000  # linear16 mono, Flux's plain (non-containerized) input format
CHANNELS = 1
# ~0.1s per mic chunk sent to Flux — small enough that end-of-turn detection
# feels responsive, big enough not to spam the socket.
BLOCK_SIZE = 1600


async def listen_and_transcribe(client: AsyncDeepgramClient) -> tuple[str, float]:
    """Open a Flux connection, stream mic audio in, return once Flux signals
    EndOfTurn. Returns (transcript, stt_latency_seconds) — latency covers
    the whole listen (mic capture is the dominant cost, same as a human
    transport would report "how long did listening take").
    """
    loop = asyncio.get_running_loop()
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

    def _on_audio(indata, _frames, _time_info, status) -> None:
        if status:
            print(f"[mic warning] {status}")
        loop.call_soon_threadsafe(audio_queue.put_nowait, bytes(indata))

    start = time.perf_counter()
    transcript = ""

    async with client.listen.v2.connect(
        model=FLUX_MODEL, encoding="linear16", sample_rate=SAMPLE_RATE
    ) as socket:

        async def _send_mic_audio() -> None:
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=BLOCK_SIZE,
                callback=_on_audio,
            ):
                while True:
                    chunk = await audio_queue.get()
                    await socket.send_media(chunk)

        sender_task = asyncio.create_task(_send_mic_audio())
        try:
            async for message in socket:
                msg_type = getattr(message, "type", None)
                if msg_type == "TurnInfo" and message.event == "EndOfTurn":
                    transcript = message.transcript
                    break
                if msg_type == "Error":
                    raise RuntimeError(f"Deepgram Flux error [{message.code}]: {message.description}")
        finally:
            sender_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender_task

    return transcript, time.perf_counter() - start


def _decode_wav(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes (as returned by transport/tts.py's backends) into a
    numpy array sounddevice can play. Stdlib `wave` only reads WAV/linear
    PCM — this is exactly why transport/tts.py explicitly requests that
    format instead of trusting each provider's default.
    """
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
    audio = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels)
    return audio, sample_rate


async def speak(text: str, backend: TTSBackend) -> float:
    """Synthesize `text` and play it back, blocking until playback finishes
    (Phase 7's loop is strictly sequential — no barge-in). Returns TTS
    latency in seconds, covering synthesis + playback.
    """
    start = time.perf_counter()
    # Strip markdown before synthesis — see transport/tts.py's speakable().
    # Applied here rather than at each call site so the greeting, the reply
    # and the escalation notice are all covered by one boundary.
    audio_bytes = await backend.synthesize(speakable(text))
    audio, sample_rate = _decode_wav(audio_bytes)
    sd.play(audio, samplerate=sample_rate)
    await asyncio.get_running_loop().run_in_executor(None, sd.wait)
    return time.perf_counter() - start


async def main() -> None:
    configure_logging()
    customer_id = input(f"Customer ID [{DEFAULT_CUSTOMER_ID}]: ").strip() or DEFAULT_CUSTOMER_ID
    session = create_session(customer_id, transport="voice_local")
    tts_backend = get_tts_backend()
    deepgram_client = AsyncDeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))

    print(f"\nVoice session started for {customer_id}. Speak naturally after each 'Listening...' "
          "prompt; say something like \"that's all, thanks\" to end the call. Ctrl+C to abort.\n")

    # Greet before the first listen, so the customer hears the line is live
    # rather than opening onto silence.
    print(f"Agent: {GREETING}")
    await speak(GREETING, tts_backend)

    while True:
        print("\U0001f3a4 Listening...")
        try:
            transcript, stt_latency = await listen_and_transcribe(deepgram_client)
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not transcript.strip():
            print("(heard nothing — try again)")
            continue
        print(f"You: {transcript}")

        outcome = await run_turn(session, transcript)
        print(f"Agent: {outcome.reply}")
        for warning in outcome.warnings:
            print(f"({warning})")

        tts_latency = await speak(outcome.reply, tts_backend)
        if outcome.notice:
            print(outcome.notice)
            tts_latency += await speak(outcome.notice, tts_backend)

        total = stt_latency + outcome.llm_latency_seconds + tts_latency
        print(
            f"[latency] STT: {stt_latency * 1000:.0f}ms | LLM: {outcome.llm_latency_seconds * 1000:.0f}ms "
            f"| TTS: {tts_latency * 1000:.0f}ms | Total: {total * 1000:.0f}ms"
        )

        if outcome.ended:
            break

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
