"""Real-time streaming voice loop over the local mic — Phase 8 interface
(Pipecat).

Real-time barge-in (stop the bot the instant the caller starts talking) and
partial transcripts, over sounddevice/PyAudio instead of transport/
voice_local.py's strictly sequential loop. The actual Pipecat wiring —
ClaudeTurnProcessor, LatencyLogger, the TTS backend switch, and
build_pipeline() — lives in transport/pipecat_processors.py, shared with
transport/telephony.py (Phase 9) since both transports assemble the exact
same pipeline shape and differ only in which `transport` object feeds it.
See that module's docstring for why the split exists.

Run with: python -m transport.pipeline
"""

from __future__ import annotations

import asyncio

from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)
from pipecat.workers.runner import WorkerRunner

from agent.core import configure_logging
from agent.session import DEFAULT_CUSTOMER_ID, close_session, create_session
from transport.pipecat_processors import build_pipeline


async def main() -> None:
    configure_logging()
    customer_id = input(f"Customer ID [{DEFAULT_CUSTOMER_ID}]: ").strip() or DEFAULT_CUSTOMER_ID
    session = create_session(customer_id, transport="pipeline")

    transport = LocalAudioTransport(LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True))
    # mute_mic_during_tts=True: local mic/speaker have no acoustic echo
    # cancellation, so without headphones the mic picks up the bot's own
    # voice and Flux reads it as the caller barging in (PROGRESS.md's Phase 8
    # note). Muting the STT service while the bot talks stops that at the
    # cost of real barge-in during that window too — see MicMuteGate's
    # docstring in transport/pipecat_processors.py for the full tradeoff.
    pipeline = build_pipeline(transport, session, mute_mic_during_tts=True)

    worker = PipelineWorker(pipeline, params=PipelineParams(enable_metrics=True))
    runner = WorkerRunner()
    await runner.add_workers(worker)

    print(
        f"\nVoice session started for {customer_id} (real-time). Speak naturally — "
        "note the mic is muted while the bot is talking (mute_mic_during_tts), so wait "
        "for it to finish before replying. Ctrl+C to abort.\n"
    )

    await runner.run()

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
