// Runs on the audio thread. Converts Float32 samples to the Int16 PCM the
// pipeline expects and posts them to the main thread, which owns the socket.
// An AudioWorklet rather than the deprecated ScriptProcessorNode: capture
// must not compete with the animation for the main thread, or the ripples
// stutter exactly when someone is speaking.
//
// Audio is BATCHED to ~20 ms before being posted. An AudioWorklet's render
// quantum is 128 frames, which at 16 kHz is 8 ms — posting every quantum would
// send 125 WebSocket messages a second, each carrying 256 bytes of audio
// under a full frame's worth of overhead. Batching to 320 samples cuts that
// to 50 messages a second and matches the chunk size telephony transports
// use. Sending unbatched is a plausible cause of choppy audio on its own,
// so this is built in rather than discovered later.
const BATCH_SAMPLES = 320; // 20 ms at 16 kHz

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Int16Array(BATCH_SAMPLES);
    this._filled = 0;
    this._energy = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const samples = input[0];

    for (let i = 0; i < samples.length; i++) {
      const clamped = Math.max(-1, Math.min(1, samples[i]));
      this._buffer[this._filled++] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
      this._energy += clamped * clamped;

      if (this._filled === BATCH_SAMPLES) {
        // RMS travels with the audio so the animation never re-measures it.
        const level = Math.sqrt(this._energy / BATCH_SAMPLES);
        const batch = this._buffer.slice();
        this.port.postMessage({ pcm: batch, level }, [batch.buffer]);
        this._filled = 0;
        this._energy = 0;
      }
    }
    return true;
  }
}
registerProcessor("capture-processor", CaptureProcessor);
