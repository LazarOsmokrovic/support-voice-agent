// One button, one call. Press to talk to the agent, press again to hang up,
// press again for a conversation that remembers nothing of the last one —
// the server creates a new session per socket, so a fresh press is a fresh
// caller.
const SAMPLE_RATE = 16000;

// How far ahead of the audio clock each chunk is scheduled. See playChunk.
const PLAYBACK_LEAD_SECONDS = 0.1;

const button = document.getElementById("call");
const label = document.getElementById("label");
const status = document.getElementById("status");

// Read by the animation (style.css / Task 5). Kept on window rather than
// passed around because the animation loop runs independently of call state.
window.audioLevel = { mic: 0, agent: 0 };

let ctx = null;
let socket = null;
let stream = null;
let playHead = 0;

function setState(state, message = "") {
  button.dataset.state = state;
  label.textContent = state === "in-call" ? "Hang up" : "Call";
  status.textContent = message;
}

function playChunk(bytes) {
  // Int16 back to Float32, scheduled end-to-end so consecutive chunks play
  // as continuous speech rather than overlapping or gapping.
  const pcm = new Int16Array(bytes);
  const buffer = ctx.createBuffer(1, pcm.length, SAMPLE_RATE);
  const channel = buffer.getChannelData(0);
  let sum = 0;
  for (let i = 0; i < pcm.length; i++) {
    channel[i] = pcm[i] / 0x8000;
    sum += channel[i] * channel[i];
  }
  window.audioLevel.agent = Math.sqrt(sum / pcm.length);

  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  // Schedule a jitter buffer ahead of the clock rather than at it. Starting
  // a chunk at exactly ctx.currentTime means any scheduling jitter lands it
  // late, and a late chunk is an audible gap mid-word. A ~100 ms lead costs
  // a tenth of a second of latency nobody notices and removes the most
  // likely source of stutter.
  playHead = Math.max(playHead, ctx.currentTime + PLAYBACK_LEAD_SECONDS);
  source.start(playHead);
  playHead += buffer.duration;
}

async function startCall() {
  setState("connecting", "Connecting…");
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      // Browser-side echo cancellation, so the agent's own voice coming out
      // of the speakers is not fed back in as the caller interrupting — the
      // self-echo problem Phase 8 hit with a local microphone.
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
    });
  } catch (err) {
    setState("idle", "Microphone permission denied.");
    return;
  }

  // Opening the context at the pipeline's rate means nothing resamples,
  // anywhere.
  ctx = new AudioContext({ sampleRate: SAMPLE_RATE });

  // sampleRate is a HINT, not a guarantee. A browser that ignores it and
  // hands back 48 kHz breaks everything downstream by a factor of three:
  // Deepgram receives what sounds like nonsense, never fires end-of-turn,
  // and the agent simply never replies. That symptom points nowhere near
  // its cause, so fail loudly here instead of debugging silence later.
  if (ctx.sampleRate !== SAMPLE_RATE) {
    setState("idle", `Browser gave ${ctx.sampleRate} Hz, not ${SAMPLE_RATE} Hz. Try Chrome.`);
    await ctx.close();
    stream.getTracks().forEach((track) => track.stop());
    ctx = null;
    stream = null;
    return;
  }

  await ctx.audioWorklet.addModule("/static/capture-worklet.js");

  socket = new WebSocket(`ws://${location.host}/browser-stream`);
  socket.binaryType = "arraybuffer";

  socket.onmessage = (event) => playChunk(event.data);
  socket.onclose = () => endCall();
  socket.onerror = () => setState("idle", "Connection failed.");

  socket.onopen = () => {
    const source = ctx.createMediaStreamSource(stream);
    const capture = new AudioWorkletNode(ctx, "capture-processor");
    capture.port.onmessage = ({ data }) => {
      window.audioLevel.mic = data.level;
      if (socket && socket.readyState === WebSocket.OPEN) socket.send(data.pcm.buffer);
    };
    source.connect(capture);
    playHead = ctx.currentTime;
    setState("in-call", "");
  };
}

function endCall() {
  if (button.dataset.state === "idle") return;
  setState("ending", "");
  if (socket && socket.readyState === WebSocket.OPEN) socket.close();
  if (stream) stream.getTracks().forEach((track) => track.stop());
  if (ctx) ctx.close();
  socket = null;
  stream = null;
  ctx = null;
  window.audioLevel = { mic: 0, agent: 0 };
  setState("idle", "");
}

button.addEventListener("click", () => {
  if (button.dataset.state === "idle") startCall();
  else endCall();
});

// ---------------------------------------------------------------------
// Ripple animation (Task 5)
//
// Ripples are a visible history of the last few seconds of speech, not a
// single reactive pulse. Each ring is coloured exactly once, at the instant
// it spawns, from whichever of mic/agent was louder right then, and that
// colour never changes for the rest of the ring's life — only its opacity
// fades as it travels outward. That is why a loud syllable is still
// visible as a bright ring several seconds later, trailing behind whatever
// is being said now.
// ---------------------------------------------------------------------

const canvas = document.getElementById("ripples");
const canvasCtx = canvas.getContext("2d");

// The two ends of the colour scale: a cool, human cyan for the caller and a
// warm amber for the agent. A ring's hue is interpolated continuously
// between them by how much each side contributed at spawn time, so two
// people talking over each other reads as an in-between hue rather than a
// jarring swap.
const CALLER_HUE = 190;
const AGENT_HUE = 28;
const SATURATION = 78;

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

let rings = [];
let lastSpawn = 0;
let lastFrameTime = performance.now();
let canvasDpr = 1;

function resizeCanvas() {
  const parent = canvas.parentElement.getBoundingClientRect();
  canvasDpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(parent.width * canvasDpr);
  canvas.height = Math.round(parent.height * canvasDpr);
  canvas.style.width = `${parent.width}px`;
  canvas.style.height = `${parent.height}px`;
  canvasCtx.setTransform(canvasDpr, 0, 0, canvasDpr, 0, 0);
}
window.addEventListener("resize", resizeCanvas);
resizeCanvas();

// Read fresh every frame (cheap) rather than cached, so a state-change
// transform on the button (e.g. its connecting-state scale) never leaves
// rings spawning from a stale centre.
function layout() {
  const buttonRect = button.getBoundingClientRect();
  const parentRect = canvas.parentElement.getBoundingClientRect();
  return {
    cx: buttonRect.left - parentRect.left + buttonRect.width / 2,
    cy: buttonRect.top - parentRect.top + buttonRect.height / 2,
    r: buttonRect.width / 2,
    w: parentRect.width,
    h: parentRect.height,
  };
}

// Where on the caller–agent scale a ring is born, and how loud it was.
function levelsToColor(mic, agent) {
  const total = mic + agent;
  const t = total > 0 ? agent / total : 0.5;
  const hue = CALLER_HUE + (AGENT_HUE - CALLER_HUE) * t;
  const level = Math.min(1, Math.max(mic, agent));
  const lightness = 44 + level * 26;
  const alpha = 0.3 + level * 0.55;
  return { hue, lightness, alpha, level };
}

function spawnRing(now, geo) {
  const { mic, agent } = window.audioLevel;
  const { hue, lightness, alpha, level } = levelsToColor(mic, agent);
  rings.push({
    radius: geo.r,
    startRadius: geo.r,
    // Fixed for life: a Float-free CSS colour string baked in at birth.
    color: `hsla(${hue.toFixed(1)}, ${SATURATION}%, ${lightness.toFixed(1)}%, ${alpha.toFixed(2)})`,
    speed: 34 + level * 100, // px/s of travel — louder speech outruns quieter speech
    maxRadius: geo.r + 130 + level * 260,
    lineWidth: 1.5 + level * 2.5,
  });
}

function spawnIntervalMs() {
  const level = Math.max(window.audioLevel.mic, window.audioLevel.agent);
  // ~900ms between rings at silence, down to ~150ms at peak loudness.
  return 900 - level * 750;
}

function drawRings(geo, alphaScale = 1) {
  canvasCtx.clearRect(0, 0, geo.w, geo.h);
  rings = rings.filter((ring) => ring.radius <= ring.maxRadius);
  for (const ring of rings) {
    const progress = (ring.radius - ring.startRadius) / (ring.maxRadius - ring.startRadius || 1);
    canvasCtx.beginPath();
    canvasCtx.arc(geo.cx, geo.cy, ring.radius, 0, Math.PI * 2);
    canvasCtx.strokeStyle = ring.color;
    canvasCtx.globalAlpha = Math.max(0, 1 - progress) * alphaScale;
    canvasCtx.lineWidth = ring.lineWidth;
    canvasCtx.stroke();
  }
  canvasCtx.globalAlpha = 1;
}

function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.05, (now - lastFrameTime) / 1000);
  lastFrameTime = now;
  const geo = layout();

  if (reduceMotion) {
    // Hold the field still rather than remove it: no spawning, no travel,
    // just a redraw so colour still answers to who is speaking.
    if (rings.length === 0) {
      const { mic, agent } = window.audioLevel;
      const { hue, lightness, alpha } = levelsToColor(mic, agent);
      for (let i = 0; i < 3; i++) {
        rings.push({
          radius: geo.r + 40 + i * 36,
          startRadius: geo.r,
          maxRadius: geo.r + 500,
          color: `hsla(${hue.toFixed(1)}, ${SATURATION}%, ${lightness.toFixed(1)}%, ${(alpha * (1 - i * 0.25)).toFixed(2)})`,
          speed: 0,
          lineWidth: 2,
        });
      }
    }
    drawRings(geo);
    return;
  }

  if (button.dataset.state === "in-call" && now - lastSpawn > spawnIntervalMs()) {
    spawnRing(now, geo);
    lastSpawn = now;
  }

  for (const ring of rings) ring.radius += ring.speed * dt;
  drawRings(geo);
}

requestAnimationFrame(frame);
