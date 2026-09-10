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

// Every press of the button bumps this. startCall() captures it before its
// first await and re-checks it after every one; if it has moved, the user
// pressed again while we were waiting on a permission prompt or a module
// load, and this attempt must abandon itself rather than finish setting up
// a call that has already been cancelled.
//
// A generation counter rather than a boolean because the question is not
// "is something cancelling?" but "is *this* attempt still the current one?"
// — a flag would itself race between the check and the act.
let generation = 0;

// A reason to display once the UI lands back on idle. Set by socket.onerror
// and by the server's text frames (see socket.onmessage), both of which are
// followed immediately by a close, and the close handler owns teardown.
let idleMessage = "";

// endCall() only *begins* hanging up (stop the mic, close the socket); the
// socket's own onclose is what actually settles the UI back to idle, so the
// "ending" state is on screen for as long as the close handshake takes. A
// floor keeps it visible even when the socket closes instantly (e.g. it was
// never opened) — hanging up should read as deliberate, not as a flicker.
const MIN_ENDING_MS = 200;
let endingSince = 0;

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

// Give back whatever a half-built call managed to acquire. Kept separate
// because an abandoned attempt owns resources that are not (and must never
// become) the module-level ctx/stream — publishing them is precisely the
// step it is skipping.
function release(localStream, localCtx) {
  if (localStream) localStream.getTracks().forEach((track) => track.stop());
  if (localCtx) localCtx.close();
}

async function startCall() {
  const myGeneration = generation;
  // True the moment another press supersedes this attempt. Checked after
  // every await, because each one is a window in which the user can press
  // again — and a double-click is one natural gesture, not an exotic input.
  const superseded = () => generation !== myGeneration;

  idleMessage = "";
  setState("connecting", "Connecting…");

  // Held locally, not in the module-level ctx/stream, until the attempt is
  // committed. An attempt that publishes early is an attempt endCall() can
  // catch half-built: it would see a stream but no socket, stop the mic,
  // settle the UI to idle — and then this function would resume and open a
  // live socket anyway, putting the user in a call they just cancelled.
  let localStream = null;
  let localCtx = null;

  try {
    localStream = await navigator.mediaDevices.getUserMedia({
      // Browser-side echo cancellation, so the agent's own voice coming out
      // of the speakers is not fed back in as the caller interrupting — the
      // self-echo problem Phase 8 hit with a local microphone.
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
    });
  } catch (err) {
    // Nothing was acquired, so nothing to release — but the message still
    // belongs to whoever owns the UI now, and that may no longer be us.
    if (superseded()) return;
    setState("idle", "Microphone permission denied.");
    return;
  }
  if (superseded()) return release(localStream, localCtx);

  // Opening the context at the pipeline's rate means nothing resamples,
  // anywhere.
  localCtx = new AudioContext({ sampleRate: SAMPLE_RATE });

  // sampleRate is a HINT, not a guarantee. A browser that ignores it and
  // hands back 48 kHz breaks everything downstream by a factor of three:
  // Deepgram receives what sounds like nonsense, never fires end-of-turn,
  // and the agent simply never replies. That symptom points nowhere near
  // its cause, so fail loudly here instead of debugging silence later.
  if (localCtx.sampleRate !== SAMPLE_RATE) {
    const actual = localCtx.sampleRate;
    release(localStream, localCtx);
    if (superseded()) return;
    setState("idle", `Browser gave ${actual} Hz, not ${SAMPLE_RATE} Hz. Try Chrome.`);
    return;
  }

  try {
    await localCtx.audioWorklet.addModule("/static/capture-worklet.js");
  } catch (err) {
    release(localStream, localCtx);
    if (superseded()) return;
    setState("idle", "Audio capture failed to load.");
    return;
  }
  if (superseded()) return release(localStream, localCtx);

  // Committed. Everything from here to the end of this function is
  // synchronous, so there is no further window in which a press can find
  // the call half-built: any press from now on runs endCall() against a
  // fully published ctx/stream/socket.
  stream = localStream;
  ctx = localCtx;
  const ws = new WebSocket(`ws://${location.host}/browser-stream`);
  ws.binaryType = "arraybuffer";
  socket = ws;

  // Handlers close over `ws` and compare it against the current `socket`
  // rather than reading the module-level one. An event from a socket the UI
  // has already moved past must not steer a newer call.
  const isCurrent = () => socket === ws;

  ws.onmessage = (event) => {
    if (!isCurrent()) return;
    // The server sends audio as binary and problems as text — a text frame
    // is it explaining why this call cannot happen (a missing
    // DEEPGRAM_API_KEY, say) just before closing. Show that on the way to
    // idle instead of a blank status line and a button that looks dead.
    if (typeof event.data === "string") {
      idleMessage = event.data;
      return;
    }
    if (!ctx) return;
    playChunk(event.data);
  };

  // If we're already in the "ending" state, this is the close we asked for
  // via endCall() — settle to idle. Otherwise the server or the network
  // closed the socket out from under us (mid-turn drop, server crash) while
  // we were still "connecting" or "in-call", so run the same hang-up
  // sequence a manual press would. Routing both cases through endCall()
  // (which no-ops once already "ending") keeps there being exactly one
  // teardown path instead of two.
  ws.onclose = () => {
    if (!isCurrent()) return;
    if (button.dataset.state === "ending") settleIdle();
    else endCall();
  };

  // onerror is ALWAYS followed by onclose, and onclose is the single
  // teardown path. Setting the state to "idle" here would make that close
  // handler call endCall(), whose first line returns immediately on "idle"
  // — so the microphone would keep recording and the AudioContext would
  // never close, one orphan per retry until AudioContext construction
  // itself starts throwing. Record the reason; let the close tear down.
  ws.onerror = () => {
    if (!isCurrent()) return;
    idleMessage = "Connection failed.";
  };

  ws.onopen = () => {
    if (!isCurrent()) {
      ws.close();
      return;
    }
    const source = ctx.createMediaStreamSource(stream);
    const capture = new AudioWorkletNode(ctx, "capture-processor");
    capture.port.onmessage = ({ data }) => {
      window.audioLevel.mic = data.level;
      if (isCurrent() && ws.readyState === WebSocket.OPEN) ws.send(data.pcm.buffer);
    };
    source.connect(capture);
    playHead = ctx.currentTime;
    setState("in-call", "");
  };
}

function settleIdle() {
  // Detach the dying call's resources immediately and hold the context in a
  // local. The delay below is a UI floor, and a timer that reached back for
  // the module-level `ctx` could find a *newer* call's context there and
  // close it — the exact class of bug this whole file is being fixed for.
  const dyingCtx = ctx;
  socket = null;
  stream = null;
  ctx = null;

  // Never resolve faster than MIN_ENDING_MS after "ending" was first shown,
  // so a socket that closes instantly (or was never open at all) still
  // leaves the state on screen long enough for the CSS to actually paint it.
  const elapsed = performance.now() - endingSince;
  setTimeout(() => {
    if (dyingCtx) dyingCtx.close();
    // No generation check is needed here: the button stays on "ending" for
    // the whole floor and presses in "ending" are ignored, so nothing new
    // can have started by the time this fires.
    window.audioLevel = { mic: 0, agent: 0 };
    setState("idle", idleMessage);
    idleMessage = "";
  }, Math.max(0, MIN_ENDING_MS - elapsed));
}

function endCall() {
  // Idempotent: a second hang-up press, or the socket's own onclose firing
  // after we've already begun ending, must not restart the sequence.
  if (button.dataset.state === "idle" || button.dataset.state === "ending") return;
  endingSince = performance.now();
  setState("ending", "");
  if (stream) stream.getTracks().forEach((track) => track.stop());
  // CONNECTING counts as well as OPEN. A socket closed mid-handshake is
  // still a socket the server may be about to accept: dropping the
  // reference without close() leaves run_call() never returning,
  // close_session() never running, and no summary ticket written — a leaked
  // session behind a button that reads "idle".
  if (socket && (socket.readyState === WebSocket.CONNECTING || socket.readyState === WebSocket.OPEN)) {
    // socket.close() completes asynchronously; onclose (above) calls
    // settleIdle() once it fires. Nothing else to do here.
    socket.close();
  } else {
    // No socket, or it's already closed/never opened (e.g. hang-up pressed
    // mid-"connecting") — nothing will ever call onclose, so settle directly.
    settleIdle();
  }
}

button.addEventListener("click", () => {
  const state = button.dataset.state;
  // A press while already hanging up has nothing to supersede — endCall()
  // would return immediately anyway — and must not bump the generation,
  // which is what keeps the "ending" floor from being invalidated by the
  // very presses it is there to absorb.
  if (state === "ending") return;
  // Otherwise: every press supersedes whatever the last one set in motion,
  // including a startCall() still suspended on one of its awaits, which
  // resumes into a generation that no longer matches and releases the
  // microphone it had acquired instead of finishing a cancelled call.
  generation++;
  if (state === "idle") startCall();
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

// window.audioLevel.mic is refreshed every 20 ms by the capture worklet,
// which posts whether or not anyone is speaking, so it falls to zero on its
// own. window.audioLevel.agent has no such heartbeat: it is written only
// when a chunk of agent audio arrives, so the instant the agent stops
// talking it freezes at the last chunk's RMS and stays there. Since the
// greeting plays at call start, that would leave *every* ring for the rest
// of the call tinted by an agent who fell silent seconds ago, and the spawn
// interval pinned fast through the caller's silence.
//
// Halving roughly every AGENT_DECAY_HALF_LIFE_MS gives it the equivalent
// heartbeat. It is long enough not to fight a live stream (Pipecat's output
// transport paces TTS audio out in ~20 ms chunks, and each arrival
// overwrites the level upward) and short enough that a real pause reads as
// silence within a few frames.
const AGENT_DECAY_HALF_LIFE_MS = 150;

// Under reduced motion the rings never travel, so nothing ever filters them
// out and the "spawn when the field is empty" condition can only ever fire
// once — the field froze at whatever colour it was born with, which for a
// page loaded before the call is always silence. Rebuild the static field
// on a slow cadence instead. Each ring is still coloured exactly once at
// spawn and never re-tinted; only the field as a whole is replaced, and at
// a rate slow enough to read as a colour change rather than as motion.
const REDUCED_MOTION_REBUILD_MS = 400;

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

  // See AGENT_DECAY_HALF_LIFE_MS. Applied before anything reads the levels,
  // and outside the reduced-motion branch, because both branches colour by
  // them.
  window.audioLevel.agent *= Math.pow(0.5, (dt * 1000) / AGENT_DECAY_HALF_LIFE_MS);

  if (reduceMotion) {
    // Hold the field still rather than remove it: no spawning, no travel,
    // just a redraw so colour still answers to who is speaking.
    if (rings.length === 0 || now - lastSpawn > REDUCED_MOTION_REBUILD_MS) {
      lastSpawn = now;
      rings = [];
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
