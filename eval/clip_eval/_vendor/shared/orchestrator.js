// orchestrator.js
// Central orchestrator with built-in signaling for WebRTC

// Stamp every log line with an ISO timestamp so downstream tools can use
// the orchestrator log as a deterministic clock (e.g., sink(A)#1 marks
// the moment A.wav recording began).
const _origLog = console.log.bind(console);
const _origErr = console.error.bind(console);
console.log = (...a) => _origLog(`[${new Date().toISOString()}]`, ...a);
console.error = (...a) => _origErr(`[${new Date().toISOString()}]`, ...a);

console.log('[Orch] Loading modules...');

const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');
const express = require('express');
const http = require('http');
const WebSocket = require('ws');
console.log('[Orch] Loading wrtc...');
const wrtc = require('wrtc');
console.log('[Orch] wrtc loaded successfully');
const { RTCPeerConnection, RTCSessionDescription } = wrtc;
const { RTCAudioSink, RTCAudioSource } = wrtc.nonstandard;
const wav = require('wav');
console.log('[Orch] All modules loaded');

// Global error handlers to catch crashes
process.on('uncaughtException', (err) => {
  console.error('[Orch] UNCAUGHT EXCEPTION:', err);
});
process.on('unhandledRejection', (reason, promise) => {
  console.error('[Orch] UNHANDLED REJECTION:', reason);
});

// Config - uses environment variables with defaults from .env.example
const env = process.env;
const SIGNAL_PORT = parseInt(env.SIGNAL_PORT, 10) || 9100;
const RECORD_DIR = env.RECORD_DIR || 'outputs';
const WIRE_SAMPLE_RATE = parseInt(env.WIRE_SAMPLE_RATE, 10) || 48000;

// State
let offers = { A: null, B: null };
let candidates = { A: [], B: [] };
let pcs = { A: null, B: null };
let sinks = { A: null, B: null };
let sources = { A: null, B: null };
let writers = { A: null, B: null };
let ready = { A: false, B: false };
let forwardBuffers = { A: Buffer.alloc(0), B: Buffer.alloc(0) };

// ───────────────────────────────────────────────────────────────────────
// Speech-onset/offset detection on incoming WebRTC audio, per role.
//
// PURPOSE: log a wall-clock timestamp whenever a role's audio energy
// rises above the speech threshold. Useful as DIAGNOSTIC telemetry to
// see when each side started/stopped speaking AT THE ORCHESTRATOR, on
// the same clock the wav writer uses.
//
// PROTOCOL NOTE: WebRTC audio is continuous. There's no "message
// started" marker in the audio stream — just a steady flow of frames
// (silence or speech) at the sender's clock rate. We infer "message
// started" purely from rising audio energy. The data channel (used by
// OpenAI's adapter to deliver `output_audio_buffer.started` events) is
// a separate, fast control-plane channel; those events arrive ahead of
// the audio bytes (different path, different latency).
//
// PARAMETERS:
//   - 10ms RMS buckets (= 480 samples at 48 kHz, half-frame at 20ms)
//   - -45 dBFS threshold (typical TTS speech is well above this)
//   - 3 consecutive above-threshold buckets → declare onset (30ms of
//     speech; protects against single-frame spikes)
//   - 20 consecutive below-threshold buckets → declare offset (200ms
//     of quiet; matches typical pause-between-utterances)
//
// WHAT THIS DOES NOT TELL YOU:
//   - WHEN OpenAI (or any sender) decided to start emitting a response.
//     The data-channel event in the adapter's log is the right source
//     for that — it's instantaneous control-plane signalling.
//   - The WAV-TIMELINE POSITION at which the audio appears. The
//     onset's wall clock is when the sink callback FIRES; the wav
//     writer adds its own pipeline latency (typically 200-600ms) so
//     `audio_onset wall_clock − anchor` is NOT the right wav position
//     for that audio. Don't use it as dashboard timing source — use
//     offline B.wav RMS scan instead. See visualization/build_viz_data.py
//     header for full timing model.
//   - The TRUE WALL of when audio physically reached the orchestrator,
//     because chunks may queue at the WebRTC receive layer before the
//     sink callback fires.
//
// USE CASES:
//   - Cross-check: agent's `output_audio_buffer.started` should
//     correlate with `audio_onset role=B` (modulo the data-channel-vs-
//     audio-track latency gap, which is the key drift signal).
//   - Quick visual scan of "how chatty was each side at what time."
//   - Debugging: if audio_onset never fires for a role, something's
//     broken upstream.
// ───────────────────────────────────────────────────────────────────────
const ONSET_RMS_THRESHOLD = Math.pow(10, -45 / 20) * 32768; // linear int16
const ONSET_BUCKET_MS = 10;
const ONSET_BUCKET_SAMPLES = Math.floor(WIRE_SAMPLE_RATE * ONSET_BUCKET_MS / 1000);
const ONSET_ON_BUCKETS = 3;   // 30ms of speech to declare onset
const ONSET_OFF_BUCKETS = 20; // 200ms of silence to declare offset
let audioActivity = {
  A: { speaking: false, bucketBuf: Buffer.alloc(0), aboveRun: 0, belowRun: 0 },
  B: { speaking: false, bucketBuf: Buffer.alloc(0), aboveRun: 0, belowRun: 0 },
};

function processAudioActivity(role, sampleBuf) {
  const st = audioActivity[role];
  // Accumulate raw bytes and process in fixed-size buckets.
  st.bucketBuf = Buffer.concat([st.bucketBuf, sampleBuf]);
  const bucketBytes = ONSET_BUCKET_SAMPLES * 2; // 16-bit
  while (st.bucketBuf.length >= bucketBytes) {
    const bucket = st.bucketBuf.subarray(0, bucketBytes);
    st.bucketBuf = st.bucketBuf.subarray(bucketBytes);
    let sumSq = 0;
    for (let i = 0; i < bucket.length; i += 2) {
      const v = bucket.readInt16LE(i);
      sumSq += v * v;
    }
    const rms = Math.sqrt(sumSq / ONSET_BUCKET_SAMPLES);
    const above = rms > ONSET_RMS_THRESHOLD;
    if (above) {
      st.aboveRun += 1; st.belowRun = 0;
      if (!st.speaking && st.aboveRun >= ONSET_ON_BUCKETS) {
        st.speaking = true;
        console.log(`[Orch] audio_onset role=${role} rms=${Math.round(rms)}`);
      }
    } else {
      st.belowRun += 1; st.aboveRun = 0;
      if (st.speaking && st.belowRun >= ONSET_OFF_BUCKETS) {
        st.speaking = false;
        console.log(`[Orch] audio_offset role=${role}`);
      }
    }
  }
}
// Debug/forwarding state
let firstLogsRemaining = { A: 12, B: 12 }; // log first few sink events per role
let framesOutPerSec = { A: 0, B: 0 };
let framesOutWindowStart = Date.now();
let backlogBytes = { A: 0, B: 0 };
let pacers = { A: null, B: null };

// Create output directory
const outDir = RECORD_DIR;
fs.mkdirSync(outDir, { recursive: true });

// WAV Writers helper - Record at WIRE_SAMPLE_RATE to match WebRTC
function createWriter(role) {
  const file = path.join(outDir, `${role}.wav`);
  return new wav.FileWriter(file, { sampleRate: WIRE_SAMPLE_RATE, channels: 1, bitDepth: 16 });
}

// Setup HTTP + WS signaling server
const app = express();
const server = http.createServer(app);

server.on('error', (err) => console.error('[Orch] HTTP server error:', err));

// Use noServer mode for both WSS instances to avoid conflicting upgrade handlers
const wss = new WebSocket.Server({ noServer: true });
wss.on('error', (err) => console.error('[Orch] WebSocket server error:', err));

// ── Audio tap endpoint — allows external processes to receive raw PCM audio ──
const tapClients = { A: [], B: [] };
const tapWss = new WebSocket.Server({ noServer: true });
tapWss.on('error', (err) => console.error('[Orch] Tap WebSocket server error:', err));

// Route upgrade requests to the correct WSS by path
server.on('upgrade', (req, socket, head) => {
  const pathname = new URL(req.url, 'ws://localhost').pathname;
  console.log(`[Orch] HTTP upgrade request: ${pathname}`);

  if (pathname === '/signal') {
    wss.handleUpgrade(req, socket, head, (ws) => {
      wss.emit('connection', ws, req);
    });
  } else if (pathname === '/tap') {
    tapWss.handleUpgrade(req, socket, head, (ws) => {
      tapWss.emit('connection', ws, req);
    });
  } else {
    socket.destroy();
  }
});

tapWss.on('connection', (ws) => {
  console.log('[Orch] Tap client connected');
  let subscribedRole = null;

  ws.on('message', (raw) => {
    try {
      const msg = JSON.parse(raw);
      if (msg.type === 'subscribe' && (msg.role === 'A' || msg.role === 'B')) {
        subscribedRole = msg.role;
        tapClients[subscribedRole].push(ws);
        console.log(`[Orch] Tap client subscribed to role ${subscribedRole}`);
      }
    } catch { }
  });

  ws.on('close', () => {
    if (subscribedRole) {
      tapClients[subscribedRole] = tapClients[subscribedRole].filter(c => c !== ws);
      console.log(`[Orch] Tap client disconnected from role ${subscribedRole}`);
    }
  });

  ws.on('error', () => {
    if (subscribedRole) {
      tapClients[subscribedRole] = tapClients[subscribedRole].filter(c => c !== ws);
    }
  });
});

wss.on('connection', ws => {
  console.log('[Orch] New WebSocket connection');

  ws.on('error', (err) => {
    console.error('[Orch] WebSocket client error:', err.message);
  });

  ws.on('close', (code, reason) => {
    console.log(`[Orch] WebSocket closed: code=${code} reason=${reason || 'none'}`);
  });

  let connectionsSetup = false;

  ws.on('message', async msg => {
    const data = JSON.parse(msg);
    const { role, type, sdp, candidate } = data;
    console.log(`[Orch] Received: role=${role} type=${type}`);

    if (type === 'offer') {
      offers[role] = sdp;
      if (offers.A && offers.B && !connectionsSetup) {
        connectionsSetup = true;
        setupPeerConnections();
      }
      return;
    }
    if (type === 'candidate') {
      if (pcs[role]) pcs[role].addIceCandidate(candidate);
      else candidates[role].push(candidate);
      return;
    }
  });

  ws.send(JSON.stringify({ type: 'ready' }));
});

// ── Reset all session state ──────────────────────────────────────────────────
function resetState() {
  console.log('[Orch] Resetting all session state...');

  // Stop pacers
  ['A', 'B'].forEach(role => {
    if (pacers[role]) { clearInterval(pacers[role]); pacers[role] = null; }
  });

  // Stop sinks
  ['A', 'B'].forEach(role => {
    if (sinks[role]) { try { sinks[role].stop(); } catch {} sinks[role] = null; }
  });

  // Close WAV writers
  ['A', 'B'].forEach(role => {
    if (writers[role]) { try { writers[role].end(); } catch {} writers[role] = null; }
  });

  // Close peer connections
  ['A', 'B'].forEach(role => {
    if (pcs[role]) { try { pcs[role].close(); } catch {} pcs[role] = null; }
  });

  // Clear all signaling and audio state
  offers = { A: null, B: null };
  candidates = { A: [], B: [] };
  sources = { A: null, B: null };
  ready = { A: false, B: false };
  forwardBuffers = { A: Buffer.alloc(0), B: Buffer.alloc(0) };
  firstLogsRemaining = { A: 12, B: 12 };
  framesOutPerSec = { A: 0, B: 0 };
  framesOutWindowStart = Date.now();
  backlogBytes = { A: 0, B: 0 };

  // Recreate output WAV directory
  fs.mkdirSync(outDir, { recursive: true });

  console.log('[Orch] State reset complete');
}

app.use(express.json());

app.post('/reset', (_req, res) => {
  resetState();
  res.json({ success: true });
});

server.listen(SIGNAL_PORT, () => console.log(`Orchestrator signaling on port ${SIGNAL_PORT}`));

// Create and configure PeerConnections
async function setupPeerConnections() {
  console.log('[Orch] Setting up peer connections...');
  ['A', 'B'].forEach(role => {
    console.log(`[Orch] Creating RTCPeerConnection for role ${role}`);
    const pc = new RTCPeerConnection();
    pcs[role] = pc;

    // Local source track for sending to adapter
    const source = new RTCAudioSource({ sampleRate: WIRE_SAMPLE_RATE, channelCount: 1 });
    sources[role] = source;
    pc.addTrack(source.createTrack());

    // Apply remote offer and ICE candidates
    console.log(`[Orch] Setting remote description for role ${role}`);
    pc.setRemoteDescription(new RTCSessionDescription({ type: 'offer', sdp: offers[role] }));
    candidates[role].forEach(c => pc.addIceCandidate(c));

    // Answer back
    pc.createAnswer().then(answer => {
      pc.setLocalDescription(answer);
      wss.clients.forEach(c => {
        if (c.readyState === WebSocket.OPEN) {
          c.send(JSON.stringify({ role, type: 'answer', sdp: answer.sdp }));
        }
      });
    });

    // Handle incoming audio. Two responsibilities per role:
    //   1. RECORD: write every incoming byte to <role>.wav. This wav
    //      file is the source of truth for "what audio happened at
    //      what time" from the orchestrator's perspective. Its
    //      timeline = cumulative_samples / sample_rate, growing one
    //      sample per arriving frame. NOT directly wall-clock-aligned:
    //      the first wav byte lands at wall_clock(sink#1) but the wav
    //      writer's stream pipeline adds ~0.8s of buffering startup
    //      before the byte is physically committed (see
    //      visualization/build_viz_data.py header for the full timing
    //      model and how it's calibrated out per run).
    //   2. FORWARD: queue the bytes into forwardBuffers[other] so the
    //      OTHER role's per-role pacer can emit them at a steady
    //      48 kHz rate. The orchestrator's per-role pacer is what
    //      gives each receiver a clean real-time stream regardless of
    //      sender-side jitter — but it's also the place where any
    //      sender-vs-orchestrator clock skew turns into buffer growth
    //      (which is exactly the issue the silence-drain fix in
    //      role_b/reference/gptRealtime_adapter.js addresses for B's
    //      inbound side).
    pc.ontrack = ({ track }) => {
      if (track.kind !== 'audio') return;
      const sink = new RTCAudioSink(track, { sampleRate: WIRE_SAMPLE_RATE, channelCount: 1, bitDepth: 16 });
      sinks[role] = sink;
      writers[role] = createWriter(role);

      let sinkEventCount = { A: 0, B: 0 };
      sink.ondata = ({ samples }) => {
        const viewBuf = Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength);
        const other = role === 'A' ? 'B' : 'A';
        sinkEventCount[role] += 1;
        if (firstLogsRemaining[role] > 0) {
          const mod960 = viewBuf.length % 960;
          const mod1920 = viewBuf.length % 1920;
          console.log(`[Orch] sink(${role})#${sinkEventCount[role]} bytesIn=${viewBuf.length} mod960=${mod960} mod1920=${mod1920}`);
          firstLogsRemaining[role] -= 1;
        }
        // Buffer audio for the other side only; pacing happens via per-role scheduler
        forwardBuffers[other] = Buffer.concat([forwardBuffers[other], viewBuf]);
        backlogBytes[other] = forwardBuffers[other].length;

        // Also record raw incoming audio (what we received from this role)
        writers[role].write(viewBuf);

        // Detect speech onsets/offsets via inline RMS on incoming
        // bytes. Logs `[Orch] audio_onset role=A|B` with the
        // wall-clock ISO at which the rising-edge bucket completed.
        //
        // CAVEAT (timing): the wall clock when this fires is NOT the
        // wav-file position of the same audio. Bytes hit this callback
        // ~150-600ms before they land at a particular wav position
        // (the wav writer's write-stream + WriteStream high-water-mark
        // buffer absorbs them in chunks). So these logs are good for
        // "when did the orchestrator start receiving audio" but bad
        // for "what wav position does that audio occupy" — for the
        // latter use offline RMS scan of the saved wav file.
        processAudioActivity(role, viewBuf);

        // Forward to tap clients subscribed to this role
        for (const tapWs of tapClients[role]) {
          if (tapWs.readyState === WebSocket.OPEN) tapWs.send(viewBuf);
        }

        // *** Per-second frames-out stats ***
        // const now = Date.now();
        // if (now - framesOutWindowStart >= 1000) {
        //   console.log(`[Orch] framesOut/s A=${framesOutPerSec.A} B=${framesOutPerSec.B} | backlogBytes A=${backlogBytes.A} B=${backlogBytes.B}`);
        //   framesOutPerSec = { A: 0, B: 0 };
        //   framesOutWindowStart = now;
        // }
      };

      ready[role] = true;
      checkBothReady();

      // ── PER-ROLE 100ms FORWARDING PACER ──────────────────────────
      // Drains forwardBuffers[target] at exactly 48 kHz local-clock
      // rate (10 × 10 ms frames per setInterval(100) tick). This is
      // what gives the destination adapter a clean real-time stream
      // regardless of how bursty the source's incoming audio is.
      //
      // INTERACTION WITH CLOCK SKEW:
      //   The pacer drains at LOCAL clock rate. The SOURCE
      //   (`forwardBuffers[target]`) fills at the inbound adapter's
      //   sender clock rate. If sender > 48 kHz (e.g. OpenAI's TTS
      //   streaming for B's outbound), forwardBuffers grows; the
      //   pacer keeps emitting at 48 kHz but the audio CONTENT it's
      //   emitting is increasingly stale. If sender < 48 kHz the
      //   buffer empties and the else-branch below pads silence.
      //
      //   For A: the source is A's adapter's downlinkPacer, which is
      //   itself driven by setInterval(100) on the SAME host clock —
      //   no skew, forwardBuffers stays roughly empty/full at a
      //   bounded depth.
      //
      //   For B: the source is B's adapter's downlinkPacer, which
      //   ALSO drains its own downlinkBuf at local-clock 48 kHz —
      //   BUT B's downlinkBuf is itself fed from OpenAI's audio
      //   track (faster sender clock). Without the silence-drain in
      //   gptRealtime_adapter.js, B's downlinkBuf grew over time, so
      //   what arrived at this pacer was already delayed audio.
      //   With the fix, B's downlinkBuf stays bounded and this
      //   pacer's forwardBuffers also stays bounded.
      //
      // The pacer ALSO writes to wav: the writer is fed earlier in
      // sink.ondata (above), not from this pacer. So wav files
      // accumulate at the SOURCE rate, while forwardBuffers absorbs
      // any rate mismatch. wav timeline ≠ forwarded-audio timeline —
      // both are useful for different views of the conversation.
      const target = role === 'A' ? 'B' : 'A';
      if (!pacers[target]) {
        console.log(`[Orch] Starting pacer for role ${target}`);
        pacers[target] = setInterval(() => {
          const src = sources[target];
          if (!src) return;
          const FRAME_10MS = Math.floor(WIRE_SAMPLE_RATE * 0.01) * 2; // bytes, 10 ms @ WIRE_SAMPLE_RATE mono PCM16
          // Emit ten 10ms frames per 100ms tick
          for (let i = 0; i < 10; i += 1) {
            let out;
            const bufNow = forwardBuffers[target];
            if (bufNow.length >= FRAME_10MS) {
              out = bufNow.slice(0, FRAME_10MS);
              forwardBuffers[target] = bufNow.slice(FRAME_10MS);
            } else if (bufNow.length > 0) {
              out = Buffer.alloc(FRAME_10MS);
              bufNow.copy(out, 0, 0, bufNow.length);
              forwardBuffers[target] = Buffer.alloc(0);
            } else {
              out = Buffer.alloc(FRAME_10MS);
            }
            // Exact 960-byte ArrayBuffer for wrtc
            const ab = new ArrayBuffer(FRAME_10MS);
            new Uint8Array(ab).set(new Uint8Array(out.buffer, out.byteOffset, FRAME_10MS));
            const samples = new Int16Array(ab);
            src.onData({ samples, sampleRate: WIRE_SAMPLE_RATE, bitsPerSample: 16, channelCount: 1 });
            framesOutPerSec[target] += 1;
          }
        }, 100);
      }
    };

    pcs[role].onicecandidate = ({ candidate }) => {
      if (!candidate) return;
      const init = { candidate: candidate.candidate, sdpMid: candidate.sdpMid, sdpMLineIndex: candidate.sdpMLineIndex };
      wss.clients.forEach(c => {
        if (c.readyState === WebSocket.OPEN) (
          c.send(JSON.stringify({ role, type: 'candidate', candidate: init }))
        );
      });
    };
  });

  // No global silence injector; pacers enforce real-time cadence
}

function checkBothReady() {
  if (ready.A && ready.B) console.log('Both peers ready—streaming and recording.');
}

// Handle shutdown and mix
async function gracefulShutdown(signame) {
  console.error(`[Orch] ${signame} caught; outDir =`, outDir);
  console.log('\nShutting down gracefully...');

  // Stop sinks
  ['A', 'B'].forEach(r => sinks[r] && sinks[r].stop());

  // Close WAV writers with a hard timeout — when the WebRTC track ends
  // without a final flush event, writer.end()'s callback can hang
  // indefinitely and the parent runner SIGKILLs us before ffmpeg has a
  // chance to mix. The wav data is already on disk; the only thing the
  // .end() callback finalizes is the RIFF header length field, which
  // ffmpeg recovers from anyway (reads until EOF). Race against a 3s
  // ceiling so the mixdown still happens.
  const closePromises = ['A', 'B'].map(role => new Promise(res => {
    if (writers[role]) writers[role].end(() => res());
    else res();
  }));
  await Promise.race([
    Promise.all(closePromises),
    new Promise(res => setTimeout(() => {
      console.error('[Orch] writer close timed out after 3s — proceeding with mix');
      res();
    }, 3000)),
  ]);

  // Close peer connections
  ['A', 'B'].forEach(r => { if (pcs[r]) pcs[r].close(); });

  // Stop pacers
  ['A', 'B'].forEach(role => { if (pacers[role]) clearInterval(pacers[role]); });

  // Mixdown - create stereo WAV with A on left channel, B on right channel
  const a = path.join(outDir, 'A.wav');
  const b = path.join(outDir, 'B.wav');
  const combined = path.join(outDir, 'combined.wav');

  console.error('[Orch] A.wav exists?', fs.existsSync(a));
  console.error('[Orch] B.wav exists?', fs.existsSync(b));

  if (fs.existsSync(a) && fs.existsSync(b)) {
    // Use amerge to create stereo output: A.wav -> left channel, B.wav -> right channel
    // -ac 2 ensures stereo output, duration=longest pads the shorter track with silence
    const cmd = `ffmpeg -y -i ${a} -i ${b} -filter_complex "[0:a][1:a]amerge=inputs=2,pan=stereo|c0<c0|c1<c1[aout]" -map "[aout]" -ac 2 ${combined}`;
    exec(cmd, (err, stdout, stderr) => {
      console.error('[Orch] ffmpeg stdout:', stdout);
      console.error('[Orch] ffmpeg stderr:', stderr);
      if (err) console.error('Error creating combined stereo mix:', err);
      else console.log('Combined stereo mix created (A=left, B=right):', combined);
      process.exit();
    });
  } else {
    console.error('Warning: Audio files not found, skipping mix creation');
    process.exit();
  }
}

process.on('SIGINT', () => gracefulShutdown('SIGINT'));
// Also handle SIGTERM so a parent runner's polite-then-hard kill ladder
// still produces combined.wav before we exit.
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
