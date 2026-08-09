#!/usr/bin/env node
/**
 * gpt4oRealtime_adapter.js (Role A/B)
 * Bridges your orchestrator (WebRTC @ 48 kHz PCM16, 50 ms cadence) <->
 * OpenAI Realtime (WebRTC) using an ephemeral session token from openai_token_server.js.
 *
 * Key behaviors
 *  - Reuses your orchestrator's WS signaling contract (/signal) and 48 kHz timing.
 *  - Creates a separate RTCPeerConnection to OpenAI; audio both directions via WebRTC tracks.
 *  - Uses a data channel to send `session.update` and (optionally) response control messages.
 */

require('dotenv').config();
const fs = require('fs');
const wrtc = require('wrtc');
const WebSocket = require('ws');
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');
const { RTCAudioSource, RTCAudioSink } = wrtc.nonstandard;

// Configuration from environment variables with defaults
const DEFAULT_MODEL = process.env.OPENAI_REALTIME_MODEL || 'gpt-realtime-2';
const DEFAULT_TOKEN_SERVER = process.env.OPENAI_TOKEN_SERVER || 'http://localhost:3002';
const DEFAULT_TOKEN_SERVER_PORT = parseInt(process.env.TOKEN_SERVER_PORT, 10) || 3002;
const WIRE_SR = parseInt(process.env.WIRE_SAMPLE_RATE, 10) || 48000;  // Orchestrator wire sample rate
const CHUNK_BYTES = Math.floor(WIRE_SR * 0.05) * 2;  // 50 ms @ WIRE_SR PCM16 mono

// Prefer global fetch (Node >=18). Fallback to node-fetch if needed.
const _fetch = (typeof fetch !== 'undefined') ? fetch : (async (...args) => (await import('node-fetch')).default(...args));

const argv = yargs(hideBin(process.argv))
  .scriptName('gpt4oRealtime_adapter')
  .usage('$0 -r <A|B> -s <ws://host:port/signal> [opts]')
  .option('role', { alias: 'r', type: 'string', choices: ['A', 'B'], demandOption: true })
  .option('signalUrl', { alias: 's', type: 'string', demandOption: true })
  .option('tokenServer', { type: 'string', default: DEFAULT_TOKEN_SERVER })
  .option('model', { type: 'string', default: DEFAULT_MODEL })
  .option('voice', { type: 'string', default: 'alloy' })
  .option('systemPrompt', { type: 'string', default: '' })
  .option('turnMode', { type: 'string', choices: ['server_vad', 'none'], default: 'server_vad' })
  .option('vadMode', { type: 'string', choices: ['slow', 'fast'], default: 'slow', describe: 'VAD timing mode: slow (default, more patient) or fast (quicker response)' })
  .option('logEvents', { type: 'boolean', default: true })
  .option('prefill', { type: 'string', default: '', describe: 'Initial user text to append as history' })
  .option('prefillFile', { type: 'string', default: '', describe: 'Path to JSON array of prior turns [{role, text}]' })
  .option('autostart', { type: 'boolean', default: true, describe: 'Create response automatically after prefill' })
  .option('speakFirst', { type: 'boolean', default: false, describe: 'Role B: allow the agent to speak first (greet) instead of canceling the initial response. Use this when the examiner is configured to wait for the agent to greet before playing the first clip.' })
  .option('enableTranscription', { type: 'boolean', default: false, describe: 'Enable input_audio_transcription on the OpenAI session' })
  .option('transcribeOnly', { type: 'boolean', default: false, describe: 'Transcription-only mode: connect to audio tap, no orchestrator' })
  .option('tapUrl', { type: 'string', default: '', describe: 'WebSocket URL for audio tap (used with --transcribeOnly)' })
  .help().argv;

const ROLE = argv.role;
const SIGNAL_URL = argv.signalUrl;
const TOKEN_SERVER_URL = argv.tokenServer;
const MODEL = argv.model;
const VOICE = argv.voice;
const SYSTEM_PROMPT = argv.systemPrompt || '';
const TURN_MODE = argv.turnMode;   // 'server_vad' or 'none' (push-to-talk)
const VAD_MODE = argv.vadMode;     // 'slow' (default) or 'fast' (stricter threshold, shorter silence)
const PREFILL_TEXT = argv.prefill || '';
const PREFILL_FILE = argv.prefillFile || '';

const ENABLE_TRANSCRIPTION = argv.enableTranscription;
const TRANSCRIBE_ONLY = argv.transcribeOnly;
const TAP_URL = argv.tapUrl || '';

// Default to autostart for Role A (Examiner), disable for Role B (Examinee) to avoid simultaneous talking
const AUTOSTART = argv.autostart !== undefined ? argv.autostart : (ROLE === 'A');

function ts() { return new Date().toISOString(); }

// ------------- Globals -------------
let orchPC, orchSource, orchSink;             // to Orchestrator
let aiPC, aiSource, aiSink, aiDC;             // to OpenAI Realtime
let ws;                                       // WS signaling to orchestrator

let uplinkBuf = Buffer.alloc(0);              // Orchestrator -> OpenAI bytes
let downlinkBuf = Buffer.alloc(0);            // OpenAI -> Orchestrator bytes
let uplinkPacer = null;
let downlinkPacer = null;
let openaiSpeaking = false;                   // heuristic flag
let prefillSent = false;                      // ensure prefill runs once

// Uplink instrumentation (Orchestrator → adapter → OpenAI). Lets us
// distinguish local-side audio starvation (uplinkBuf empties, pacer
// sends silence) from OpenAI-side backpressure (uplinkBuf grows
// because aiSource.onData can't push to OpenAI fast enough).
// Reported once per UPLINK_REPORT_MS together with the existing
// downlink drift_stats so both ends of the pipe are observable.
let _uplink_orchsink_bytes = 0;     // bytes arrived from orchestrator since last report
let _uplink_orchsink_chunks = 0;    // number of orchSink.ondata callbacks
let _uplink_sent_to_openai = 0;     // bytes actually pushed to aiSource (real audio)
let _uplink_silence_filled = 0;     // bytes filled with silence because uplinkBuf was empty
let _uplink_partial_filled = 0;     // bytes filled from a partial uplinkBuf (zero-padded tail)
let _uplink_last_report = Date.now();
const UPLINK_REPORT_MS = 1000;

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

// ------------- Stdin message handling (used by Tasky server) ----------------
// When stdin is piped, accept JSON-line messages for prompt injection and tool
// configuration.  When stdin is not piped (normal adapter usage), this is a no-op.
let pendingStdinMessages = [];
let taskyServerUrl = null;
let responseInFlight = false;
let pendingResponseCreate = null;  // { delay } queued for after response.done

function scheduleResponseCreate(delay) {
  if (!responseInFlight) {
    // No response in progress — fire after the delay
    console.error(`[${ts()}] [Adapter-${ROLE}] Triggering response in ${delay}ms`);
    setTimeout(() => safeSendDC({ type: 'response.create' }), delay);
  } else {
    // Response in progress — queue it for after response.done
    console.error(`[${ts()}] [Adapter-${ROLE}] Response in flight, queuing trigger (delay=${delay}ms after done)`);
    pendingResponseCreate = { delay };
  }
}

function onResponseDone() {
  responseInFlight = false;
  if (pendingResponseCreate) {
    const { delay } = pendingResponseCreate;
    pendingResponseCreate = null;
    console.error(`[${ts()}] [Adapter-${ROLE}] Response done, firing queued trigger in ${delay}ms`);
    setTimeout(() => safeSendDC({ type: 'response.create' }), delay);
  }
}

function processStdinMessage(msg) {
  if (msg.type === 'inject' && msg.text) {
    console.error(`[${ts()}] [Adapter-${ROLE}] Prompt injection: ${msg.text}`);
    safeSendDC({
      type: 'conversation.item.create',
      item: {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: msg.text }]
      }
    });
    if (msg.trigger_response) {
      scheduleResponseCreate(msg.trigger_response_delay || 500);
    }
  } else if (msg.type === 'configure_tools' && msg.tools) {
    taskyServerUrl = msg.serverUrl || null;
    const tools = msg.tools.map(t => ({
      type: 'function',
      name: t.name,
      description: t.description,
      parameters: t.parameters,
    }));
    const isGA = true /* Beta API retired; all Realtime models now use GA endpoints */;
    const sess = isGA ? { type: 'realtime', tools } : { tools };
    safeSendDC({ type: 'session.update', session: sess });
    console.error(`[${ts()}] [Adapter-${ROLE}] Configured ${tools.length} tools (server=${taskyServerUrl})`);
  }
}

if (process.stdin && process.stdin.readable) {
  let stdinBuf = '';
  process.stdin.setEncoding('utf8');
  process.stdin.on('data', (chunk) => {
    stdinBuf += chunk;
    let nl;
    while ((nl = stdinBuf.indexOf('\n')) !== -1) {
      const line = stdinBuf.slice(0, nl).trim();
      stdinBuf = stdinBuf.slice(nl + 1);
      if (!line) continue;
      try {
        const msg = JSON.parse(line);
        if (aiDC && aiDC.readyState === 'open') {
          processStdinMessage(msg);
        } else {
          pendingStdinMessages.push(msg);
        }
      } catch (e) {
        // ignore non-JSON lines
      }
    }
  });
  process.stdin.on('error', () => {}); // stdin may close unexpectedly
}

// ------------- Function call handler (tool use via Tasky server) ------------
async function handleFunctionCall(evt) {
  const { call_id, name, arguments: argsStr } = evt;
  let args;
  try { args = JSON.parse(argsStr); } catch { args = {}; }
  console.error(`[${ts()}] [Adapter-${ROLE}] Tool call: ${name}(${JSON.stringify(args)})`);

  let output;
  if (taskyServerUrl) {
    try {
      const res = await _fetch(`${taskyServerUrl}/api/tool`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, arguments: args }),
      });
      const data = await res.json();
      output = JSON.stringify(data.result);
    } catch (e) {
      console.error(`[${ts()}] [Adapter-${ROLE}] Tool call HTTP error:`, e.message);
      output = JSON.stringify({ error: e.message });
    }
  } else {
    output = JSON.stringify({ error: 'No server configured for tool execution' });
  }

  safeSendDC({
    type: 'conversation.item.create',
    item: { type: 'function_call_output', call_id, output },
  });
  safeSendDC({ type: 'response.create' });
}

(async function main() {
  if (TRANSCRIBE_ONLY) {
    console.error(`[${ts()}] [Adapter-${ROLE}] Starting in transcribeOnly mode (tapUrl=${TAP_URL})`);
    await connectOpenAITranscribeOnly();
    startTranscribeOnlyPacer();
  } else {
    console.error(`[${ts()}] [Adapter-${ROLE}] Starting (signal=${SIGNAL_URL}) model=${MODEL} voice=${VOICE} turn=${TURN_MODE} vad=${VAD_MODE}`);
    await connectOrchestrator();
    await connectOpenAI();
    startBridgePacers();
  }
})();

// ------------- Orchestrator leg -------------
async function connectOrchestrator() {
  orchPC = new wrtc.RTCPeerConnection();
  orchSource = new RTCAudioSource({ sampleRate: WIRE_SR, channelCount: 1 });
  const outTrack = orchSource.createTrack();
  orchPC.addTrack(outTrack);

  orchPC.ontrack = ({ track }) => {
    if (track.kind !== 'audio') return;
    if (orchSink) orchSink.stop();
    orchSink = new RTCAudioSink(track, { sampleRate: WIRE_SR, channelCount: 1, bitDepth: 16 });
    let count = 0;
    orchSink.ondata = ({ samples }) => {
      const view = Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength);
      if (count++ < 5) console.error(`[${ts()}] [Adapter-${ROLE}] orchSink first bytesIn=${view.length}`);
      uplinkBuf = Buffer.concat([uplinkBuf, view]);
      _uplink_orchsink_bytes += view.length;
      _uplink_orchsink_chunks += 1;
    };
  };

  ws = new WebSocket(SIGNAL_URL);
  ws.on('error', (err) => console.error(`[${ts()}] [Adapter-${ROLE}] Orchestrator WS error:`, err));
  await new Promise((res) => ws.once('open', res));

  orchPC.onicecandidate = ({ candidate }) => {
    if (!candidate) return;
    ws.send(JSON.stringify({
      role: ROLE, type: 'candidate', candidate: {
        candidate: candidate.candidate, sdpMid: candidate.sdpMid, sdpMLineIndex: candidate.sdpMLineIndex
      }
    }));
  };

  const offer = await orchPC.createOffer();
  await orchPC.setLocalDescription(offer);
  ws.send(JSON.stringify({ role: ROLE, type: 'offer', sdp: offer.sdp }));

  ws.on('message', async (msg) => {
    const data = JSON.parse(msg);
    if (data.type === 'answer' && data.role === ROLE) {
      await orchPC.setRemoteDescription({ type: 'answer', sdp: data.sdp });
      console.error(`[${ts()}] [Adapter-${ROLE}] Orchestrator answer applied.`);
    } else if (data.type === 'candidate' && data.role === ROLE) {
      // Orchestrator broadcasts; only pick candidates destined for our peer
      try { await orchPC.addIceCandidate(data.candidate); } catch { }
    }
  });
}

// ------------- OpenAI leg (WebRTC) -------------
async function connectOpenAI() {
  // 1) Get ephemeral session
  const session = await postJSON(`${TOKEN_SERVER_URL}/session`, {
    model: MODEL,
    // You can preconfigure these here or send via data channel later:
    // voice: VOICE,
    // instructions: SYSTEM_PROMPT,
    // turn_detection: TURN_MODE === 'server_vad' ? { type: 'server_vad' } : undefined,
  });
  const EPHEMERAL_KEY = session?.client_secret?.value;
  if (!EPHEMERAL_KEY) throw new Error('Ephemeral key missing from token server response.');

  // 2) Build PC toward OpenAI
  aiPC = new wrtc.RTCPeerConnection();
  aiSource = new RTCAudioSource({ sampleRate: WIRE_SR, channelCount: 1 });
  const aiOutTrack = aiSource.createTrack();
  aiPC.addTrack(aiOutTrack);

  // Receive audio back from OpenAI
  aiPC.ontrack = ({ track }) => {
    if (track.kind !== 'audio') return;
    if (aiSink) aiSink.stop();
    aiSink = new RTCAudioSink(track, { sampleRate: WIRE_SR, channelCount: 1, bitDepth: 16 });
    let first = true;
    // ── Buffer drift control + drain-during-silence ────────────────────
    // OpenAI streams audio continuously at slightly-faster-than-48 kHz
    // (~0.4% measured), so downlinkBuf would monotonically grow over a
    // long session without intervention — pushing all of B's audio
    // increasingly late vs wall clock in B.wav (and in everything
    // downstream of B's pacer). Fix: when the buffer is above target
    // AND the incoming chunk is silence (RMS below -45 dBFS), drop the
    // chunk instead of appending it. The pacer fills with local silence,
    // the buffer drains naturally back to target. Glitches happen only
    // during silent stretches and are inaudible.
    const SILENCE_RMS_INT16 = Math.pow(10, -45 / 20) * 32768; // ~184
    const BUFFER_TARGET_BYTES = WIRE_SR * 0.05 * 2;            // 50 ms = 4800 bytes
    let _drift_incoming_bytes = 0;
    let _drift_callback_count = 0;
    let _drift_dropped_bytes = 0;
    let _drift_last_report = Date.now();
    const _DRIFT_REPORT_MS = 1000;
    aiSink.ondata = ({ samples }) => {
      const view = Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength);
      if (first) { console.error(`[${ts()}] [Adapter-${ROLE}] OpenAI sink first bytesIn=${view.length}`); first = false; }
      _drift_incoming_bytes += view.length;
      _drift_callback_count += 1;

      // Compute incoming chunk RMS to decide whether it's silent
      let sumSq = 0;
      const nSamples = view.length / 2;
      for (let i = 0; i < view.length; i += 2) {
        const v = view.readInt16LE(i);
        sumSq += v * v;
      }
      const rms = Math.sqrt(sumSq / Math.max(nSamples, 1));
      const isSilence = rms < SILENCE_RMS_INT16;

      if (isSilence && downlinkBuf.length > BUFFER_TARGET_BYTES) {
        // Drop the silence chunk. The pacer's silence padding will fill
        // the gap and the buffer drains toward the target.
        _drift_dropped_bytes += view.length;
      } else {
        downlinkBuf = Buffer.concat([downlinkBuf, view]);
        openaiSpeaking = true;
      }

      const now = Date.now();
      if (now - _drift_last_report >= _DRIFT_REPORT_MS) {
        const elapsed_ms = now - _drift_last_report;
        console.error(`[${ts()}] [Adapter-${ROLE}] drift_stats elapsed_ms=${elapsed_ms} bytes_in=${_drift_incoming_bytes} dropped=${_drift_dropped_bytes} buf_bytes=${downlinkBuf.length}`);
        _drift_incoming_bytes = 0;
        _drift_callback_count = 0;
        _drift_dropped_bytes = 0;
        _drift_last_report = now;
      }
    };
  };

  // Data channel for Realtime events
  aiDC = aiPC.createDataChannel('oai-events');
  aiDC.onopen = () => {
    console.error(`[${ts()}] [Adapter-${ROLE}] OpenAI data channel open. VAD mode: ${VAD_MODE}`);
    // Build turn_detection config based on VAD_MODE
    let turn_detection_config;
    if (TURN_MODE === 'none') {
      turn_detection_config = undefined;
    } else if (VAD_MODE === 'fast') {
      // Fast mode: stricter threshold, shorter silence duration for quicker response
      turn_detection_config = {
        type: 'server_vad',
        create_response: true,
        interrupt_response: false,
        threshold: 0.6,
        prefix_padding_ms: 1200,
        silence_duration_ms: 450
      };
    } else {
      // Slow mode (default): more patient, allows interruption, higher threshold
      turn_detection_config = {
        type: 'server_vad',
        create_response: true,
        interrupt_response: true,
        threshold: 0.9,
      };
    }

    // Send initial session.update (GA models require session.type=realtime
    // and a different shape: voice/turn_detection nested under audio).
    const isGA = true /* Beta API retired; all Realtime models now use GA endpoints */;
    const session = isGA ? {
      type: 'realtime',
      instructions: SYSTEM_PROMPT,
      audio: {
        input: {
          turn_detection: turn_detection_config,
          ...(ENABLE_TRANSCRIPTION ? { transcription: { model: 'gpt-4o-transcribe' } } : {}),
        },
        output: { voice: VOICE },
      },
    } : {
      instructions: SYSTEM_PROMPT,
      voice: VOICE,
      turn_detection: turn_detection_config,
      input_audio_transcription: ENABLE_TRANSCRIPTION ? { model: 'gpt-4o-transcribe' } : undefined,
    };
    safeSendDC({ type: 'session.update', session });

    // Process any stdin messages that arrived before the DC was open
    for (const pending of pendingStdinMessages) {
      processStdinMessage(pending);
    }
    pendingStdinMessages = [];
  };
  let connectionStartTime = Date.now();

  aiDC.onmessage = (ev) => {
    try {
      const evt = JSON.parse(ev.data);
      if (argv.logEvents && evt?.type !== 'response.audio_transcript.delta') {
        console.error(`[${new Date().toISOString()}] [Adapter-${ROLE}] ← ${evt.type}`, ev.data);
      }
      if (evt?.type === 'session.updated' && !prefillSent) {
        sendPrefillHistory();
      }

      // Track response lifecycle for queued trigger_response
      if (evt?.type === 'response.created') {
        responseInFlight = true;
      } else if (evt?.type === 'response.done') {
        onResponseDone();
      }

      // agent_stop fires when audio playback ends, not when generation completes
      if (evt?.type === 'output_audio_buffer.stopped') {
        process.stdout.write(JSON.stringify({ type: 'event', event: 'agent_stop' }) + '\n');
      }

      // Handle function calls from GPT (tool use)
      if (evt?.type === 'response.function_call_arguments.done') {
        handleFunctionCall(evt);
      }

      // Emit transcripts to stdout as structured JSON
      if (evt?.type === 'conversation.item.input_audio_transcription.delta') {
        process.stdout.write(JSON.stringify({ type: 'transcript_delta', source: 'input', delta: evt.delta }) + '\n');
      }
      if (evt?.type === 'conversation.item.input_audio_transcription.completed') {
        process.stdout.write(JSON.stringify({ type: 'transcript', source: 'input', text: evt.transcript }) + '\n');
      }
      if (evt?.type === 'response.audio_transcript.done') {
        process.stdout.write(JSON.stringify({ type: 'transcript', source: 'response', text: evt.transcript }) + '\n');
      }

      // Prevent Examinee from speaking before Examiner — UNLESS the
      // examiner is configured to wait for the agent to greet first.
      if (ROLE === 'B' && evt?.type === 'response.created' && !argv.speakFirst) {
        if (Date.now() - connectionStartTime < 4000) {
          console.error(`[${ts()}] [Adapter-B] Canceling early response to wait for Examiner`);
          safeSendDC({ type: 'response.cancel' });
        }
      }
    } catch { }
  };
  aiDC.onerror = (e) => console.error(`[${ts()}] [Adapter-${ROLE}] datachannel error`, e);

  // 3) Offer/answer via HTTP SDP exchange
  // Make sure ICE candidates are gathered before POSTing the SDP
  const offer = await aiPC.createOffer({ offerToReceiveAudio: true, offerToReceiveVideo: false });
  await aiPC.setLocalDescription(offer);
  await waitForIceGatheringComplete(aiPC);

  const sdpAnswer = await postSDP(`https://api.openai.com/v1/realtime${true /* Beta API retired; all Realtime models now use GA endpoints */ ? '/calls' : ''}?model=${encodeURIComponent(MODEL)}`, EPHEMERAL_KEY, aiPC.localDescription.sdp);
  await aiPC.setRemoteDescription({ type: 'answer', sdp: sdpAnswer });
  console.error(`[${ts()}] [Adapter-${ROLE}] OpenAI answer applied.`);
}

function safeSendDC(obj) {
  if (!aiDC || aiDC.readyState !== 'open') return;
  try { aiDC.send(JSON.stringify(obj)); } catch (e) { console.error(`[${ts()}] DC send error`, e); }
}

// ------------- Bridging -------------
function startBridgePacers() {
  if (uplinkPacer) clearInterval(uplinkPacer);
  if (downlinkPacer) clearInterval(downlinkPacer);

  const bridgeStartTime = Date.now();

  // Orchestrator -> OpenAI: emit ten 10ms frames per 100ms tick
  const FRAME_10MS = Math.floor(WIRE_SR * 0.01) * 2; // bytes at WIRE_SR PCM16 mono
  uplinkPacer = setInterval(() => {
    // For Role B (examinee), ignore the first 4 seconds of audio from orchestrator
    // to prevent fake silence/noise from triggering VAD before the examiner speaks.
    const shouldMute = (ROLE === 'B' && (Date.now() - bridgeStartTime < 4000));

    for (let i = 0; i < 10; i += 1) {
      let out;
      let _filled_kind = 'real'; // for uplink stats
      if (uplinkBuf.length >= FRAME_10MS) {
        out = uplinkBuf.slice(0, FRAME_10MS);
        uplinkBuf = uplinkBuf.slice(FRAME_10MS);
      } else if (uplinkBuf.length > 0) {
        out = Buffer.alloc(FRAME_10MS);
        uplinkBuf.copy(out, 0, 0, uplinkBuf.length);
        uplinkBuf = Buffer.alloc(0);
        _filled_kind = 'partial'; // had some real audio but ran short
      } else {
        out = Buffer.alloc(FRAME_10MS); // silence
        _filled_kind = 'silence';
      }

      if (shouldMute) {
        out.fill(0); // overwrite with pure silence
      }

      const ab = new ArrayBuffer(FRAME_10MS);
      new Uint8Array(ab).set(out);
      const samples = new Int16Array(ab);
      aiSource.onData({ samples, sampleRate: WIRE_SR, bitsPerSample: 16, channelCount: 1 });

      // Track what kind of frame we pushed so we can tell silence-starve
      // (uplinkBuf empty) from real audio flow.
      if (_filled_kind === 'real') {
        _uplink_sent_to_openai += FRAME_10MS;
      } else if (_filled_kind === 'partial') {
        _uplink_partial_filled += FRAME_10MS;
      } else {
        _uplink_silence_filled += FRAME_10MS;
      }
    }

    // Periodic uplink-side report (paired w/ existing drift_stats on downlink).
    // Reading: bytes_in = arrived from orchestrator; sent = pushed to OpenAI;
    // silence = pacer ran dry and emitted silence; partial = tail-padded.
    // buf_bytes = current uplinkBuf depth.
    //   - silence stays HIGH with bytes_in LOW   → local CPU/IO starves us
    //     (orchestrator → adapter audio not flowing).
    //   - buf_bytes GROWS over time              → OpenAI-side backpressure
    //     (aiSource.onData not draining fast enough into the WebRTC PC).
    //   - both healthy + small silence            → normal operation.
    const now = Date.now();
    if (now - _uplink_last_report >= UPLINK_REPORT_MS) {
      const elapsed_ms = now - _uplink_last_report;
      console.error(
        `[${ts()}] [Adapter-${ROLE}] uplink_stats elapsed_ms=${elapsed_ms} ` +
        `bytes_in=${_uplink_orchsink_bytes} orchsink_chunks=${_uplink_orchsink_chunks} ` +
        `sent=${_uplink_sent_to_openai} silence=${_uplink_silence_filled} ` +
        `partial=${_uplink_partial_filled} buf_bytes=${uplinkBuf.length}`
      );
      _uplink_orchsink_bytes = 0;
      _uplink_orchsink_chunks = 0;
      _uplink_sent_to_openai = 0;
      _uplink_silence_filled = 0;
      _uplink_partial_filled = 0;
      _uplink_last_report = now;
    }
  }, 100);

  // OpenAI -> Orchestrator: emit ten 10ms frames per 100ms tick
  downlinkPacer = setInterval(() => {
    for (let i = 0; i < 10; i += 1) {
      let out;
      if (downlinkBuf.length >= FRAME_10MS) {
        out = downlinkBuf.slice(0, FRAME_10MS);
        downlinkBuf = downlinkBuf.slice(FRAME_10MS);
        openaiSpeaking = true;
      } else if (downlinkBuf.length > 0) {
        out = Buffer.alloc(FRAME_10MS);
        downlinkBuf.copy(out, 0, 0, downlinkBuf.length);
        downlinkBuf = Buffer.alloc(0);
        openaiSpeaking = true;
      } else {
        out = Buffer.alloc(FRAME_10MS); // silence
        openaiSpeaking = false;
      }
      const ab = new ArrayBuffer(FRAME_10MS);
      new Uint8Array(ab).set(out);
      const samples = new Int16Array(ab);
      orchSource.onData({ samples, sampleRate: WIRE_SR, bitsPerSample: 16, channelCount: 1 });
    }
  }, 100);
}

// ------------- TranscribeOnly mode -------------
let tapWs = null;

async function connectOpenAITranscribeOnly() {
  // 1) Get ephemeral session
  const session = await postJSON(`${TOKEN_SERVER_URL}/session`, { model: MODEL });
  const EPHEMERAL_KEY = session?.client_secret?.value;
  if (!EPHEMERAL_KEY) throw new Error('Ephemeral key missing from token server response.');

  // 2) Build PC toward OpenAI — transcription only, no response generation
  aiPC = new wrtc.RTCPeerConnection();
  aiSource = new RTCAudioSource({ sampleRate: WIRE_SR, channelCount: 1 });
  const aiOutTrack = aiSource.createTrack();
  aiPC.addTrack(aiOutTrack);

  // We don't need to receive audio back in transcribeOnly mode, but must handle ontrack
  aiPC.ontrack = () => {};

  // Data channel for Realtime events
  aiDC = aiPC.createDataChannel('oai-events');
  aiDC.onopen = () => {
    console.error(`[${ts()}] [Adapter-${ROLE}] TranscribeOnly: OpenAI data channel open.`);
    safeSendDC({
      type: 'session.update',
      session: {
        instructions: '',
        voice: VOICE,
        turn_detection: { type: 'server_vad', silence_duration_ms: 500 },
        input_audio_transcription: { model: 'gpt-4o-transcribe' },
      }
    });
  };

  aiDC.onmessage = (ev) => {
    try {
      const evt = JSON.parse(ev.data);
      if (argv.logEvents && evt?.type !== 'response.audio_transcript.delta') {
        console.error(`[${new Date().toISOString()}] [Adapter-${ROLE}] ← ${evt.type}`);
      }
      // Cancel any auto-generated responses — we only want transcription
      if (evt?.type === 'response.created') {
        safeSendDC({ type: 'response.cancel' });
      }
      // Emit transcripts to stdout
      if (evt?.type === 'conversation.item.input_audio_transcription.delta') {
        process.stdout.write(JSON.stringify({ type: 'transcript_delta', source: 'input', delta: evt.delta }) + '\n');
      }
      if (evt?.type === 'conversation.item.input_audio_transcription.completed') {
        process.stdout.write(JSON.stringify({ type: 'transcript', source: 'input', text: evt.transcript }) + '\n');
      }
    } catch { }
  };
  aiDC.onerror = (e) => console.error(`[${ts()}] [Adapter-${ROLE}] transcribeOnly DC error`, e);

  // 3) SDP exchange
  const offer = await aiPC.createOffer({ offerToReceiveAudio: true, offerToReceiveVideo: false });
  await aiPC.setLocalDescription(offer);
  await waitForIceGatheringComplete(aiPC);

  const sdpAnswer = await postSDP(`https://api.openai.com/v1/realtime${true /* Beta API retired; all Realtime models now use GA endpoints */ ? '/calls' : ''}?model=${encodeURIComponent(MODEL)}`, EPHEMERAL_KEY, aiPC.localDescription.sdp);
  await aiPC.setRemoteDescription({ type: 'answer', sdp: sdpAnswer });
  console.error(`[${ts()}] [Adapter-${ROLE}] TranscribeOnly: OpenAI answer applied.`);

  // 4) Connect to audio tap
  if (!TAP_URL) throw new Error('--tapUrl is required for --transcribeOnly mode');
  tapWs = new WebSocket(TAP_URL);
  tapWs.binaryType = 'arraybuffer';
  tapWs.on('open', () => {
    console.error(`[${ts()}] [Adapter-${ROLE}] TranscribeOnly: connected to tap at ${TAP_URL}`);
    tapWs.send(JSON.stringify({ type: 'subscribe', role: ROLE }));
  });
  tapWs.on('message', (data) => {
    if (data instanceof ArrayBuffer || Buffer.isBuffer(data)) {
      const buf = Buffer.isBuffer(data) ? data : Buffer.from(data);
      uplinkBuf = Buffer.concat([uplinkBuf, buf]);
    }
  });
  tapWs.on('error', (err) => console.error(`[${ts()}] [Adapter-${ROLE}] Tap WS error:`, err.message));
  tapWs.on('close', () => console.error(`[${ts()}] [Adapter-${ROLE}] Tap WS closed`));
}

function startTranscribeOnlyPacer() {
  // Only uplink pacer: tap audio → OpenAI for transcription
  const FRAME_10MS = Math.floor(WIRE_SR * 0.01) * 2;
  uplinkPacer = setInterval(() => {
    for (let i = 0; i < 10; i += 1) {
      let out;
      if (uplinkBuf.length >= FRAME_10MS) {
        out = uplinkBuf.slice(0, FRAME_10MS);
        uplinkBuf = uplinkBuf.slice(FRAME_10MS);
      } else if (uplinkBuf.length > 0) {
        out = Buffer.alloc(FRAME_10MS);
        uplinkBuf.copy(out, 0, 0, uplinkBuf.length);
        uplinkBuf = Buffer.alloc(0);
      } else {
        out = Buffer.alloc(FRAME_10MS);
      }
      const ab = new ArrayBuffer(FRAME_10MS);
      new Uint8Array(ab).set(out);
      const samples = new Int16Array(ab);
      aiSource.onData({ samples, sampleRate: WIRE_SR, bitsPerSample: 16, channelCount: 1 });
    }
  }, 100);
}

// ------------- Utils -------------
async function postJSON(url, body) {
  const r = await _fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`POST ${url} ${r.status}`);
  return await r.json();
}

async function postSDP(url, bearer, sdp) {
  const r = await _fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/sdp', 'Authorization': `Bearer ${bearer}` }, body: sdp });
  if (!r.ok) {
    const t = await r.text().catch(() => '');
    throw new Error(`SDP POST failed ${r.status}: ${t}`);
  }
  return await r.text();
}

function waitForIceGatheringComplete(pc) {
  if (pc.iceGatheringState === 'complete') return Promise.resolve();
  return new Promise(resolve => {
    const check = () => {
      if (pc.iceGatheringState === 'complete') {
        pc.removeEventListener('icegatheringstatechange', check);
        resolve();
      }
    };
    pc.addEventListener('icegatheringstatechange', check);
    // Safety timeout in case some environments never report complete
    setTimeout(() => { pc.removeEventListener('icegatheringstatechange', check); resolve(); }, 1500);
  });
}

async function shutdown() {
  console.error(`[${ts()}] [Adapter-${ROLE}] Shutting down...`);
  try { if (uplinkPacer) clearInterval(uplinkPacer); } catch { }
  try { if (downlinkPacer) clearInterval(downlinkPacer); } catch { }
  try { if (orchSink) orchSink.stop(); } catch { }
  try { if (aiSink) aiSink.stop(); } catch { }
  try { if (orchPC) orchPC.close(); } catch { }
  try { if (aiPC) aiPC.close(); } catch { }
  try { if (ws && ws.readyState === WebSocket.OPEN) ws.close(); } catch { }
  try { if (tapWs && tapWs.readyState === WebSocket.OPEN) tapWs.close(); } catch { }
  process.exit(0);
}

// ------------- Prefill helpers -------------
function coercePrefillTurns() {
  // Accept either CLI --prefill as one user turn, or --prefillFile as JSON array of {role, text}
  // Valid roles per Realtime message are 'system'|'user'|'assistant'.
  try {
    if (PREFILL_FILE) {
      const raw = fs.readFileSync(PREFILL_FILE, 'utf8');
      const arr = JSON.parse(raw);
      if (Array.isArray(arr) && arr.length > 0) {
        return arr
          .filter(t => t && typeof t.role === 'string' && typeof t.text === 'string' && t.text.length > 0)
          .map(t => ({ role: t.role, text: t.text }));
      }
    }
  } catch (e) {
    console.warn(`[${ts()}] [Adapter-${ROLE}] Failed to parse prefillFile:`, e?.message || e);
  }
  if (PREFILL_TEXT) {
    return [{ role: 'user', text: PREFILL_TEXT }];
  }
  return [];
}

function sendPrefillHistory() {
  const turns = coercePrefillTurns();
  if (!turns.length) {
    prefillSent = true;
    if (AUTOSTART) {
      safeSendDC({ type: 'response.create', response: (true /* Beta API retired; all Realtime models now use GA endpoints */) ? {} : { modalities: ['audio', 'text'] } });
    }
    return;
  }
  // Send systemPrompt as a system message if provided
  // if (SYSTEM_PROMPT) {
  //   safeSendDC({
  //     type: 'conversation.item.create',
  //     item: { type: 'message', role: 'system', content: [{ type: 'input_text', text: SYSTEM_PROMPT }] }
  //   });
  // }
  for (const t of turns) {
    safeSendDC({
      type: 'conversation.item.create',
      item: { type: 'message', role: t.role, content: [{ type: 'input_text', text: t.text }] }
    });
  }
  prefillSent = true;
  if (AUTOSTART) {
    safeSendDC({ type: 'response.create', response: { modalities: ['audio', 'text'] } });
  }
}
