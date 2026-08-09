#!/usr/bin/env node
/**
 * clipPlayer_adapter.js (Role A — Clip Player)
 *
 * Bridges the orchestrator (WebRTC @ 48 kHz PCM16) <-> OpenAI Realtime (WebRTC).
 * Instead of generating speech, the Realtime model acts as a routing oracle:
 *   - It calls select_clip(node_id) to choose which pre-recorded clip to play, or
 *   - It calls select_fallback() when the examinee's response is off-script.
 *
 * Clips are injected directly into downlinkBuf (same pacer as gptRealtime_adapter).
 * tool_choice: 'required' ensures the model NEVER generates audio.
 */

require('dotenv').config();
const fs = require('fs');
const path = require('path');
const wrtc = require('wrtc');
const WebSocket = require('ws');
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');
const { RTCAudioSource, RTCAudioSink } = wrtc.nonstandard;
const wavDecoder = require('wav-decoder');
const { createDecisionTree } = require('./decision_tree');

// ── Configuration ─────────────────────────────────────────────────────────────
// Role A's routing oracle is pinned to gpt-realtime-2 — Role B's model is
// what we vary in experiments, but Role A needs to stay consistent across
// runs so the customer-side routing behaviour is the same baseline for
// every Role B impl we compare. Don't honour OPENAI_REALTIME_MODEL here:
// that env var is meant for Role B selection in the eval runner.
const DEFAULT_MODEL = 'gpt-realtime-2';
const DEFAULT_TOKEN_SERVER = process.env.OPENAI_TOKEN_SERVER || 'http://localhost:3002';
const WIRE_SR = parseInt(process.env.WIRE_SAMPLE_RATE, 10) || 48000;

const _fetch = (typeof fetch !== 'undefined')
  ? fetch
  : (async (...args) => (await import('node-fetch')).default(...args));

const argv = yargs(hideBin(process.argv))
  .scriptName('clipPlayer_adapter')
  .usage('$0 -r A -s <ws://host:port/signal> --decisionTree PATH [opts]')
  .option('role',         { alias: 'r', type: 'string', choices: ['A', 'B'], demandOption: true })
  .option('signalUrl',    { alias: 's', type: 'string', demandOption: true })
  .option('tokenServer',  { type: 'string', default: DEFAULT_TOKEN_SERVER })
  .option('model',        { type: 'string', default: DEFAULT_MODEL })
  .option('voice',        { type: 'string', default: 'alloy' })
  .option('vadMode',      { type: 'string', choices: ['slow', 'fast'], default: 'slow',
                            describe: 'VAD timing mode: slow (default) or fast' })
  .option('decisionTree', { type: 'string', demandOption: true,
                            describe: 'Path to decision tree JSON' })
  .option('dt.clipsBaseDir', { type: 'string', default: '',
                            describe: 'Override clips_base_dir from JSON' })
  .option('noiseDir',     { type: 'string', default: '',
                            describe: 'Directory of background-noise wavs to cycle through between clips. Empty = emit digital silence (old behavior).' })
  .option('initialDelayMs', { type: 'number', default: 8000,
                              describe: 'Fallback timeout: if the examinee has not started speaking within this many ms after the session opens, fire the initial clip anyway. Reset when speech_started arrives, so the actual wait is unbounded once the agent begins their greeting.' })
  .option('toolServerUrl',  { type: 'string', default: '',
                              describe: 'Tool server URL (http://host:port). When set, used for the customer-side auto-submit fallback fired only when missing_answer is picked at end-of-order (no remaining clips to play and the adapter would otherwise sit silent).' })
  .option('profile',      { type: 'string', default: 'default',
                            describe: 'Customer routing profile name. Resolves to clipPlayer_profiles/<name>.md and is appended to the base system prompt via {{PROFILE_INSTRUCTIONS}}.' })
  .option('taskOverrides', { type: 'string', default: '',
                             describe: 'Optional per-task override JSON sidecar (same file the complex builder consumes). Read for clip_router_hints: a map from clip_id to a guidance string appended to that clip\'s description in valid_choices, used for stage-specific routing cues the generic kind-based description can\'t derive (e.g. customer mispronunciations the agent is expected to surface).' })
  .option('taskConfig',   { type: 'string', default: '',
                             describe: 'Optional task config JSON path (the .complex.task.json or .task.json that the eval is running against). Used to derive DEFAULT_ELIGIBLE_OPTIONS for the just_the_defaults fallback clip — the adapter computes which required-with-default option groups apply to each expected_cart item and injects the list into the customer-side system prompt.' })
  .option('logEvents',    { type: 'boolean', default: true })
  .help().argv;

const ROLE           = argv.role;
const SIGNAL_URL     = argv.signalUrl;
const TOKEN_SERVER_URL = argv.tokenServer;
const TOOL_SERVER_URL = argv.toolServerUrl;
const MODEL          = argv.model;
const VOICE          = argv.voice;
const VAD_MODE       = argv.vadMode;

// ── Decision tree library ────────────────────────────────────────────────
// All tree loading, override patching, DEFAULT-ELIGIBLE-OPTIONS derivation,
// clip indexing, yes/no/jtd/thats_all detection, valid_choices computation,
// and progression state (dt.getCurrentStage(), dt.getCurrentItem(), playedClips,
// dt.getLastPlayedClipId(), dt.isConversationEnded()) live in decision_tree.js. The
// adapter handles WebRTC, audio playback, VAD, and tool-call I/O.
const dt = createDecisionTree({
  treePath: path.resolve(process.cwd(), argv.decisionTree),
  taskConfigPath: argv.taskConfig ? path.resolve(process.cwd(), argv.taskConfig) : null,
  overridesPath: argv.taskOverrides ? path.resolve(process.cwd(), argv.taskOverrides) : null,
  clipsBaseDirOverride: argv.clipsBaseDir,
});
const DEFAULT_ELIGIBLE_OPTIONS_TEXT = dt.defaultEligibleOptionsText;

// ── Adapter-side state (orchestration only) ─────────────────────────
// Tree-progression state (dt.getCurrentStage()/dt.getCurrentItem()/playedClips/
// dt.getLastPlayedClipId()/dt.isConversationEnded()/dt.getLastValidChoices()) lives in `dt`.
let bootstrapSent  = false;
let initialTurnFired = false;
let initialFallbackTimer = null;
let greetingItemId = null;  // The conversation.item id for the agent's greeting (we delete it before bootstrap)
let routingResponseInFlight = false;  // true while we're between response.created and the tool's function_call_output
let clipPlaybackEndsAt = 0;  // epoch ms when the currently-playing clip will finish draining
// NOTE: A-side whisper transcription of B's audio is enabled in the
// session config so the events flow into adapter_A.log (where
// build_viz_data picks them up for the dashboard's "🎤 A heard" band).
// We deliberately do NOT track the latest transcript in adapter state
// or inject it into tool results — by the time A fires a routing
// decision, whisper is always one turn stale, and the resulting frame
// mismatch was empirically harmful.

// (Clip indexing, helpers, yes/no/jtd/thats_all detection moved to
// decision_tree.js. See dt.parseClipId, dt.nextItemAfter,
// dt.yesClipBasename, etc.)

// Pre-agent-response suppression for the routing oracle.
// Empirically A's VAD detects two kinds of "extra" speech_stopped events
// before the agent has actually responded:
//   1) The agent's GREETING split into multiple VAD segments (e.g., the
//      period in "Hi, welcome to Harbor House. What would you like to
//      order?" causes VAD to fire two speech_stopped events).
//   2) Any pipeline noise / partial segments occurring during clip playback.
// With `create_response: true`, each speech_stopped auto-fires the routing
// model, and the early ones produce "no examinee response" fallbacks.
//
// The agent physically cannot respond before the clip finishes playing on
// their side (they need to hear it first). So we suppress any auto-response
// that fires within `clip_duration + pad` of the inject. After that window,
// the first speech_stopped is the agent's actual response.
let awaitingAgentTurn = false;        // true after inject until routing fires
let suppressAutoResponseUntil = 0;    // epoch ms; cancel any auto-fired response before this
let agentResponseTimer = null;        // hard ceiling: fire routing even if no agent speech
const AGENT_RESPONSE_TIMEOUT_MS = 8000;  // 8s after the suppression window ends
const POST_CLIP_PAD_MS = 800;         // pipeline lag + VAD silence_duration_ms cushion

// Wrap console.log/error so every line has an ISO wallclock timestamp (used
// by the visualization pipeline to align A's events with B's events that
// already carry [ISO] timestamps).
const _origLog = console.log.bind(console);
const _origErr = console.error.bind(console);
console.log = (...a) => _origLog(`[${new Date().toISOString()}]`, ...a);
console.error = (...a) => _origErr(`[${new Date().toISOString()}]`, ...a);

// Tool call accumulation
let pendingCallId   = null;
let pendingCallName = null;
let pendingArgsBuf  = '';

// WebRTC / bridge
let orchPC, orchSource, orchSink;
let aiPC, aiSource, aiSink, aiDC;
let ws;
let uplinkBuf    = Buffer.alloc(0);
let downlinkBuf  = Buffer.alloc(0);
let uplinkPacer  = null;
let downlinkPacer = null;
let stallWatchdog = null;

// Stall watchdog: if no successful inject happens for this many ms
// (and clips remain to play, and the customer-auto end-of-order submit
// hasn't already fired), log a diagnostic and re-prompt the LLM. Covers
// the case where the
// router silently stopped calling play() — e.g. several response.done
// cancellations in a row before the cancelled_retry recovery above
// could land, or some other path that doesn't trigger the cancelled
// branch. Tuned to 45s: well past normal inter-clip latency (~5-15s)
// but short enough to recover most of the SIGTERM headroom on a
// typical eval.
const STALL_WATCHDOG_INTERVAL_MS = 5_000;
const STALL_WATCHDOG_THRESHOLD_MS = 45_000;
let stallWatchdogLastWarnAt = 0;

// Background-noise cycle (filled at startup from --noiseDir; empty means
// emit digital silence between clips). Each entry is PCM16 mono @ WIRE_SR.
let noiseClips = [];           // Legacy alphabetical list of PCM buffers.
                               // Kept as the fallback rotation when a
                               // manifest has no `noise_rotation` field.
const noiseByName = new Map(); // filename (e.g. "noise_003.wav") → PCM buf.
let noiseRotation = null;      // Active per-clip rotation (PCM buffer list).
                               // Null = use `noiseClips` (legacy fallback).
let noiseClipIdx = 0;          // Cursor into noiseRotation/noiseClips.
let noiseBuf = Buffer.alloc(0);// Partial-frame carryover between pulls.

// Tracks whether agent B is currently speaking from A-VAD's perspective.
// Used by select_clip / select_fallback to refuse injection when the
// agent is mid-utterance: even though the routing oracle only fires after
// silence_duration_ms of quiet, the model then spends 1-2s generating the
// tool call, and B can start speaking again in that gap. Without this
// gate the next clip would land on top of B's continuing audio.
//
// Driven by a simple energy-based VAD on the audio A receives from the
// orchestrator (which IS B's audio). Replaces two prior paths that this
// gate had: A's session VAD speech_started/stopped events (delivered
// laggy/missed under load), and the post-eval calibrated interrupt scan
// (post_eval_calibrate_interrupts.py) — both consolidated into a single
// realtime VAD whose `agentSpeaking` we trust everywhere.
const AGENT_VAD_RMS_INT16 = 300;        // ≈ -41 dBFS. The agent broadcasts true
                                        // silence between turns so a coarse
                                        // threshold suffices.
const AGENT_VAD_HANGOFF_MS = 1000;      // require 1s of continuous silence after
                                        // the agent's last non-silent chunk
                                        // before marking them stopped — covers
                                        // inter-word gaps and brief pauses.
const INTERRUPT_FLAG_WINDOW_MS = 1000;  // if the agent transitions from silence
                                        // to speech within this many ms after a
                                        // clip starts playing, the clip is
                                        // flagged (but not cancelled) as an
                                        // interrupt.
let agentSpeaking = false;
let lastAgentNonSilentAt = 0;
let lastInjectAt = 0;
let lastInjectClipId = null;
let lastInjectFlagged = false;

// Interrupt-recovery gate. When the agent talks OVER a clip (interrupt_flag
// fires), the next play() choice is refused: the agent's interrupting speech
// is likely off-script (they didn't actually hear the clip cleanly), so
// playing another clip immediately would compound the misalignment. Instead
// we wait for the agent to finish their interrupting turn and start a NEW
// turn (a fresh response on top of having heard the clip's full audio), or
// until INTERRUPT_RECOVERY_TIMEOUT_MS elapses — whichever comes first.
const INTERRUPT_RECOVERY_TIMEOUT_MS = 8000;
let interruptRecoveryAt = 0;            // ts when interrupt_flag was raised; 0 = no pending recovery
let agentSilentSinceInterrupt = false;  // set when VAD transitions to silent after interrupt_flag

process.on('SIGINT',  shutdown);
process.on('SIGTERM', shutdown);

// ── Tool definitions ──────────────────────────────────────────────────────────
// Single play tool. `choice` is one of:
//   "next_item" — advance the conversation: usually moves to the next
//                 item, but if the just-played clip has a pending
//                 self-correction, plays that correction instead (the
//                 router never picks corrections directly)
//   "yes"       — play this game's yes clip (confirms a yes/no question)
//   "no"        — play this game's no clip (rejects a yes/no guess)
//   "fallback"  — replay the most recent clip
//   <clip_id>   — one of the entries in the last tool result's valid_choices
//
// The adapter ENFORCES this against the dynamically-computed valid_choices
// set: choices outside the set return status="invalid_choice" and don't
// update state. This prevents the router from selecting a clip that
// belongs to a later item / stage (jump-ahead) or one that's already
// been consumed.
const PLAY_TOOL = {
  type: 'function',
  name: 'play',
  description: (
    'Play the customer audio appropriate for the agent\'s most recent utterance. ' +
    'Pick one option from the most recent tool result\'s valid_choices list.'
  ),
  parameters: {
    type: 'object',
    properties: {
      choice: {
        type: 'string',
        description: (
          'One of: "next_item" (advance the conversation forward), "yes", "no", "fallback" ' +
          '(replay last clip), "missing_answer" (agent asked a valid question OR presented an ' +
          'option set we have no matching clip for), or a specific clip_id from the last ' +
          'valid_choices list.'
        ),
      },
      heard: {
        type: 'string',
        description: (
          'Brief verbatim transcription of WHAT YOU HEARD the agent say in '
          + 'the audio leading up to this routing decision. Use the audio '
          + 'directly, do NOT defer to the whisper transcript. Include '
          + 'their exact words including any question, option pair, or '
          + 'close-out prompt. If you heard nothing (no agent speech '
          + 'since the last clip played), say "(no agent speech)". This '
          + 'is an independent diagnostic signal — pipeline operators use '
          + 'it to detect when the routing decision was based on partial '
          + 'or missing audio.'
        ),
      },
      reasoning: {
        type: 'string',
        description: (
          'Why this choice fits what you heard in the `heard` field. '
          + 'Reference the heard content directly.'
        ),
      },
      acknowledged: {
        type: 'array',
        items: { type: 'string' },
        description: (
          '(Optional but expected every turn) Short clip ids from the '
          + 'PREVIOUS tool response\'s `unacknowledged` list that the '
          + 'agent\'s most recent utterance named, called back to, or '
          + 'asked a follow-up question about. A bare generic ack ("got '
          + 'it", "okay", "sure") with no item-specific words → include '
          + 'ALL current entries in `unacknowledged`. Restatement ids '
          + '(e.g. "i1_restatement1") also count — the adapter maps '
          + 'them to the parent main. Omit or pass [] when the agent '
          + 'ack\'d nothing.'
        ),
      },
    },
    required: ['choice', 'heard', 'reasoning'],
  },
};

// ── System prompt generation ──────────────────────────────────────────────────
// High-level customer-role description. Per-turn reachability is conveyed
// by the `valid_choices` list returned in every tool result — the system
// prompt no longer enumerates the full decision tree.
//
// Body lives in clipPlayer_system_prompt.md (loaded once at module init).
const SYSTEM_PROMPT_PATH = path.join(__dirname, 'clipPlayer_system_prompt.md');
const SYSTEM_PROMPT_TEMPLATE = fs.readFileSync(SYSTEM_PROMPT_PATH, 'utf8').replace(/\n+$/, '');
const PROFILE_PATH = path.join(__dirname, 'clipPlayer_profiles', `${argv.profile}.md`);
if (!fs.existsSync(PROFILE_PATH)) {
  console.error(`[ClipPlayer] profile not found: ${PROFILE_PATH}`);
  process.exit(2);
}
const PROFILE_INSTRUCTIONS = fs.readFileSync(PROFILE_PATH, 'utf8').replace(/\n+$/, '');
console.log(`[ClipPlayer] profile=${argv.profile} (${PROFILE_INSTRUCTIONS ? PROFILE_INSTRUCTIONS.split('\n').length + ' lines' : 'no extra instructions'})`);

function generateSystemPrompt(t) {
  const rendered = SYSTEM_PROMPT_TEMPLATE
    .replace('{{TASK_DESCRIPTION}}', t.description ?? 'undefined')
    .replace('{{DEFAULT_ELIGIBLE_OPTIONS}}', DEFAULT_ELIGIBLE_OPTIONS_TEXT)
    .replace('{{PROFILE_INSTRUCTIONS}}', PROFILE_INSTRUCTIONS);
  // Emit the exact rendered system prompt once per session config so
  // adapter_A.log carries the literal instructions the model saw. Used
  // by the router-turn debug log (see _logRouterTurn) — pair this line
  // with the per-turn entries to reconstruct what the model knew at
  // each decision point.
  console.log(
    `[ClipPlayer:rendered_system_prompt] ${JSON.stringify({
      chars: rendered.length,
      default_eligible_options_text: DEFAULT_ELIGIBLE_OPTIONS_TEXT,
      jtd_basename: dt.justTheDefaultsClipBasename,
      prompt: rendered,
    })}`
  );
  return rendered;
}

// ── VAD config ────────────────────────────────────────────────────────────────
function buildVadConfig(mode) {
  if (mode === 'fast') {
    return {
      type: 'server_vad',
      create_response: true,
      interrupt_response: false,
      threshold: 0.6,
      prefix_padding_ms: 1200,
      silence_duration_ms: 600
    };
  }
  return {
    // Conservative-defaults mode. The routing oracle should only act when
    // the agent (role B, heard through the orchestrator) is clearly done
    // speaking — premature routing injects the next clip on top of the
    // agent's still-streaming audio, which makes the evaluation noisier.
    //
    //   threshold (0→1): minimum VAD confidence to register speech. OpenAI's
    //     default is 0.5. Bumping to 0.6 makes the routing less twitchy about
    //     stray transients (clicks, line noise, brief filler "uh").
    //   prefix_padding_ms: how much audio BEFORE the energy crossing is
    //     included in the "speech_started" event. 600ms makes sure soft
    //     onsets aren't clipped from the audio the model gets to evaluate.
    //   silence_duration_ms: how long of continuous silence before the
    //     server fires speech_stopped. 1750ms covers the empirically
    //     observed ~1.7s speech_started event delivery latency from
    //     OpenAI's VAD: if the agent resumes speaking after a brief
    //     pause, the resume event lands before our router auto-fires,
    //     so we don't accidentally inject during the next utterance.
    //     Tradeoff: slower end-of-turn detection by ~1s.
    type: 'server_vad',
    create_response: true,
    interrupt_response: true,
    threshold: 0.6,
    prefix_padding_ms: 600,
    silence_duration_ms: 1750
  };
}

// ── Audio helpers ─────────────────────────────────────────────────────────────
function resampleLinear(samples, srIn, srOut) {
  const ratio  = srOut / srIn;
  const outLen = Math.ceil(samples.length * ratio);
  const out    = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos  = i / ratio;
    const idx  = Math.floor(pos);
    const frac = pos - idx;
    out[i] = (samples[idx] || 0) * (1 - frac) + (samples[idx + 1] || 0) * frac;
  }
  return out;
}

// ── Background noise ─────────────────────────────────────────────────────────
async function loadNoiseClips(dir) {
  if (!dir) {
    console.log('[ClipPlayer] No --noiseDir; emitting digital silence between clips.');
    return;
  }
  const abs = path.resolve(process.cwd(), dir);
  if (!fs.existsSync(abs)) {
    console.warn(`[ClipPlayer] --noiseDir ${abs} does not exist; falling back to silence.`);
    return;
  }
  const entries = fs.readdirSync(abs).filter((f) => f.toLowerCase().endsWith('.wav'));
  entries.sort();  // deterministic cycle order — see note below
  let totalSec = 0;
  for (const f of entries) {
    try {
      const buf = fs.readFileSync(path.join(abs, f));
      const decoded = await wavDecoder.decode(buf);
      let samples = decoded.channelData[0];
      if (decoded.sampleRate !== WIRE_SR) {
        samples = resampleLinear(samples, decoded.sampleRate, WIRE_SR);
      }
      const pcm = Buffer.alloc(samples.length * 2);
      for (let i = 0; i < samples.length; i += 1) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        pcm.writeInt16LE(Math.round(s * 32767), i * 2);
      }
      noiseClips.push(pcm);
      totalSec += samples.length / WIRE_SR;
    } catch (e) {
      console.warn(`[ClipPlayer] skip noise clip ${f}: ${e.message}`);
    }
  }
  // Build a name → PCM map (in addition to the legacy `noiseClips`
  // ordered list) so the per-clip `noise_rotation` lookup in
  // `applyNoiseRotation()` can pull buffers by filename. The legacy
  // ordered list is preserved as the fallback rotation when a clip
  // has no `noise_rotation` field (manifests built before this feature).
  for (let i = 0; i < entries.length && i < noiseClips.length; i += 1) {
    noiseByName.set(entries[i], noiseClips[i]);
  }
  // No startup shuffle. The order each rotation visits noise files is
  // determined at BUILD TIME — `applyNoiseRotation(filenames)` swaps
  // the active sequence of PCM buffers used by `pullNoiseFrame()`.
  // - Pre-first-clip noise uses the manifest's `initial_noise_rotation`
  //   (set via `applyNoiseRotation` after manifest load).
  // - After each `injectClip()`, the rotation swaps to that clip's own
  //   `noise_rotation` (sequence baked in at build time, seeded by
  //   clip_id so rebuilds are reproducible).
  // Result: between any two consecutive clips, the bytes of background
  // noise played are identical across runs of the same task.
  console.log(`[ClipPlayer] Loaded ${noiseClips.length} background-noise clips (${totalSec.toFixed(1)}s total) from ${abs}`);
}

// Switch the active noise rotation to a build-time-assigned ordered list
// of noise filenames. Resets position to 0 and clears any partial frame
// buffered from the prior rotation so the swap is bit-exact reproducible.
function applyNoiseRotation(filenames, label) {
  if (!filenames || !filenames.length || noiseClips.length === 0) {
    return;
  }
  const next = [];
  for (const fname of filenames) {
    const buf = noiseByName.get(fname);
    if (buf) next.push(buf);
  }
  if (next.length === 0) return;
  noiseRotation = next;
  noiseClipIdx = 0;
  noiseBuf = Buffer.alloc(0);
  if (label) {
    console.log(`[ClipPlayer] noise_rotation=${label} (${next.length} clips: ${filenames.join(',')})`);
  }
}

// Pull FRAME_BYTES of background noise from the ACTIVE rotation
// (build-time-assigned per clip, or the alphabetical fallback if no
// rotation is set). Returns digital silence if no noise clips were
// loaded at all.
function pullNoiseFrame(frameBytes) {
  const pool = noiseRotation || noiseClips;
  if (pool.length === 0) return Buffer.alloc(frameBytes);
  while (noiseBuf.length < frameBytes) {
    noiseBuf = Buffer.concat([noiseBuf, pool[noiseClipIdx]]);
    noiseClipIdx = (noiseClipIdx + 1) % pool.length;
  }
  const out = noiseBuf.slice(0, frameBytes);
  noiseBuf = noiseBuf.slice(frameBytes);
  return out;
}

async function injectClip(node) {
  // Stamp the moment we DECIDED to inject — useful for downstream timeline
  // alignment. The existing "Injected" log below fires after sync wav
  // decode + Float32→PCM16 conversion (≈100-200ms for a 1-2s clip), so
  // it's a poor anchor for "when did this clip's audio start playing".
  console.log(`[ClipPlayer] inject_start ${node.id}`);
  // Arm the interrupt-flag check: orchSink's RMS VAD watches for the
  // agent transitioning silent→speech within INTERRUPT_FLAG_WINDOW_MS
  // of this inject. If it fires, the clip's id is logged as an
  // interrupt_flag (one-shot per inject via lastInjectFlagged).
  lastInjectAt = Date.now();
  lastInjectClipId = node.id || null;
  lastInjectFlagged = false;
  const clipPath   = dt.resolveClipPath(node);
  const fileBuffer = fs.readFileSync(clipPath);
  const decoded    = await wavDecoder.decode(fileBuffer);
  let samples      = decoded.channelData[0]; // Float32 in [-1, 1]

  if (decoded.sampleRate !== WIRE_SR) {
    samples = resampleLinear(samples, decoded.sampleRate, WIRE_SR);
  }

  const pcm16 = Buffer.alloc(samples.length * 2);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    pcm16.writeInt16LE(Math.round(s * 32767), i * 2);
  }

  downlinkBuf = Buffer.concat([downlinkBuf, pcm16]);
  // Track when the currently-pending audio will finish draining out of the
  // buffer so we can block subsequent clip routings until then. WIRE_SR
  // bytes/sample = 2; total bytes / (WIRE_SR * 2) = duration in seconds.
  const bytesPending = downlinkBuf.length;
  const drainSeconds = bytesPending / (WIRE_SR * 2);
  clipPlaybackEndsAt = Math.max(clipPlaybackEndsAt, Date.now() + drainSeconds * 1000);
  console.log(`[ClipPlayer] Injected ${node.id} (${pcm16.length} bytes @ ${WIRE_SR}Hz)`);

  // Swap to this clip's pre-shuffled noise rotation. Once the clip
  // audio in downlinkBuf drains, the noise frames pulled to fill
  // silence between clips will come from this rotation — making the
  // inter-clip noise byte-identical across runs of the same task.
  if (Array.isArray(node.noise_rotation) && node.noise_rotation.length) {
    applyNoiseRotation(node.noise_rotation, node.id);
  }

  // Arm pre-response suppression. The agent can't respond before the clip
  // finishes playing on their side (they need to hear it first). Any
  // speech_stopped that fires during the clip's playback window must be
  // noise / greeting-tail / VAD artifact, not an agent response — cancel
  // any auto-fired routing response in that window.
  const clipMs = pcm16.length / (WIRE_SR * 2 / 1000);
  awaitingAgentTurn = true;
  suppressAutoResponseUntil = Date.now() + clipMs + POST_CLIP_PAD_MS;
  if (agentResponseTimer) clearTimeout(agentResponseTimer);
  // After the suppression window closes, give the agent up to
  // AGENT_RESPONSE_TIMEOUT_MS to actually start speaking, otherwise fire
  // routing with whatever input the oracle has (will likely fallback).
  agentResponseTimer = setTimeout(() => {
    if (!awaitingAgentTurn) return;
    awaitingAgentTurn = false;
    agentResponseTimer = null;
    console.log(`[ClipPlayer] No agent speech within ${AGENT_RESPONSE_TIMEOUT_MS}ms after clip — firing routing.`);
    safeSendDC({ type: 'response.create' });
  }, clipMs + POST_CLIP_PAD_MS + AGENT_RESPONSE_TIMEOUT_MS);

  return pcm16.length;
}

async function waitForCurrentClipToFinish() {
  const now = Date.now();
  if (clipPlaybackEndsAt <= now) return;
  const wait = clipPlaybackEndsAt - now;
  console.log(`[ClipPlayer] Holding next routing for ${wait}ms — current clip still draining.`);
  await new Promise((r) => setTimeout(r, wait));
}

function getClipDurationMs(node) {
  const clipPath = dt.resolveClipPath(node);
  const buf      = fs.readFileSync(clipPath);
  const decoded  = wavDecoder.decode.sync(buf);
  return Math.ceil((decoded.channelData[0].length / decoded.sampleRate) * 1000);
}

// ── Initial turn (examiner speaks first) ─────────────────────────────────────
// The initial-turn auto-fires the first item's main clip without going
// through the routing oracle (the agent's greeting needs to be answered
// before the model has anything to route on). State transitions to
// (stage=0, item=0) and playedClips records the initial main.
async function doInitialTurn() {
  const initialNode = dt.nodes[dt.initialNodeId];
  if (!initialNode) {
    console.error(`[ClipPlayer] Initial node "${dt.initialNodeId}" not found in tree.`);
    return;
  }
  const parsed = dt.parseClipId(initialNode.id);
  if (parsed) {
    // Same multi-item rule as _handleNextItem: if the initial clip covers
    // multiple items in one breath, advance the cursor to the highest
    // covered so all introduced items are reachable in valid_choices.
    const covered = initialNode.covered_item_indices || [];
    const initItem = covered.length ? Math.max(...covered) : parsed.item;
    dt.advanceTo(parsed.stage, initItem);
  } else {
    dt.advanceTo(0, 0);
  }
  const durationMs = getClipDurationMs(initialNode);
  // Clear any buffered agent audio (the greeting) and delete the greeting
  // conversation item so the routing oracle routes ONLY on what the agent
  // says AFTER this clip plays.
  safeSendDC({ type: 'input_audio_buffer.clear' });
  if (greetingItemId) {
    safeSendDC({ type: 'conversation.item.delete', item_id: greetingItemId });
    console.log(`[ClipPlayer] Deleted greeting conversation item ${greetingItemId}`);
  }
  await injectClip(initialNode);
  // Log a synthetic play(initial) line so build_viz_data can attach a
  // "routed: …" reasoning band to the first customer_speech row in the
  // dashboard. There's no model decision for the initial turn — it's an
  // auto-play triggered by the agent's greeting ending.
  console.log(
    `[ClipPlayer] play(initial) → s${dt.getCurrentStage()} i${dt.getCurrentItem()} ${initialNode.id} | ` +
    `(initial-turn auto-play: customer's first line, fired automatically after the agent's greeting — no routing model decision)`
  );
  dt.markPlayed(initialNode.id);
  // After the clip has played + a small buffer, inject synthetic conversation
  // history so the model knows the current state when VAD fires.
  setTimeout(() => {
    const isGA = true; // Beta API retired; all Realtime models use GA shape
    safeSendDC({ type: 'conversation.item.create', item: {
      type: 'message', role: 'assistant',
      content: [{ type: isGA ? 'output_text' : 'text', text: initialNode.text }]
    }});
    const valid_choices = dt.computeValidChoices();
    safeSendDC({ type: 'conversation.item.create', item: {
      type: 'message', role: 'system',
      content: [{ type: 'input_text', text: JSON.stringify({
        status: 'played',
        item_index: dt.getCurrentItem(),
        last_played: dt.shortenClipId(initialNode.id),
        last_played_text: initialNode.text || '',
        valid_choices,
      }) }]
    }});
    console.log(`[ClipPlayer] Bootstrapped at s${dt.getCurrentStage()} i${dt.getCurrentItem()} (clip ${initialNode.id}). Waiting for examinee response…`);
  }, durationMs + 200);
}

// ── Tool handlers ─────────────────────────────────────────────────────────────

// Detect whether the routing oracle picked a clip from the current node's
// back_references list (i.e. a clip belonging to a previously-handled item
// that the agent regressed to). When true, we play the audio but leave
// flow afterward — that clip's own transitions point at its original
// item's next_main, which would jump the conversation backward if used.
// ── Valid-choices computation ─────────────────────────────────────────────────
// For each turn, compute the set of choices the router can validly pick.
// Choices fall into two categories:
//   - "action": "advance", "yes", "no", "fallback" — abstract actions the
//      adapter resolves at play time
//   - "clip":   a specific clip_id from the current stage's item set
//
// Per-item clip substitution rules (applied after a clip has been played):
//   - main_X played + has _corr that hasn't played → main consumed; _corr available
//   - main_X played + has _post_revision → _post_revision substitutes for main
//     ONLY for the fallback action (replay scenarios). The post_revision clip
//     is NOT exposed in the valid_choices set as a direct routable target —
//     the router should pick next_item / yes / no for confirmations, and
//     fallback for "agent asked to repeat" cases.
//   - main_X played, no sibling → consumed (omitted unless explicitly added)
//   - Same rules apply per resp_N and opt_N
//   - _corr clips appear once their parent has played, ONLY via next_item
//     auto-play (not as direct router targets)
//   - _post_revision clips appear via the fallback action, not as direct
//     entries in valid_choices
// (_siblingByKind, _resolveItemClip, _clipDescription, computeValidChoices
// moved to decision_tree.js — see dt.resolveItemClip / dt.clipDescription /
// dt.computeValidChoices.)
// ── Play action dispatch ──────────────────────────────────────────────────────
async function _emitFunctionCallOutput(callId, payload) {
  // Tool results do NOT include any whisper transcript. The model has
  // real-time audio of the agent's turn; injecting a whisper transcript
  // here would always be one turn STALE (whisper delivery ~1-2s lags
  // the speech_stopped that triggers routing) and creates frame
  // mismatches like the Bella Roma "I don't see teriyaki burger" /
  // "Got it cheeseburger" confusion. A-side transcription stays enabled
  // in the session config purely so the events flow into adapter_A.log
  // for the dashboard's "🎤 A heard" debug band.
  //
  // Also mirror the payload to stdout so the dashboard timeline can
  // attach it to the corresponding routing event and render it on
  // hover — one JSON per line, prefixed with the tag the parser
  // watches for (`[ClipPlayer:tool_response]`).
  console.log(`[ClipPlayer:tool_response] ${JSON.stringify({ call_id: callId, ...payload })}`);
  safeSendDC({
    type: 'conversation.item.create',
    item: {
      type: 'function_call_output',
      call_id: callId,
      output: JSON.stringify(payload),
    },
  });
}
async function _playInjectAndAck(callId, node, statusLabel, extraPayload, onCommit) {
  // `onCommit` (optional) runs AFTER the speaking-check passes and BEFORE
  // the clip is injected. Callers that need to mutate adapter state (e.g.
  // `_handleNextItem` advancing the cursor) pass the mutation here so it
  // only happens when we're committed to playing. Without this, a refused
  // inject would leave the cursor advanced past a clip that never
  // actually played, and the next turn's routing would silently skip it.
  await waitForCurrentClipToFinish();
  if (agentSpeaking) {
    console.log(`[ClipPlayer] REFUSED ${statusLabel} "${node.id || '(synthetic)'}" — agent is currently speaking.`);
    const valid_choices = dt.computeValidChoices();
    await _emitFunctionCallOutput(callId, {
      status: 'not_played_agent_speaking',
      item_index: dt.getCurrentItem(),
      valid_choices,
      note: "Agent was still speaking; the clip was not played and state did not advance. The agent speech that prompted this refused choice is STILL UNANSWERED — include it in your next routing window alongside any newer agent speech. If the agent asked an option / yes-no / clarifier question before this refusal, that question is still pending and your next call should answer it (retry the same clip if still in valid_choices, pick a different listed clip that fits, or `missing_answer` only if no clip can answer).",
    });
    return false;
  }
  if (onCommit) onCommit();
  await injectClip(node);
  // Synthetic yes/no nodes: log them via injectClip but DON'T pollute
  // playedClips (they're abstract actions, not concrete item-flow clips)
  // and DON'T move dt.getLastPlayedClipId() (fallback should target the real
  // most-recent clip).
  if (!node._synthetic && node.id) {
    dt.markPlayed(node.id);
  }
  const valid_choices = dt.computeValidChoices();
  // For main clips only, surface the canonical name(s) + aliases of
  // every item the audio covered (cross-item-merged mains can cover >1
  // item). The agent's tool result thus carries enough info to know
  // which menu item it just heard ordered, even when the customer used
  // a verbal variant ("Bianca pizza" → menu's "White Pizza"). The
  // guidance string tells the routing model to prefer the restatement
  // clip over yes/no when the agent confirms the item — that way the
  // customer re-states the canonical name and the agent gets an
  // unambiguous answer in one turn.
  const playedItemExtras = dt.mainClipItemExtras(node);
  await _emitFunctionCallOutput(callId, {
    status: statusLabel,
    item_index: dt.getCurrentItem(),
    last_played: node.id ? dt.shortenClipId(node.id) : null,
    last_played_text: node.text || '',
    ...playedItemExtras,
    acknowledged: dt.getAckedEntries(),
    unacknowledged: dt.getUnackedEntries(),
    valid_choices,
    ...(extraPayload || {}),
  });
  return true;
}

// For a just-played main clip, return the per-covered-item canonical
// names + aliases plus a guidance string. Non-main clips (opt, resp,
// restatement, yes/no, corr) get an empty object — the canonical-name
// hint only makes sense at the moment the agent is being introduced to
// a new item.
// (_mainClipItemExtras and _pendingCorrection moved to decision_tree.js — see dt.mainClipItemExtras / dt.pendingCorrection.)
async function _handleNextItem(callId, reasoning) {
  // If the last clip had a queued correction (e.g. "I want it iced" then
  // "actually, hot"), play the correction now instead of moving forward.
  // The router doesn't know corrections exist — `next_item` is its only
  // forward-moving lever and the adapter routes appropriately.
  const pendingCorr = dt.pendingCorrection();
  if (pendingCorr) {
    console.log(`[ClipPlayer] play(next_item) → auto-correction ${pendingCorr.id} | ${reasoning}`);
    await _playInjectAndAck(callId, pendingCorr, 'played_correction');
    return;
  }
  // Unack gate: if the agent's move-on comes before every played item
  // has been acknowledged, replay the oldest un-acked item's sub-clip
  // instead of advancing. Same failure mode as premature `thats_all`,
  // just triggered by `next_item`.
  //
  // Prefer a per-item sub-clip of the compound main (so we don't
  // re-say the whole compound). Fall back to substituteForPrematureClose
  // for non-compound cases.
  if (dt.getUnackedCount && dt.getUnackedCount() > 0) {
    const unackedEntries = dt.getUnackedEntries ? dt.getUnackedEntries() : [];
    const first = unackedEntries[0];
    const lastNodeId = dt.getLastPlayedClipId();
    const lastNode = lastNodeId ? dt.nodes[lastNodeId] : null;
    let subNode = null;
    if (first && lastNode && Array.isArray(lastNode.fallback_clips)) {
      const fc = lastNode.fallback_clips.find(
        (x) => x.item_index === first.item_index
      ) || lastNode.fallback_clips[0];
      if (fc) {
        subNode = {
          ...lastNode,
          id: `${lastNodeId}__unack_next_item_i${fc.item_index}`,
          clip: fc.clip,
          text: fc.text || lastNode.text,
        };
      }
    }
    if (!subNode && dt.substituteForPrematureClose) {
      subNode = dt.substituteForPrematureClose();
    }
    if (subNode) {
      console.log(
        `[ClipPlayer:next_item_unack_replay] next_item → ${subNode.id} `
        + `(unacked=${JSON.stringify(dt.getUnackedIds())}) | ${reasoning}`
      );
      await _playInjectAndAck(callId, subNode, 'played_next_item_unack_replay');
      return;
    }
    console.warn(
      `[ClipPlayer:next_item_unack_replay] WARN no_substitution_available; `
      + `advancing despite unacked=${JSON.stringify(dt.getUnackedIds())}`
    );
  }
  let target = null;
  if (dt.getCurrentStage() < 0) {
    target = dt.allItems[0] || null;
  } else {
    target = dt.nextItemAfter(dt.getCurrentStage(), dt.getCurrentItem());
  }
  if (!target) {
    // End-of-order: backend-only close-out. When `next_item` is picked
    // but there's no next item, auto-play the "that's all" clip so the
    // router doesn't have to make a separate close-out decision. The
    // earlier "reject + return invalid_choice" behavior had the router
    // staring at a hint and not always picking it — observed loops of
    // 5+ next_item picks followed by the model wedging entirely. With
    // this we collapse the close-out path: any next_item past the last
    // item plays thats_all directly.
    if (dt.thatsAllClip) {
      console.log(`[ClipPlayer] play(next_item) → auto-thats_all ${dt.thatsAllClip.id} (no more items) | ${reasoning}`);
      await _playInjectAndAck(callId, dt.thatsAllClip, 'played_thats_all');
      return;
    }
    // Fallback for tasks (legacy / pre-#251 trees) where no thats_all
    // sentinel is wired in: surface valid_choices and let the router
    // pick yes/no manually.
    const valid_choices = dt.computeValidChoices();
    console.log(`[ClipPlayer] REJECTED next_item — order complete and no thats_all clip configured; use yes/no close-out. | ${reasoning}`);
    await _emitFunctionCallOutput(callId, {
      status: 'invalid_choice',
      item_index: dt.getCurrentItem(),
      valid_choices,
      note: 'No more items to play and no thats_all clip available. Pick "yes" / "no" depending on whether the agent\'s move-on prompt is yes-framed or no-framed.',
    });
    return;
  }
  const [ns, ni] = target;
  const items = dt.clipsByItem.get(`${ns}.${ni}`) || [];
  const main = items.find((c) => c._kind === 'main');
  if (!main) {
    console.warn(`[ClipPlayer] play(next_item) — no main clip at s${ns} i${ni}; nothing to play.`);
    await _emitFunctionCallOutput(callId, {
      status: 'invalid_state',
      item_index: dt.getCurrentItem(),
      valid_choices: dt.computeValidChoices(),
      note: `No main clip exists at stage ${ns} item ${ni}.`,
    });
    return;
  }
  console.log(`[ClipPlayer] play(next_item) → s${ns} i${ni} ${main.id} | ${reasoning}`);
  // State mutation deferred to the commit callback so a refusal (agent
  // still speaking) leaves the cursor untouched. Without this, observed
  // failure in g7: next_item was refused at s0_i0 but advanced cursor to
  // s0_i1, then a follow-up specific-clip play stayed at i1, and
  // subsequent next_items walked past the dal-makhani item entirely
  // without ever injecting it.
  await _playInjectAndAck(callId, main, 'advanced', undefined, () => {
    // advanceTo handles playedClips reset on stage transition
    dt.advanceTo(ns, dt.getCurrentItem());
    // A multi-item main (customer says "X and Y" in one breath) owns
    // item_index = lowest item but covers every item in its
    // customer_spans. Advance the cursor to the highest covered index
    // so the subsequent valid_choices include every item the clip
    // actually introduced — without this the higher items (whose mains
    // got absorbed by the merger) stay unreachable.
    const covered = main.covered_item_indices || [];
    const maxCovered = covered.length
      ? covered.reduce((a, b) => Math.max(a, b), ni)
      : ni;
    dt.advanceTo(dt.getCurrentStage(), maxCovered);
  });
}
async function _handleYesNo(callId, which, reasoning) {
  const basename = which === 'yes' ? dt.yesClipBasename : dt.noClipBasename;
  if (!basename) {
    await _emitFunctionCallOutput(callId, {
      status: 'invalid_choice',
      item_index: dt.getCurrentItem(),
      valid_choices: dt.computeValidChoices(),
      note: `"${which}" not available — no yes/no clip is configured for this game.`,
    });
    return;
  }
  // Synthetic node: id is recognizable for logging / dashboard timeline
  // alignment, but the node isn't a real manifest clip so we DON'T add it
  // to playedClips and DON'T update dt.getLastPlayedClipId() (fallback should
  // still replay the real most-recent clip, not yes/no).
  const synth = {
    id: `g${dt.gameIndex}_s${dt.getCurrentStage()}_i${dt.getCurrentItem()}_main_${which}`,
    clip: basename,
    text: which === 'yes' ? 'Yes.' : 'No.',
    _synthetic: true,
  };
  console.log(`[ClipPlayer] play(${which}) — ${basename} (${synth.id}) | ${reasoning}`);
  await _playInjectAndAck(callId, synth, `played_${which}`);
}
async function _handleFallback(callId, reasoning) {
  if (!dt.getLastPlayedClipId()) {
    await _emitFunctionCallOutput(callId, {
      status: 'invalid_choice',
      item_index: dt.getCurrentItem(),
      valid_choices: dt.computeValidChoices(),
      note: 'Nothing to fall back to — no clip has been played yet.',
    });
    return;
  }
  const lastNode = dt.nodes[dt.getLastPlayedClipId()];
  let node = lastNode;
  let fbSource = 'replay';
  // Compound mains carry per-item fallback_clips — pick the sub-clip
  // for the oldest unacked item so the fallback targets exactly what
  // the agent missed. Falls back to the first sub-clip when every
  // item's already been ack'd (rare — router usually stops before then).
  const fbClips = (lastNode && lastNode.fallback_clips) || [];
  if (fbClips.length > 0) {
    const unackedIds = new Set(dt.getUnackedIds ? dt.getUnackedIds() : []);
    let pick = null;
    for (const fc of fbClips) {
      const shortId = `i${fc.item_index}_${fc.kind || 'main'}`;
      if (unackedIds.has(shortId)) { pick = fc; break; }
    }
    if (!pick) pick = fbClips[0];
    node = { id: `${dt.getLastPlayedClipId()}__fb_i${pick.item_index}`, clip: pick.clip, text: pick.text || lastNode.text };
    fbSource = `sub_i${pick.item_index}`;
  } else if (lastNode && lastNode.fallback_clip) {
    // Legacy: single fallback_clip (e.g. post-revision sub-slice) still
    // wins over a full replay for non-compound clips.
    node = { id: `${dt.getLastPlayedClipId()}__fb`, clip: lastNode.fallback_clip, text: lastNode.fallback_text || lastNode.text };
    fbSource = 'sub-slice';
  }
  console.log(`[ClipPlayer] play(fallback) — replaying ${dt.getLastPlayedClipId()} (${fbSource}) | ${reasoning}`);
  await _playInjectAndAck(callId, node, 'fallback_played');
}
async function _handleSpecificClip(callId, clipId, reasoning) {
  // Validate against the most recently computed valid_choices set. If we
  // haven't computed one yet (model picked a clip on its first turn), fall
  // through to recomputing now. `clipId` is the short form the model
  // saw (e.g. `i1_main`); `valid[].choice` is also short. The decision
  // tree's `nodes` map is keyed by the LONG form, so re-attach the
  // `g{gameIndex}_s{currentStage}_` prefix before the node lookup.
  const valid = dt.getLastValidChoices() || dt.computeValidChoices();
  const shortId = dt.shortenClipId(clipId);
  const match = valid.find((v) => v.kind === 'clip' && v.choice === shortId);
  if (!match) {
    console.log(`[ClipPlayer] REJECTED play("${clipId}") — not in valid_choices. | ${reasoning}`);
    await _emitFunctionCallOutput(callId, {
      status: 'invalid_choice',
      item_index: dt.getCurrentItem(),
      valid_choices: valid,
      note: `"${clipId}" is not in the current valid_choices. Pick one of the listed options.`,
    });
    return;
  }
  const longId = dt.expandClipId(shortId, dt.gameIndex, dt.getCurrentStage());
  let node = dt.nodes[longId];
  if (!node) {
    await _emitFunctionCallOutput(callId, {
      status: 'invalid_choice',
      item_index: dt.getCurrentItem(),
      valid_choices: valid,
      note: `"${clipId}" is in valid_choices but not in the tree — adapter bug.`,
    });
    return;
  }
  // Premature-close guard: if the router picks `thats_all` while
  // any ackable clip is still un-ack'd, substitute the oldest
  // un-ack'd item's last restatement (or its own main clip) in
  // place of the close-out. Gives the customer one more turn to
  // force the ack before the conversation ends.
  let statusLabel = 'played';
  if (node._kind === 'thats_all' && dt.getUnackedCount() > 0) {
    const sub = dt.substituteForPrematureClose();
    if (sub) {
      console.log(
        `[ClipPlayer:premature_close] thats_all → ${sub.id} `
        + `(unacked=${JSON.stringify(dt.getUnackedIds())})`
      );
      node = sub;
      statusLabel = 'played_premature_close_substitution';
    } else {
      console.warn(
        `[ClipPlayer:premature_close] WARN no_substitution_available; `
        + `playing thats_all despite unacked=${JSON.stringify(dt.getUnackedIds())}`
      );
    }
  }
  // Compound-main fallback substitution: if the router picks a
  // compound main that's already been played (i.e. this is a replay
  // fallback), swap the compound wav for one of its per-item sub-clips
  // so the customer doesn't re-say the whole compound. Prefer the
  // oldest unacked item; fall back to the first sub-clip.
  if (
    node.fallback_clips
    && Array.isArray(node.fallback_clips)
    && node.fallback_clips.length > 0
    && dt.hasPlayed(node.id)
  ) {
    const unackedIds = new Set(dt.getUnackedIds ? dt.getUnackedIds() : []);
    let pick = null;
    for (const fc of node.fallback_clips) {
      const s = `i${fc.item_index}_${fc.kind || 'main'}`;
      if (unackedIds.has(s)) { pick = fc; break; }
    }
    if (!pick) pick = node.fallback_clips[0];
    node = {
      ...node,
      id: `${node.id}__fb_i${pick.item_index}`,
      clip: pick.clip,
      text: pick.text || node.text,
    };
    console.log(
      `[ClipPlayer] play("${clipId}") — compound replay swapped to sub_i${pick.item_index} `
      + `(${pick.canonical_name || ''}) | ${reasoning}`
    );
  } else {
    console.log(`[ClipPlayer] play("${clipId}") | ${reasoning}`);
  }
  // The cursor never decreases on a specific pick. If the routing model
  // picks a clip for an EARLIER item (e.g. a corr for item 1 after a
  // multi-item main already covered items 0-2 and advanced the cursor to
  // 2), keeping the cursor at the higher index preserves access to all
  // already-introduced items in valid_choices. The cursor still advances
  // FORWARD if the picked clip belongs to a later item in the same stage,
  // which catches the original regression case (stale cursor at an item
  // we haven't actually engaged with yet). Deferred to the commit
  // callback so a refusal leaves the cursor untouched.
  const parsed = dt.parseClipId(node.id || longId);
  await _playInjectAndAck(callId, node, statusLabel, undefined, () => {
    if (!parsed) return;
    if (parsed.stage !== dt.getCurrentStage()) {
      dt.advanceTo(parsed.stage, parsed.item);
      return;
    }
    dt.advanceTo(dt.getCurrentStage(), Math.max(dt.getCurrentItem(), parsed.item));
  });
}

async function _checkAgentSubmitted() {
  // Returns true if the agent has already called submit_order (the
  // tool server flips its /api/status `submitted` flag the moment the
  // call succeeds). The customer always has a `thats_all` clip at end
  // of order, so the conversation would otherwise keep replaying
  // thats_all + yes/no on every move-on prompt until duration timeout.
  // Once the agent submits, there is nothing left to measure, so the
  // clipPlayer should exit cleanly.
  if (!TOOL_SERVER_URL) return false;
  try {
    const r = await fetch(new URL('/api/status', TOOL_SERVER_URL).toString(),
      { method: 'GET' });
    if (!r.ok) return false;
    const j = await r.json();
    return j && j.submitted === true;
  } catch (_e) {
    return false;  // network errors aren't a reason to give up the conversation
  }
}

async function handlePlay(callId, args) {
  // If the agent has already finalized the cart, the conversation is
  // effectively over; bail out before invoking the route logic. Mirrors
  // the existing auto-end exit hook so the orchestrator/runner sees A
  // close gracefully and moves on to scoring.
  if (await _checkAgentSubmitted()) {
    if (!dt.isConversationEnded()) {
      dt.endConversation();
      console.log('[ClipPlayer] agent submitted via submit_order — exiting.');
      await _emitFunctionCallOutput(callId, {
        status: 'conversation_ended',
        item_index: dt.getCurrentItem(),
        note: 'Agent already called submit_order; clipPlayer exiting.',
      });
      setTimeout(() => { process.exit(0); }, 1500);
    }
    return;
  }
  const choice = (args.choice || '').trim();
  const heard = (args.heard || '').trim();
  const reasoningRaw = (args.reasoning || '').trim();
  // Ingest the router's acknowledgement report BEFORE any choice
  // handling — every subsequent state read (valid_choices, tool
  // response acknowledged/unacknowledged, thats_all substitution)
  // reflects the ack list this turn. Ack ingestion is deliberately
  // side-effecting even for invalid choices: the agent's utterance
  // that produced this bad choice still ack'd items, and refusing
  // the choice shouldn't erase that.
  const rawAck = Array.isArray(args.acknowledged) ? args.acknowledged : [];
  if (rawAck.length > 0) {
    const ackResult = dt.recordAcks(rawAck);
    const acceptedShort = ackResult.accepted.map(a => a.short);
    const rejectedShort = ackResult.rejected.map(r => `${r.raw}:${r.reason}`);
    console.log(
      `[ClipPlayer:acks] accepted=${JSON.stringify(acceptedShort)} `
      + `rejected=${JSON.stringify(rejectedShort)} `
      + `unacked_after=${JSON.stringify(dt.getUnackedIds())}`
    );
  }
  // Fold `heard` into the reasoning string used by all the per-choice
  // log lines below, so every play(...) log entry exposes both signals
  // without having to thread a second arg through 7 handler functions.
  // The structured field is still in the raw response.done event for
  // any downstream parser that wants it independently.
  const reasoning = heard
    ? `heard="${heard}" | ${reasoningRaw}`
    : reasoningRaw;
  // Per-turn debug log — emits the exact tool-call args + the
  // valid_choices the model was looking at (compact form: just
  // {choice, kind}) when it made this decision. Pair with the once-
  // per-session `[ClipPlayer:rendered_system_prompt]` line to
  // reconstruct exactly what the model saw at each turn. Used to
  // debug router decisions like "model picked missing_answer when
  // just_the_defaults was exposed".
  const _vcCompact = (dt.getLastValidChoices() || []).map((v) => ({
    choice: v.choice, kind: v.kind,
  }));
  console.log(
    `[ClipPlayer:router_turn] ${JSON.stringify({
      item_index: dt.getCurrentItem(),
      tool_call: { choice, heard, reasoning: reasoningRaw },
      valid_choices: _vcCompact,
      jtd_exposed: _vcCompact.some((v) => v.choice === 'just_the_defaults'),
      missing_answer_exposed: _vcCompact.some((v) => v.choice === 'missing_answer'),
      default_eligible_options_text: DEFAULT_ELIGIBLE_OPTIONS_TEXT,
    })}`
  );
  if (!choice) {
    await _emitFunctionCallOutput(callId, {
      status: 'invalid_choice',
      item_index: dt.getCurrentItem(),
      valid_choices: dt.computeValidChoices(),
      note: 'Missing `choice`. Pick one of the valid_choices.',
    });
    return;
  }
  // Interrupt-recovery gate: when the most recent inject was interrupted
  // by the agent (interrupt_flag fired), refuse the next play choice
  // outright. Wait for either the agent to finish that off-script speech
  // and start a fresh turn (cleared by the VAD onset handler when
  // agentSilentSinceInterrupt is set) OR for INTERRUPT_RECOVERY_TIMEOUT_MS
  // to elapse. Playing another clip mid-interrupt would compound the
  // misalignment — the agent's interrupting speech didn't hear the
  // previous clip cleanly, so they're already off-script; we want them
  // to re-engage before we feed more audio.
  if (interruptRecoveryAt) {
    const elapsed = Date.now() - interruptRecoveryAt;
    if (elapsed < INTERRUPT_RECOVERY_TIMEOUT_MS) {
      console.log(
        `[ClipPlayer] REFUSED play "${choice}" — interrupt_recovery_pending `
        + `(elapsed_ms=${elapsed}, timeout_ms=${INTERRUPT_RECOVERY_TIMEOUT_MS}, `
        + `agent_silent_since=${agentSilentSinceInterrupt})`
      );
      const valid_choices = dt.computeValidChoices();
      await _emitFunctionCallOutput(callId, {
        status: 'not_played_interrupt_recovery',
        item_index: dt.getCurrentItem(),
        valid_choices,
        note: (
          "Previous clip was interrupted by agent speech (the agent started "
          + "talking before they had heard the clip). Refusing to play another "
          + "clip until either (a) the agent finishes this off-script turn and "
          + `starts a new response, or (b) ${INTERRUPT_RECOVERY_TIMEOUT_MS}ms `
          + `elapses (already ${elapsed}ms in). State did not advance and the `
          + "agent speech that prompted this refused choice remains UNANSWERED — "
          + "carry it forward into the next routing window. The interrupted "
          + "clip ALSO never fully landed, so the question it was supposed to "
          + "answer is still pending too. Re-route on the next turn; answer "
          + "the oldest still-pending question first."
        ),
      });
      return;
    }
    // Timed out — clear the gate and proceed.
    console.log(
      `[ClipPlayer] interrupt_recovery_clear via_timeout `
      + `elapsed_ms=${elapsed}`
    );
    interruptRecoveryAt = 0;
    agentSilentSinceInterrupt = false;
  }
  if (choice === 'next_item' || choice === 'advance') return _handleNextItem(callId, reasoning);
  if (choice === 'yes' || choice === 'no') return _handleYesNo(callId, choice, reasoning);
  if (choice === 'fallback') return _handleFallback(callId, reasoning);
  if (choice === 'missing_answer') return _handleMissingAnswer(callId, reasoning);
  if (choice === 'just_the_defaults') return _handleJustTheDefaults(callId, reasoning);
  if (choice === 'first_option' || choice === 'second_option') {
    return _handlePositionalOption(callId, choice, reasoning);
  }
  return _handleSpecificClip(callId, choice, reasoning);
}

async function _handlePositionalOption(callId, which, reasoning) {
  // Global multi-choice fallback: "the first option" / "the second option".
  // Mirrors _handleJustTheDefaults — synthetic node so we don't touch
  // playedClips / lastPlayedClipId. Always available (no gating).
  const basename = which === 'first_option'
    ? dt.firstOptionClipBasename
    : dt.secondOptionClipBasename;
  if (!basename) {
    await _emitFunctionCallOutput(callId, {
      status: 'invalid_choice',
      item_index: dt.getCurrentItem(),
      valid_choices: dt.computeValidChoices(),
      note: `"${which}" not available — shared TTS clip missing. Run build_tts_synth.py.`,
    });
    return;
  }
  const text = which === 'first_option' ? 'The first option.' : 'The second option.';
  const synth = {
    id: `g${dt.gameIndex}_s${dt.getCurrentStage()}_i${dt.getCurrentItem()}_${which}`,
    clip: basename,
    text,
    _synthetic: true,
  };
  console.log(
    `[ClipPlayer] play(${which}) — ${basename} (${synth.id}) | ${reasoning}`
  );
  await _playInjectAndAck(callId, synth, `played_${which}`);
}

async function _handleJustTheDefaults(callId, reasoning) {
  if (!dt.justTheDefaultsClipBasename) {
    await _emitFunctionCallOutput(callId, {
      status: 'invalid_choice',
      item_index: dt.getCurrentItem(),
      valid_choices: dt.computeValidChoices(),
      note: 'just_the_defaults not available — task has no default-eligible options.',
    });
    return;
  }
  // Synthetic node like _handleYesNo: surfaces a recognizable id for
  // logging + dashboard timeline alignment, but not a real manifest clip
  // so we don't update playedClips / dt.getLastPlayedClipId().
  const synth = {
    id: `g${dt.gameIndex}_s${dt.getCurrentStage()}_i${dt.getCurrentItem()}_just_the_defaults`,
    clip: dt.justTheDefaultsClipBasename,
    text: 'Just the defaults.',
    _synthetic: true,
  };
  console.log(
    `[ClipPlayer] play(just_the_defaults) — ${dt.justTheDefaultsClipBasename} ` +
    `(${synth.id}) | ${reasoning}`
  );
  await _playInjectAndAck(callId, synth, 'played_just_the_defaults');
}

async function _handleMissingAnswer(callId, reasoning) {
  // Record the (stage, item) the agent's unanswerable question pertained
  // to. This log line is the canonical signal for downstream metrics:
  // the most-recent inject before this event is the customer turn that
  // elicited an on-topic question we couldn't answer. Format is grep-
  // friendly: keys are bare identifiers, values are unquoted ints, and
  // the reasoning is the trailing text.
  console.log(
    `[ClipPlayer] missing_answer stage_index=${dt.getCurrentStage()} ` +
    `item_index=${dt.getCurrentItem()} | ${reasoning}`
  );
  // End-of-order escape: if missing_answer fires with no remaining
  // items to advance to (and no pending correction), the adapter has
  // nothing left to play — sitting silent would let the SIGTERM
  // backstop finalize the cart non-deterministically. Submit on the
  // customer's behalf here; this is the ONLY path that fires customer-
  // side auto-submit (it has no standalone action — the only trigger
  // is missing_answer with nothing left to advance to).
  const pendingCorr = dt.pendingCorrection();
  const nextItem = dt.nextItemAfter(dt.getCurrentStage(), dt.getCurrentItem());
  if (!pendingCorr && !nextItem) {
    return _customerAutoSubmitAndExit(callId, reasoning);
  }
  // Otherwise advance via the same logic as next_item — same cursor
  // mutation, same refusal-while-agent-speaking guard.
  return _handleNextItem(callId, reasoning);
}

// Auto-end one-shot guard. Once we've initiated the customer_auto submit
// path, subsequent play tool calls (the model may fire one more before
// the process exits) should not re-trigger the POST.
let autoEnded = false;

async function _customerAutoSubmitAndExit(callId, reasoning) {
  if (autoEnded) {
    await _emitFunctionCallOutput(callId, {
      status: 'conversation_ended',
      item_index: dt.getCurrentItem(),
      note: 'Conversation already ended; process is shutting down.',
    });
    return;
  }
  // Mirror the playGuarded gate: don't submit while the agent is mid-
  // utterance, otherwise we'd discard whatever they were about to say.
  await waitForCurrentClipToFinish();
  if (agentSpeaking) {
    console.log("[ClipPlayer] REFUSED end-of-order auto-submit — agent is currently speaking.");
    const valid_choices = dt.computeValidChoices();
    await _emitFunctionCallOutput(callId, {
      status: 'not_played_agent_speaking',
      item_index: dt.getCurrentItem(),
      valid_choices,
      note: "Agent was still speaking; missing_answer end-of-order auto-submit deferred. Re-route on the next turn once the agent's full utterance is available.",
    });
    return;
  }
  autoEnded = true;
  console.log(`[ClipPlayer] missing_answer at end-of-order — auto-submitting on customer's behalf | ${reasoning}`);

  if (TOOL_SERVER_URL) {
    try {
      const url = new URL('/api/tool', TOOL_SERVER_URL);
      const body = JSON.stringify({
        tool: 'submit_order',
        args: {
          source: 'customer_auto',
          reasoning: (reasoning || '').toString(),
        },
      });
      await new Promise((resolve) => {
        const http = require('http');
        const req = http.request({
          method: 'POST',
          hostname: url.hostname,
          port: url.port,
          path: url.pathname,
          headers: {
            'content-type': 'application/json',
            'content-length': Buffer.byteLength(body),
          },
        }, (res) => {
          res.resume();
          res.on('end', resolve);
        });
        req.on('error', (e) => {
          console.warn(`[ClipPlayer] auto-submit POST failed: ${e.message}`);
          resolve();
        });
        req.write(body);
        req.end();
      });
      console.log('[ClipPlayer] auto-submit posted to tool server.');
    } catch (e) {
      console.warn(`[ClipPlayer] auto-submit POST exception: ${e.message}`);
    }
  } else {
    console.warn('[ClipPlayer] no --toolServerUrl set; skipping auto-submit POST. The eval will rely on the tool server\'s SIGTERM finalize for cart capture.');
  }

  await _emitFunctionCallOutput(callId, {
    status: 'conversation_ended',
    item_index: dt.getCurrentItem(),
    note: (
      'missing_answer at end-of-order: no remaining clips to play. ' +
      'The customer adapter submitted the cart on the customer\'s behalf ' +
      '(finalize_reason=customer_auto_submit). The conversation is over.'
    ),
  });

  setTimeout(() => {
    console.log('[ClipPlayer] exiting after auto-end.');
    process.exit(0);
  }, 1500);
}

async function handleToolCall(callId, toolName, argsStr) {
  let args;
  try {
    args = JSON.parse(argsStr);
  } catch (e) {
    console.error(`[ClipPlayer] Failed to parse tool args: ${argsStr}`);
    return;
  }
  if (toolName === 'play') return handlePlay(callId, args);
  // Backwards-compat shims for older system prompts: redirect to play.
  if (toolName === 'select_clip') {
    return handlePlay(callId, { choice: args.node_id, reasoning: args.reasoning });
  }
  if (toolName === 'select_fallback') {
    return handlePlay(callId, { choice: 'fallback', reasoning: args.reason });
  }
  console.warn(`[ClipPlayer] Unknown tool: ${toolName}`);
}

// ── Main ──────────────────────────────────────────────────────────────────────
(async function main() {
  console.log(`[ClipPlayer] Starting — tree="${dt.taskId}" initial="${dt.initialNodeId}" vad=${VAD_MODE}`);
  await loadNoiseClips(argv.noiseDir);
  // Seed the noise rotation for the pre-first-clip window from the
  // tree's `initial_noise_rotation` (assigned at build time by
  // build_customer_clips). Older trees without this field fall back
  // to the alphabetical noiseClips list inside pullNoiseFrame().
  if (Array.isArray(dt.initialNoiseRotation) && dt.initialNoiseRotation.length) {
    applyNoiseRotation(dt.initialNoiseRotation, 'initial');
  }
  await connectOrchestrator();
  await connectOpenAI();
  startBridgePacers();
})();

// ── Orchestrator leg ──────────────────────────────────────────────────────────
async function connectOrchestrator() {
  orchPC     = new wrtc.RTCPeerConnection();
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
      if (count++ < 5) console.log(`[ClipPlayer] orchSink first bytesIn=${view.length}`);
      uplinkBuf = Buffer.concat([uplinkBuf, view]);

      // Realtime energy-based VAD on B's audio (what A receives from
      // the orchestrator). Drives the agentSpeaking gate that the inject
      // path checks, and surfaces interrupt-flag log lines when the
      // agent transitions silent→speech within the post-inject window.
      let sumSq = 0;
      for (let i = 0; i < samples.length; i++) sumSq += samples[i] * samples[i];
      const rms = Math.sqrt(sumSq / Math.max(1, samples.length));
      const now = Date.now();
      if (rms >= AGENT_VAD_RMS_INT16) {
        if (!agentSpeaking) {
          agentSpeaking = true;
          // Onset: if the agent started speaking within the
          // post-inject window for the most recent clip, the clip
          // overlapped agent speech — flag it. Playback is NOT
          // cancelled; we want the eval to see how the agent
          // recovers from being interrupted.
          if (
            lastInjectAt
            && !lastInjectFlagged
            && (now - lastInjectAt) <= INTERRUPT_FLAG_WINDOW_MS
            && lastInjectClipId
          ) {
            const deltaMs = now - lastInjectAt;
            lastInjectFlagged = true;
            console.log(
              `[ClipPlayer] interrupt_flag clip_id=${lastInjectClipId} `
              + `delta_ms=${deltaMs} `
              + `reason=agent_started_within_${INTERRUPT_FLAG_WINDOW_MS}ms_of_inject`
            );
            // Arm the interrupt-recovery gate: the next play() will be
            // refused until either the agent finishes this interrupting
            // turn AND starts a new one, or INTERRUPT_RECOVERY_TIMEOUT_MS
            // elapses.
            interruptRecoveryAt = now;
            agentSilentSinceInterrupt = false;
          } else if (interruptRecoveryAt && agentSilentSinceInterrupt) {
            // Recovery condition met: agent went silent after the
            // interrupt and is now speaking again (a fresh turn that
            // presumably reflects having heard the clip's full audio).
            console.log(
              `[ClipPlayer] interrupt_recovery_clear via_agent_turn `
              + `elapsed_ms=${now - interruptRecoveryAt}`
            );
            interruptRecoveryAt = 0;
            agentSilentSinceInterrupt = false;
          }
        }
        lastAgentNonSilentAt = now;
      } else if (agentSpeaking && (now - lastAgentNonSilentAt) >= AGENT_VAD_HANGOFF_MS) {
        agentSpeaking = false;
        // First silence after an interrupt — mark recovery as awaiting
        // the next agent-turn onset. (See the interrupt_recovery_clear
        // branch in the onset handler above.)
        if (interruptRecoveryAt) {
          agentSilentSinceInterrupt = true;
        }
      }
    };
  };

  ws = new WebSocket(SIGNAL_URL);
  ws.on('error', (err) => console.error(`[ClipPlayer] Orchestrator WS error:`, err));
  await new Promise((res) => ws.once('open', res));

  orchPC.onicecandidate = ({ candidate }) => {
    if (!candidate) return;
    ws.send(JSON.stringify({
      role: ROLE, type: 'candidate', candidate: {
        candidate: candidate.candidate,
        sdpMid: candidate.sdpMid,
        sdpMLineIndex: candidate.sdpMLineIndex
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
      console.log(`[ClipPlayer] Orchestrator answer applied.`);
    } else if (data.type === 'candidate' && data.role === ROLE) {
      try { await orchPC.addIceCandidate(data.candidate); } catch { }
    }
  });
}

// ── OpenAI leg ────────────────────────────────────────────────────────────────
async function connectOpenAI() {
  const session = await postJSON(`${TOKEN_SERVER_URL}/session`, { model: MODEL });
  const EPHEMERAL_KEY = session?.client_secret?.value;
  if (!EPHEMERAL_KEY) throw new Error('Ephemeral key missing from token server response.');

  aiPC     = new wrtc.RTCPeerConnection();
  aiSource = new RTCAudioSource({ sampleRate: WIRE_SR, channelCount: 1 });
  const aiOutTrack = aiSource.createTrack();
  aiPC.addTrack(aiOutTrack);

  // Receive the audio track but do NOT route it to downlinkBuf —
  // with tool_choice:'required' the model never generates audio.
  aiPC.ontrack = ({ track }) => {
    if (track.kind !== 'audio') return;
    if (aiSink) aiSink.stop();
    aiSink = new RTCAudioSink(track, { sampleRate: WIRE_SR, channelCount: 1, bitDepth: 16 });
    // Intentionally not forwarding model audio to downlinkBuf
  };

  aiDC = aiPC.createDataChannel('oai-events');

  aiDC.onopen = () => {
    console.log(`[ClipPlayer] OpenAI data channel open. VAD mode: ${VAD_MODE}`);
    // GA models require `session.type` on every session.update. The Beta
    // path was retired in late 2025 — all Realtime models now use GA shape.
    const isGA = true;
    const sessionUpdate = isGA ? {
      type: 'realtime',
      instructions: generateSystemPrompt(dt),
      audio: {
        input: { transcription: { model: 'whisper-1' }, turn_detection: buildVadConfig(VAD_MODE) },
        output: { voice: VOICE },
      },
      tools: [PLAY_TOOL],
      tool_choice: 'required',
    } : {
      instructions: generateSystemPrompt(dt),
      voice: VOICE,
      turn_detection: buildVadConfig(VAD_MODE),
      input_audio_transcription: { model: 'whisper-1' },
      tools: [PLAY_TOOL],
      tool_choice: 'required',
    };
    safeSendDC({ type: 'session.update', session: sessionUpdate });
  };

  aiDC.onmessage = (ev) => {
    try {
      if (argv.logEvents) console.log(`[ClipPlayer] ← event`, ev.data);
      const evt = JSON.parse(ev.data);

      switch (evt?.type) {
        case 'session.updated':
          if (!bootstrapSent) {
            bootstrapSent = true;
            // Don't fire the initial clip immediately — wait until the
            // examinee has greeted (their VAD-detected speech has stopped).
            // Fall back to firing after `initialDelayMs` if the agent stays
            // silent (some models won't speak first).
            console.log(`[ClipPlayer] Armed — waiting for examinee greeting (input_audio_buffer.speech_stopped) or ${argv.initialDelayMs}ms fallback timeout.`);
            initialFallbackTimer = setTimeout(() => {
              if (!initialTurnFired) {
                initialTurnFired = true;
                console.log(`[ClipPlayer] No greeting heard within ${argv.initialDelayMs}ms — playing initial clip on fallback.`);
                doInitialTurn();
              }
            }, argv.initialDelayMs);
          }
          break;

        case 'input_audio_buffer.speech_started':
          // NOTE: agentSpeaking is driven exclusively by the RMS VAD on
          // orchSink (the B-audio A receives). We intentionally do NOT
          // touch agentSpeaking here — A's session VAD on the uplink
          // direction is laggy and unreliable for this purpose. The
          // OpenAI event is still useful as a bootstrap signal for
          // the examinee's greeting.
          //
          // Cancel the post-clip "no agent speech" fallback timer.
          // Purpose of the timer (set in injectClip) is to handle the
          // case where the agent NEVER starts speaking — a deadlock
          // guard. Once speech_started fires the agent IS speaking, so
          // we should wait for the natural speech_stopped to drive
          // routing instead of forcing it mid-utterance. Without this
          // cancellation, the 8 s ceiling can expire DURING the agent's
          // continuous speech (observed: agent says "Okay, let's get
          // the crust choice... garlic crust or regular crust?" — timer
          // fires before "garlic crust or regular crust?" is committed,
          // routing model sees only the first half and picks
          // next_item).
          if (awaitingAgentTurn && agentResponseTimer) {
            clearTimeout(agentResponseTimer);
            agentResponseTimer = null;
          }
          if (bootstrapSent && !initialTurnFired) {
            // Agent has started their greeting. Cancel the fallback timer so
            // we wait however long the greeting takes (no more racing).
            if (initialFallbackTimer) {
              clearTimeout(initialFallbackTimer);
              initialFallbackTimer = null;
            }
            // Capture the conversation item id for this greeting so we can
            // delete it AFTER the clip plays — preventing the routing oracle
            // from later treating the greeting as the agent's response to
            // the clip.
            greetingItemId = evt.item_id || null;
            console.log(`[ClipPlayer] Detected examinee speech start (greeting, item_id=${greetingItemId}). Will fire initial clip when they stop.`);
          }
          break;

        case 'input_audio_buffer.speech_stopped':
          // agentSpeaking is driven by the RMS VAD on orchSink — see
          // the matching note on speech_started above.
          if (bootstrapSent && !initialTurnFired) {
            initialTurnFired = true;
            if (initialFallbackTimer) {
              clearTimeout(initialFallbackTimer);
              initialFallbackTimer = null;
            }
            // server_vad auto-fires response.create on speech_stopped. We
            // need to cancel that auto-response before the routing oracle
            // routes on the agent's greeting (which is pre-clip and not a
            // valid input for any transition).
            console.log('[ClipPlayer] Examinee greeting finished — canceling auto-response, then playing initial clip.');
            safeSendDC({ type: 'response.cancel' });
            doInitialTurn();
            break;
          }

          // Suppress auto-response while we're inside the post-inject
          // window where the agent physically can't have responded yet
          // (greeting tail, VAD artifacts, etc.). After the window, the
          // first speech_stopped is the agent's actual response — let its
          // auto-response.create proceed.
          if (awaitingAgentTurn && Date.now() < suppressAutoResponseUntil) {
            const remainingMs = suppressAutoResponseUntil - Date.now();
            console.log(`[ClipPlayer] Suppressing pre-agent auto-response (still ${remainingMs}ms inside post-clip window).`);
            safeSendDC({ type: 'response.cancel' });
            break;
          }
          if (awaitingAgentTurn) {
            awaitingAgentTurn = false;
            if (agentResponseTimer) {
              clearTimeout(agentResponseTimer);
              agentResponseTimer = null;
            }
          }
          break;

        // (We deliberately do NOT handle
        // 'conversation.item.input_audio_transcription.completed' here.
        // The events still fire and land in adapter_A.log — build_viz_data
        // parses them there for the dashboard's "🎤 A heard" band — but
        // we don't feed any transcript into the routing model, since
        // whisper delivery is too laggy to be timely on the current turn.)

        case 'response.created':
          // If a routing decision is already in flight, cancel this new
          // response — VAD auto-fires response.create on every speech_stopped,
          // which can stack up secondary responses that pile on top of the
          // pending tool call. We want one routing decision per agent turn.
          if (routingResponseInFlight) {
            console.log('[ClipPlayer] Suppressing concurrent response.create (routing already in flight).');
            safeSendDC({ type: 'response.cancel' });
          } else {
            routingResponseInFlight = true;
          }
          break;

        case 'response.done': {
          // Free the flag once the model finishes its response (the
          // function_call_output we sent in handlePlay is what
          // completes the response loop).
          routingResponseInFlight = false;
          // Recovery: when server VAD cancels the response mid-stream
          // (e.g. agent resumed talking before the router LLM finished
          // its play() function call), we MUST send a synthetic
          // function_call_output back to the LLM. Otherwise the LLM
          // sees its in-progress tool call hanging in conversation
          // history with no result and stops trying — it silently
          // emits commentary text ("still working", "please hold on")
          // for the remainder of the run until SIGTERM. Observed on
          // Robin Salmi 02920A03.g1: response.done arrived with
          // status="cancelled" reason="turn_detected" after the 4th
          // play() call was killed by the agent's barge-in. The
          // router went silent for 2 minutes 7 seconds, no clips
          // injected, no error. With the synthetic
          // function_call_output below, the LLM's tool call gets
          // resolved with status="cancelled_retry" and a fresh
          // response.create wakes it up to try again.
          const resp = evt.response || {};
          if (resp.status === 'cancelled') {
            const reason = resp.status_details?.reason || 'unknown';
            const incompleteCalls = (resp.output || []).filter(
              o => o?.type === 'function_call' && o?.status !== 'completed'
            );
            for (const fc of incompleteCalls) {
              console.log(
                `[ClipPlayer] router_response_cancelled `
                + `call_id=${fc.call_id} name=${fc.name} `
                + `reason=${reason} — sending synthetic `
                + `cancelled_retry function_call_output`
              );
              safeSendDC({
                type: 'conversation.item.create',
                item: {
                  type: 'function_call_output',
                  call_id: fc.call_id,
                  output: JSON.stringify({
                    status: 'cancelled_retry',
                    reason,
                    note: (
                      'Your previous play() call was cancelled by the '
                      + 'server (the agent resumed speaking before your '
                      + 'response completed). Re-evaluate the agent\'s '
                      + 'latest utterance and call play() again with a '
                      + 'fresh choice from the valid_choices list below.'
                    ),
                    valid_choices: dt.getLastValidChoices() || [],
                  }),
                },
              });
              // Clear stale streaming buffers — handleToolCall on the
              // dead call would post the cancelled args back as if it
              // succeeded.
              if (pendingCallId === fc.call_id) {
                pendingCallId = null;
                pendingCallName = null;
                pendingArgsBuf = '';
              }
            }
            if (incompleteCalls.length > 0) {
              // Kick the LLM to make a new decision. Without this the
              // server waits for the next agent VAD event, which may
              // never come (or comes wrapped in another cancellation).
              safeSendDC({ type: 'response.create' });
              routingResponseInFlight = true;
            }
          }
          break;
        }

        case 'response.output_item.added':
          if (evt.item?.type === 'function_call') {
            pendingCallId   = evt.item.call_id;
            pendingCallName = evt.item.name;
            pendingArgsBuf  = '';
          }
          break;

        case 'response.function_call_arguments.delta':
          pendingArgsBuf += evt.delta || '';
          break;

        case 'response.function_call_arguments.done':
          if (pendingCallName) {
            handleToolCall(pendingCallId, pendingCallName, pendingArgsBuf);
            pendingCallId   = null;
            pendingCallName = null;
            pendingArgsBuf  = '';
          }
          break;
      }
    } catch { }
  };

  aiDC.onerror = (e) => console.error(`[ClipPlayer] datachannel error`, e);

  const offer = await aiPC.createOffer({ offerToReceiveAudio: true, offerToReceiveVideo: false });
  await aiPC.setLocalDescription(offer);
  await waitForIceGatheringComplete(aiPC);

  // Beta API retired in late 2025; all models now use the GA /calls path.
  const sdpAnswer = await postSDP(
    `https://api.openai.com/v1/realtime/calls?model=${encodeURIComponent(MODEL)}`,
    EPHEMERAL_KEY,
    aiPC.localDescription.sdp
  );
  await aiPC.setRemoteDescription({ type: 'answer', sdp: sdpAnswer });
  console.log(`[ClipPlayer] OpenAI answer applied.`);
}

function safeSendDC(obj) {
  if (!aiDC || aiDC.readyState !== 'open') return;
  try { aiDC.send(JSON.stringify(obj)); } catch (e) { console.error('DC send error', e); }
}

// ── Bridge pacers ─────────────────────────────────────────────────────────────
function startBridgePacers() {
  if (uplinkPacer)   clearInterval(uplinkPacer);
  if (downlinkPacer) clearInterval(downlinkPacer);
  if (stallWatchdog) clearInterval(stallWatchdog);

  // Stall watchdog: catches the case where the router LLM stopped
  // calling play() for any reason (e.g. cascading cancellations the
  // single-shot cancelled_retry recovery didn't cover, or the LLM
  // got stuck in a commentary loop). When clips remain to play and
  // we haven't injected one in STALL_WATCHDOG_THRESHOLD_MS, log a
  // warning and force a fresh response.create to wake the LLM up.
  // Throttled so repeated warnings don't spam the log.
  stallWatchdog = setInterval(() => {
    if (autoEnded) return;
    if (!lastInjectAt) return;  // not started yet
    const sinceInject = Date.now() - lastInjectAt;
    if (sinceInject < STALL_WATCHDOG_THRESHOLD_MS) return;
    const sinceWarn = Date.now() - stallWatchdogLastWarnAt;
    if (sinceWarn < STALL_WATCHDOG_THRESHOLD_MS) return;
    stallWatchdogLastWarnAt = Date.now();
    console.log(
      `[ClipPlayer] STALL ${Math.round(sinceInject / 1000)}s since `
      + `last inject (${lastInjectClipId || '?'}); last play() call `
      + `in flight=${routingResponseInFlight}; forcing response.create `
      + `to wake the router LLM. If this fires repeatedly the LLM is `
      + `stuck and the eval will SIGTERM at duration limit.`
    );
    // Cancel any in-flight response so create_response below isn't
    // suppressed by the in-flight guard.
    if (routingResponseInFlight) {
      safeSendDC({ type: 'response.cancel' });
      routingResponseInFlight = false;
    }
    safeSendDC({ type: 'response.create' });
    routingResponseInFlight = true;
  }, STALL_WATCHDOG_INTERVAL_MS);

  const FRAME_10MS = Math.floor(WIRE_SR * 0.01) * 2; // bytes at WIRE_SR PCM16 mono

  // Examinee audio → OpenAI (VAD + Whisper transcription)
  uplinkPacer = setInterval(() => {
    for (let i = 0; i < 10; i++) {
      let out;
      if (uplinkBuf.length >= FRAME_10MS) {
        out       = uplinkBuf.slice(0, FRAME_10MS);
        uplinkBuf = uplinkBuf.slice(FRAME_10MS);
      } else if (uplinkBuf.length > 0) {
        out = Buffer.alloc(FRAME_10MS);
        uplinkBuf.copy(out, 0, 0, uplinkBuf.length);
        uplinkBuf = Buffer.alloc(0);
      } else {
        out = Buffer.alloc(FRAME_10MS); // silence
      }
      const ab = new ArrayBuffer(FRAME_10MS);
      new Uint8Array(ab).set(out);
      aiSource.onData({ samples: new Int16Array(ab), sampleRate: WIRE_SR, bitsPerSample: 16, channelCount: 1 });
    }
  }, 100);

  // Clip audio → Orchestrator (plays pre-recorded clips to examinee).
  // When no clip is queued, fill with cycled background noise so the
  // examinee never hears total silence (which it might interpret as a
  // dropped channel) — matches the ambient room tone from the source
  // session.
  let _downlinkPacerFirstTickLogged = false;
  downlinkPacer = setInterval(() => {
    if (!_downlinkPacerFirstTickLogged) {
      // The first tick is the moment A's pacer first emits frames into
      // orchSource. This is the wall-clock anchor downstream tooling
      // can use: A.wav is recorded by the orchestrator's sink which has
      // been firing for WebRTC silence/comfort-noise frames since the
      // track was negotiated, so sink(A)#1 lands BEFORE this moment by
      // however long the pacer took to set up.
      _downlinkPacerFirstTickLogged = true;
      console.log(`[ClipPlayer] downlinkPacer_first_tick`);
    }
    for (let i = 0; i < 10; i++) {
      let out;
      if (downlinkBuf.length >= FRAME_10MS) {
        // Full clip frame — clip playback takes priority over noise.
        out         = downlinkBuf.slice(0, FRAME_10MS);
        downlinkBuf = downlinkBuf.slice(FRAME_10MS);
      } else if (downlinkBuf.length > 0) {
        // Partial clip frame: clip ends mid-frame. Pad with noise (not
        // silence) so the transition into noise is seamless.
        out = Buffer.alloc(FRAME_10MS);
        downlinkBuf.copy(out, 0, 0, downlinkBuf.length);
        const tail = pullNoiseFrame(FRAME_10MS - downlinkBuf.length);
        tail.copy(out, downlinkBuf.length);
        downlinkBuf = Buffer.alloc(0);
      } else {
        // No clip queued — pull from the background-noise cycle.
        out = pullNoiseFrame(FRAME_10MS);
      }
      const ab = new ArrayBuffer(FRAME_10MS);
      new Uint8Array(ab).set(out);
      orchSource.onData({ samples: new Int16Array(ab), sampleRate: WIRE_SR, bitsPerSample: 16, channelCount: 1 });
    }
  }, 100);
}

// ── Utils ─────────────────────────────────────────────────────────────────────
async function postJSON(url, body) {
  const r = await _fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  if (!r.ok) throw new Error(`POST ${url} ${r.status}`);
  return await r.json();
}

async function postSDP(url, bearer, sdp) {
  const r = await _fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/sdp', 'Authorization': `Bearer ${bearer}` },
    body: sdp
  });
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
    setTimeout(() => { pc.removeEventListener('icegatheringstatechange', check); resolve(); }, 1500);
  });
}

async function shutdown() {
  console.log(`[ClipPlayer] Shutting down…`);
  try { if (uplinkPacer)   clearInterval(uplinkPacer);   } catch { }
  try { if (downlinkPacer) clearInterval(downlinkPacer); } catch { }
  try { if (stallWatchdog) clearInterval(stallWatchdog); } catch { }
  try { if (orchSink) orchSink.stop(); } catch { }
  try { if (aiSink)   aiSink.stop();   } catch { }
  try { if (orchPC) orchPC.close(); } catch { }
  try { if (aiPC)   aiPC.close();   } catch { }
  try { if (ws && ws.readyState === WebSocket.OPEN) ws.close(); } catch { }
  process.exit(0);
}
