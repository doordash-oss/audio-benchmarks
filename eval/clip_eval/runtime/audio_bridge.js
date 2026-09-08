#!/usr/bin/env node
// Copyright 2026 DoorDash, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * Local two-party WebRTC audio bridge for benchmark roles A and B.
 *
 * Each role offers one bidirectional mono audio track over `/signal`. Incoming
 * audio is recorded to `<role>.wav` and forwarded to the other role at a fixed
 * real-time cadence. `/tap` exposes the received PCM stream to optional
 * diagnostics. SIGINT/SIGTERM finalizes the recordings and creates a stereo
 * `combined.wav` (A left, B right) when ffmpeg is available.
 */

'use strict';

require('dotenv').config();

const fs = require('fs');
const http = require('http');
const path = require('path');
const { spawn } = require('child_process');
const WebSocket = require('ws');
const wrtc = require('wrtc');
const wav = require('wav');

const { RTCAudioSink, RTCAudioSource } = wrtc.nonstandard;
const SAMPLE_RATE = Number.parseInt(process.env.WIRE_SAMPLE_RATE || '48000', 10);
const PORT = Number.parseInt(process.env.SIGNAL_PORT || '9100', 10);
const OUTPUT_DIR = path.resolve(process.env.RECORD_DIR || 'outputs');
const FRAME_SAMPLES = Math.floor(SAMPLE_RATE / 100); // 10 ms
const FRAME_BYTES = FRAME_SAMPLES * 2; // mono PCM16

const roles = ['A', 'B'];
const peers = new Map();
const sockets = new Set();
const taps = { A: new Set(), B: new Set() };
let shuttingDown = false;

function now() {
  return new Date().toISOString();
}

function log(...parts) {
  console.log(`[${now()}] [audio-bridge]`, ...parts);
}

function emptyRoleState() {
  return {
    offer: null,
    pendingCandidates: [],
    pc: null,
    source: null,
    sink: null,
    writer: null,
    queue: Buffer.alloc(0),
    pacer: null,
    remoteDescriptionSet: false,
    speech: false,
    loudFrames: 0,
    quietFrames: 0,
    receivedFrames: 0,
    // Wall-clock time of this role's first real audio byte (the sink(<role>)#1
    // moment) — each role's own wav writer starts at sample 0 right then, not
    // at some shared instant, so this is what makeStereoMix() needs to align
    // the two files onto one wall-clock timeline instead of naively assuming
    // both start at the same time.
    firstByteAt: null,
  };
}

for (const role of roles) peers.set(role, emptyRoleState());
fs.mkdirSync(OUTPUT_DIR, { recursive: true });

function validRole(role) {
  return role === 'A' || role === 'B';
}

function otherRole(role) {
  return role === 'A' ? 'B' : 'A';
}

function sendToRole(role, payload) {
  const encoded = JSON.stringify({ role, ...payload });
  for (const socket of sockets) {
    if (socket.readyState === WebSocket.OPEN) socket.send(encoded);
  }
}

function pcmRms(samples) {
  if (!samples.length) return 0;
  let sum = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const value = samples[i];
    sum += value * value;
  }
  return Math.sqrt(sum / samples.length);
}

function observeSpeech(role, samples) {
  const state = peers.get(role);
  const loud = pcmRms(samples) >= 184; // approximately -45 dBFS
  if (loud) {
    state.loudFrames += 1;
    state.quietFrames = 0;
    if (!state.speech && state.loudFrames >= 3) {
      state.speech = true;
      log(`audio_onset role=${role}`);
    }
  } else {
    state.quietFrames += 1;
    state.loudFrames = 0;
    if (state.speech && state.quietFrames >= 20) {
      state.speech = false;
      log(`audio_offset role=${role}`);
    }
  }
}

function appendAudio(targetRole, bytes) {
  const target = peers.get(targetRole);
  target.queue = target.queue.length
    ? Buffer.concat([target.queue, bytes])
    : Buffer.from(bytes);
}

function nextFrame(state) {
  if (state.queue.length >= FRAME_BYTES) {
    const frame = state.queue.subarray(0, FRAME_BYTES);
    state.queue = state.queue.subarray(FRAME_BYTES);
    return frame;
  }
  const frame = Buffer.alloc(FRAME_BYTES);
  if (state.queue.length) state.queue.copy(frame);
  state.queue = Buffer.alloc(0);
  return frame;
}

function startPacer(role) {
  const state = peers.get(role);
  if (state.pacer) return;
  // Drop whatever piled up in this role's forwarding queue while its OWN
  // connection was still coming up. `appendAudio` starts queuing the other
  // role's audio the moment THAT role connects, regardless of whether this
  // role is ready yet — so a slow-to-connect role's queue accumulates a
  // real-time backlog for the entire connection-setup delay. A fixed-cadence
  // 10ms drain can never catch up a backlog (it only ever plays it back 1:1),
  // so without this reset the role hears audio stale by exactly its own
  // startup delay for the rest of the session, not just its first turn.
  state.queue = Buffer.alloc(0);
  state.pacer = setInterval(() => {
    if (!state.source) return;
    const frame = nextFrame(state);
    const samples = new Int16Array(
      frame.buffer.slice(frame.byteOffset, frame.byteOffset + frame.byteLength),
    );
    state.source.onData({
      samples,
      sampleRate: SAMPLE_RATE,
      bitsPerSample: 16,
      channelCount: 1,
    });
  }, 10);
}

function createWriter(role) {
  return new wav.FileWriter(path.join(OUTPUT_DIR, `${role}.wav`), {
    sampleRate: SAMPLE_RATE,
    channels: 1,
    bitDepth: 16,
  });
}

function attachIncomingTrack(role, track) {
  const state = peers.get(role);
  if (state.sink) state.sink.stop();
  if (!state.writer) state.writer = createWriter(role);
  state.sink = new RTCAudioSink(track, {
    sampleRate: SAMPLE_RATE,
    channelCount: 1,
    bitDepth: 16,
  });
  state.sink.ondata = ({ samples }) => {
    if (shuttingDown) return;
    const bytes = Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength);
    if (state.firstByteAt == null) state.firstByteAt = Date.now();
    state.receivedFrames += 1;
    if (state.receivedFrames <= 12) {
      log(`sink(${role})#${state.receivedFrames} bytesIn=${bytes.length}`);
    }
    state.writer.write(bytes);
    appendAudio(otherRole(role), bytes);
    observeSpeech(role, samples);
    for (const tap of taps[role]) {
      if (tap.readyState === WebSocket.OPEN) tap.send(bytes);
    }
  };
  log(`incoming audio ready role=${role}`);
  if (roles.every((candidate) => peers.get(candidate).sink)) {
    log('Both peers ready—streaming and recording.');
  }
}

async function addCandidate(role, candidate) {
  const state = peers.get(role);
  if (!candidate?.candidate) return;
  if (!state.pc || !state.remoteDescriptionSet) {
    state.pendingCandidates.push(candidate);
    return;
  }
  try {
    await state.pc.addIceCandidate(candidate);
  } catch (error) {
    log(`candidate rejected role=${role}: ${error.message}`);
  }
}

async function establishPeer(role) {
  const state = peers.get(role);
  if (!state.offer || state.pc) return;

  const pc = new wrtc.RTCPeerConnection({ iceServers: [] });
  const source = new RTCAudioSource({ sampleRate: SAMPLE_RATE, channelCount: 1 });
  state.pc = pc;
  state.source = source;
  pc.addTrack(source.createTrack());
  pc.ontrack = ({ track }) => {
    if (track.kind === 'audio') attachIncomingTrack(role, track);
  };
  pc.onicecandidate = ({ candidate }) => {
    if (!candidate) return;
    sendToRole(role, {
      type: 'candidate',
      candidate: {
        candidate: candidate.candidate,
        sdpMid: candidate.sdpMid,
        sdpMLineIndex: candidate.sdpMLineIndex,
      },
    });
  };
  pc.onconnectionstatechange = () => log(`connection role=${role} state=${pc.connectionState}`);

  await pc.setRemoteDescription({ type: 'offer', sdp: state.offer });
  state.remoteDescriptionSet = true;
  for (const candidate of state.pendingCandidates.splice(0)) {
    await addCandidate(role, candidate);
  }
  const answer = await pc.createAnswer();
  await pc.setLocalDescription(answer);
  sendToRole(role, { type: 'answer', sdp: pc.localDescription.sdp });
  startPacer(role);
  log(`answer sent role=${role}`);
}

async function handleSignal(raw) {
  let message;
  try {
    message = JSON.parse(raw.toString());
  } catch {
    return;
  }
  if (!validRole(message.role)) return;
  const state = peers.get(message.role);
  if (message.type === 'offer' && typeof message.sdp === 'string') {
    state.offer = message.sdp;
    await establishPeer(message.role);
  } else if (message.type === 'candidate') {
    await addCandidate(message.role, message.candidate);
  }
}

function closeRole(role) {
  const state = peers.get(role);
  if (state.pacer) clearInterval(state.pacer);
  if (state.sink) {
    try { state.sink.stop(); } catch {}
  }
  if (state.pc) {
    try { state.pc.close(); } catch {}
  }
  state.pacer = null;
  state.sink = null;
  state.pc = null;
  state.source = null;
}

function finishWriter(role) {
  const writer = peers.get(role).writer;
  if (!writer) return Promise.resolve();
  return new Promise((resolve) => {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      resolve();
    };
    // `wav.FileWriter` emits `done` only after its underlying stream has
    // flushed and it has sought back to replace the placeholder RIFF sizes.
    // The generic Writable `end` callback fires earlier and therefore is not
    // a safe signal for readers such as ffmpeg or the metrics pipeline.
    writer.once('done', finish);
    writer.once('error', finish);
    writer.end();
    setTimeout(() => {
      log(`writer finalization timed out role=${role}`);
      finish();
    }, 5000).unref();
  });
}

// Each role's wav writer starts at sample 0 whenever THAT role's own first
// byte arrives (firstByteAt), not at one shared instant — A and B routinely
// differ by several seconds (A is local/instant, B waits on its own
// pipeline/provider session to come up). Naively merging A.wav+B.wav by
// sample index (as ffmpeg's amerge does with no further input) silently
// drops that gap, making B's audio play back several seconds EARLIER,
// relative to A, than it actually happened in wall-clock time. Returns
// each role's lag (ms) behind whichever role connected first, so the
// caller can pad the lagging channel with real silence before merging.
function computeStartOffsetsMs() {
  const firstByteAt = {};
  for (const role of roles) firstByteAt[role] = peers.get(role).firstByteAt;
  const knownTimes = roles.map((r) => firstByteAt[r]).filter((t) => t != null);
  const anchor = knownTimes.length ? Math.min(...knownTimes) : 0;
  const offsets = {};
  for (const role of roles) {
    offsets[role] = firstByteAt[role] != null
      ? Math.max(0, firstByteAt[role] - anchor)
      : 0;
  }
  return offsets;
}

function makeStereoMix() {
  const left = path.join(OUTPUT_DIR, 'A.wav');
  const right = path.join(OUTPUT_DIR, 'B.wav');
  const output = path.join(OUTPUT_DIR, 'combined.wav');
  if (!fs.existsSync(left) || !fs.existsSync(right)) {
    log('mix skipped because A.wav or B.wav is missing');
    return Promise.resolve();
  }
  const pcmDuration = (filename) => Math.max(0, fs.statSync(filename).size - 44)
    / (SAMPLE_RATE * 2);
  const offsetsMs = computeStartOffsetsMs();
  const offsetA = Math.round(offsetsMs.A);
  const offsetB = Math.round(offsetsMs.B);
  log(`combined.wav alignment: offsetA=${offsetA}ms offsetB=${offsetB}ms`);
  const outputDuration = Math.max(
    pcmDuration(left) + offsetA / 1000,
    pcmDuration(right) + offsetB / 1000,
  );
  return new Promise((resolve) => {
    const ffmpeg = spawn('ffmpeg', [
      '-y', '-i', left, '-i', right,
      '-filter_complex',
      `[0:a]adelay=delays=${offsetA}[a0d];[1:a]adelay=delays=${offsetB}[a1d];`
      + '[a0d]apad[a0];[a1d]apad[a1];[a0][a1]amerge=inputs=2,pan=stereo|c0<c0|c1<c1[aout]',
      '-map', '[aout]', '-ac', '2', '-t', outputDuration.toFixed(6), output,
    ], { stdio: ['ignore', 'ignore', 'pipe'] });
    let diagnostic = '';
    ffmpeg.stderr.on('data', (chunk) => {
      if (diagnostic.length < 8000) diagnostic += chunk.toString();
    });
    ffmpeg.on('error', (error) => {
      log(`mix skipped: ${error.message}`);
      resolve();
    });
    ffmpeg.on('close', (code) => {
      if (code === 0) log(`wrote ${output}`);
      else log(`ffmpeg exited ${code}: ${diagnostic.slice(-1000)}`);
      resolve();
    });
  });
}

async function reset() {
  for (const role of roles) closeRole(role);
  for (const role of roles) {
    const writer = peers.get(role).writer;
    if (writer) {
      try { writer.end(); } catch {}
    }
    peers.set(role, emptyRoleState());
  }
  log('state reset');
}

const server = http.createServer(async (request, response) => {
  if (request.method === 'GET' && request.url === '/healthz') {
    response.writeHead(200, { 'content-type': 'text/plain' });
    response.end('ok');
    return;
  }
  if (request.method === 'POST' && request.url === '/reset') {
    await reset();
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end('{"success":true}');
    return;
  }
  response.writeHead(404);
  response.end();
});

const signalServer = new WebSocket.Server({ noServer: true });
const tapServer = new WebSocket.Server({ noServer: true });

server.on('upgrade', (request, socket, head) => {
  const pathname = new URL(request.url, 'http://localhost').pathname;
  const target = pathname === '/signal' ? signalServer : pathname === '/tap' ? tapServer : null;
  if (!target) {
    socket.destroy();
    return;
  }
  target.handleUpgrade(request, socket, head, (client) => target.emit('connection', client));
});

signalServer.on('connection', (socket) => {
  sockets.add(socket);
  socket.send('{"type":"ready"}');
  socket.on('message', (raw) => {
    handleSignal(raw).catch((error) => log(`signaling error: ${error.stack || error}`));
  });
  socket.on('close', () => sockets.delete(socket));
  socket.on('error', (error) => log(`signal socket error: ${error.message}`));
});

tapServer.on('connection', (socket) => {
  let role = null;
  socket.on('message', (raw) => {
    try {
      const message = JSON.parse(raw.toString());
      if (message.type === 'subscribe' && validRole(message.role)) {
        if (role) taps[role].delete(socket);
        role = message.role;
        taps[role].add(socket);
      }
    } catch {}
  });
  const remove = () => {
    if (role) taps[role].delete(socket);
  };
  socket.on('close', remove);
  socket.on('error', remove);
});

async function shutdown(signal) {
  if (shuttingDown) return;
  shuttingDown = true;
  log(`received ${signal}`);
  for (const role of roles) closeRole(role);
  await Promise.all(roles.map(finishWriter));
  await makeStereoMix();
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 1000).unref();
}

process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('uncaughtException', (error) => {
  console.error(`[${now()}] [audio-bridge] uncaught exception`, error);
  shutdown('uncaughtException');
});
process.on('unhandledRejection', (error) => {
  console.error(`[${now()}] [audio-bridge] unhandled rejection`, error);
});

server.listen(PORT, () => log(`listening on port ${PORT}`));
