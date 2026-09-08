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
 * Native OpenAI Realtime Role B adapter for the Order Up benchmark.
 *
 * This is a clean implementation of the repository's Role B protocol. It
 * connects the local benchmark audio bridge to OpenAI Realtime directly over
 * WebSocket, without Pipecat or code from the former bundled harness:
 *
 *   audio bridge WebRTC (48 kHz PCM) <-> this adapter <-> OpenAI WS (24 kHz PCM)
 *
 * stdin accepts configure_tools/inject JSON lines. Tool calls are forwarded to
 * the benchmark tool server, and stdout emits the provider-neutral transcript
 * and agent_stop events used by the runtime and visualizer.
 */

'use strict';

require('dotenv').config();

const readline = require('readline');
const WebSocket = require('ws');
const wrtc = require('wrtc');

const { RTCAudioSink, RTCAudioSource } = wrtc.nonstandard;
const WIRE_SAMPLE_RATE = 48000;
const OPENAI_SAMPLE_RATE = 24000;
const PACER_MS = 10;
const OPENAI_FRAME_SAMPLES = OPENAI_SAMPLE_RATE * PACER_MS / 1000;
const LOGGED_EVENT_TYPES = new Set([
  'session.created',
  'session.updated',
  'response.created',
  'response.done',
  'response.function_call_arguments.done',
  'conversation.item.input_audio_transcription.completed',
  'input_audio_buffer.speech_started',
  'input_audio_buffer.speech_stopped',
  'error',
]);

function isoNow() {
  return new Date().toISOString();
}

function log(message, fields) {
  const suffix = fields === undefined ? '' : ` ${JSON.stringify(fields)}`;
  process.stderr.write(`[${isoNow()}] [openai-native] ${message}${suffix}\n`);
}

function viz(kind, fields = {}) {
  process.stderr.write(`[${isoNow()}] [roleB-viz] ${kind} ${JSON.stringify(fields)}\n`);
}

function emit(payload) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}

function parseArgs(argv) {
  const parsed = {
    role: null,
    signalUrl: null,
    tokenServer: null,
    model: process.env.OPENAI_REALTIME_MODEL || 'gpt-realtime-2',
    voice: 'shimmer',
    systemPrompt: '',
    turnMode: 'server_vad',
    vadMode: 'slow',
    speakFirst: false,
    enableTranscription: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--speakFirst') parsed.speakFirst = true;
    else if (arg === '--enableTranscription') parsed.enableTranscription = true;
    else if (arg.startsWith('--')) {
      const key = arg.slice(2);
      if (Object.prototype.hasOwnProperty.call(parsed, key)) {
        parsed[key] = argv[index + 1];
        index += 1;
      } else if (argv[index + 1] && !argv[index + 1].startsWith('--')) {
        log(`ignoring unknown argument ${arg}`);
        index += 1;
      }
    }
  }
  if (parsed.role !== 'B') throw new Error('--role must be B');
  if (!parsed.signalUrl) throw new Error('--signalUrl is required');
  return parsed;
}

function resamplePcm16(samples, fromRate, toRate) {
  if (fromRate === toRate) return new Int16Array(samples);
  if (!samples.length) return new Int16Array(0);
  const outputLength = Math.max(1, Math.round(samples.length * toRate / fromRate));
  const output = new Int16Array(outputLength);
  const scale = fromRate / toRate;
  for (let i = 0; i < outputLength; i += 1) {
    const position = i * scale;
    const left = Math.min(samples.length - 1, Math.floor(position));
    const right = Math.min(samples.length - 1, left + 1);
    const fraction = position - left;
    output[i] = Math.round(samples[left] + (samples[right] - samples[left]) * fraction);
  }
  return output;
}

function int16ToBuffer(samples) {
  return Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength);
}

class ByteQueue {
  constructor() {
    this.chunks = [];
    this.offset = 0;
    this.length = 0;
  }

  push(bytes) {
    if (!bytes.length) return;
    this.chunks.push(Buffer.from(bytes));
    this.length += bytes.length;
  }

  take(count) {
    const output = Buffer.alloc(count);
    let written = 0;
    while (written < count && this.chunks.length) {
      const head = this.chunks[0];
      const available = head.length - this.offset;
      const copied = Math.min(count - written, available);
      head.copy(output, written, this.offset, this.offset + copied);
      written += copied;
      this.offset += copied;
      this.length -= copied;
      if (this.offset === head.length) {
        this.chunks.shift();
        this.offset = 0;
      }
    }
    return { bytes: output, available: written };
  }
}

function realtimeTools(tools) {
  return (tools || []).map((tool) => ({
    type: 'function',
    name: tool.name,
    description: tool.description || '',
    parameters: tool.parameters || { type: 'object', properties: {} },
  }));
}

function turnDetection(vadMode, turnMode) {
  if (turnMode === 'none') return null;
  const fast = String(vadMode || '').startsWith('fast');
  return fast
    ? {
      type: 'server_vad',
      threshold: 0.6,
      prefix_padding_ms: 1200,
      silence_duration_ms: 450,
      create_response: true,
      interrupt_response: false,
    }
    : {
      type: 'server_vad',
      threshold: 0.9,
      create_response: true,
      interrupt_response: true,
    };
}

function sessionUpdate(args, tools) {
  const input = {
    format: { type: 'audio/pcm', rate: OPENAI_SAMPLE_RATE },
    turn_detection: turnDetection(args.vadMode, args.turnMode),
  };
  if (args.enableTranscription) input.transcription = { model: 'gpt-4o-transcribe' };
  return {
    type: 'session.update',
    session: {
      type: 'realtime',
      output_modalities: ['audio'],
      instructions: args.systemPrompt || undefined,
      audio: {
        input,
        output: {
          format: { type: 'audio/pcm', rate: OPENAI_SAMPLE_RATE },
          voice: args.voice || 'shimmer',
        },
      },
      tools: realtimeTools(tools),
      tool_choice: 'auto',
    },
  };
}

function safeEventForLog(event) {
  if (!event || typeof event !== 'object') return event;
  if (event.type === 'session.created' || event.type === 'session.updated') {
    return {
      type: event.type,
      event_id: event.event_id,
      session: {
        id: event.session?.id,
        model: event.session?.model,
        output_modalities: event.session?.output_modalities,
      },
    };
  }
  if (event.type === 'response.done') {
    return {
      type: event.type,
      event_id: event.event_id,
      response: {
        id: event.response?.id,
        status: event.response?.status,
        output_types: (event.response?.output || []).map((item) => item.type),
        usage: event.response?.usage,
      },
    };
  }
  return event;
}

function shouldLogEvent(type) {
  return LOGGED_EVENT_TYPES.has(type);
}

class NativeOpenAIAdapter {
  constructor(args, dependencies = {}) {
    this.args = args;
    this.WebSocket = dependencies.WebSocket || WebSocket;
    this.fetch = dependencies.fetch || globalThis.fetch;
    this.pc = null;
    this.signalWs = null;
    this.modelWs = null;
    this.source = null;
    this.sink = null;
    this.pacer = null;
    this.output = new ByteQueue();
    this.inputChunks = [];
    this.inputBytes = 0;
    this.tools = [];
    this.toolServerUrl = null;
    this.toolsConfigured = false;
    this.sessionCreated = false;
    this.sessionConfigured = false;
    this.greeted = false;
    this.shuttingDown = false;
    this.audioTurn = null;
    this.responses = new Map();
  }

  sendModel(payload) {
    if (this.modelWs?.readyState !== this.WebSocket.OPEN) return false;
    this.modelWs.send(JSON.stringify(payload));
    return true;
  }

  async connectSignal() {
    this.pc = new wrtc.RTCPeerConnection({ iceServers: [] });
    this.source = new RTCAudioSource({ sampleRate: WIRE_SAMPLE_RATE, channelCount: 1 });
    this.pc.addTrack(this.source.createTrack());
    this.pc.ontrack = ({ track }) => {
      if (track.kind !== 'audio' || this.sink) return;
      this.sink = new RTCAudioSink(track);
      this.sink.ondata = ({ samples, sampleRate }) => this.handleInputAudio(samples, sampleRate);
      log('customer audio track ready');
    };
    this.pc.onicecandidate = ({ candidate }) => {
      if (!candidate || this.signalWs?.readyState !== this.WebSocket.OPEN) return;
      this.signalWs.send(JSON.stringify({
        role: 'B',
        type: 'candidate',
        candidate: {
          candidate: candidate.candidate,
          sdpMid: candidate.sdpMid,
          sdpMLineIndex: candidate.sdpMLineIndex,
        },
      }));
    };
    this.pc.onconnectionstatechange = () => log(`bridge connection=${this.pc.connectionState}`);

    await new Promise((resolve, reject) => {
      this.signalWs = new this.WebSocket(this.args.signalUrl);
      this.signalWs.once('open', resolve);
      this.signalWs.once('error', reject);
    });
    this.signalWs.on('message', (raw) => this.handleSignal(raw));
    const offer = await this.pc.createOffer();
    await this.pc.setLocalDescription(offer);
    this.signalWs.send(JSON.stringify({
      role: 'B', type: 'offer', sdp: this.pc.localDescription.sdp,
    }));
    log('bridge offer sent');
  }

  async handleSignal(raw) {
    let message;
    try { message = JSON.parse(raw.toString()); } catch { return; }
    if (message.role !== 'B') return;
    try {
      if (message.type === 'answer') {
        await this.pc.setRemoteDescription({ type: 'answer', sdp: message.sdp });
        log('bridge answer applied');
      } else if (message.type === 'candidate' && message.candidate?.candidate) {
        await this.pc.addIceCandidate(new wrtc.RTCIceCandidate(message.candidate));
      }
    } catch (error) {
      log(`bridge signaling error: ${error.message}`);
    }
  }

  connectModel() {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) throw new Error('OPENAI_API_KEY is not set');
    const url = `wss://api.openai.com/v1/realtime?model=${encodeURIComponent(this.args.model)}`;
    this.modelWs = new this.WebSocket(url, { headers: { Authorization: `Bearer ${apiKey}` } });
    this.modelWs.on('open', () => log(`OpenAI Realtime connected model=${this.args.model}`));
    this.modelWs.on('message', (raw) => this.handleModelEvent(raw));
    this.modelWs.on('error', (error) => log(`OpenAI WebSocket error: ${error.message}`));
    this.modelWs.on('close', (code, reason) => {
      log(`OpenAI WebSocket closed code=${code} reason=${reason.toString()}`);
      if (!this.shuttingDown) this.shutdown('model websocket closed');
    });
  }

  maybeConfigureSession() {
    if (!this.sessionCreated || !this.toolsConfigured || this.sessionConfigured) return;
    this.sessionConfigured = this.sendModel(sessionUpdate(this.args, this.tools));
    if (this.sessionConfigured) log(`session.update sent tools=${this.tools.length}`);
  }

  handleInputAudio(samples, sampleRate) {
    if (!this.sessionConfigured || this.modelWs?.readyState !== this.WebSocket.OPEN) return;
    const downsampled = resamplePcm16(samples, sampleRate || WIRE_SAMPLE_RATE, OPENAI_SAMPLE_RATE);
    const bytes = int16ToBuffer(downsampled);
    this.inputChunks.push(bytes);
    this.inputBytes += bytes.length;
    // Batch 20 ms per WebSocket event. This avoids framework overhead without
    // adding a meaningful delay to server VAD.
    const target = OPENAI_FRAME_SAMPLES * 2 * 2;
    if (this.inputBytes < target) return;
    const payload = Buffer.concat(this.inputChunks, this.inputBytes);
    this.inputChunks = [];
    this.inputBytes = 0;
    this.sendModel({ type: 'input_audio_buffer.append', audio: payload.toString('base64') });
  }

  startPacer() {
    if (this.pacer) return;
    this.pacer = setInterval(() => {
      if (!this.source) return;
      const { bytes, available } = this.output.take(OPENAI_FRAME_SAMPLES * 2);
      const pcm24 = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
      const pcm48 = resamplePcm16(pcm24, OPENAI_SAMPLE_RATE, WIRE_SAMPLE_RATE);
      this.source.onData({
        samples: pcm48,
        sampleRate: WIRE_SAMPLE_RATE,
        bitsPerSample: 16,
        channelCount: 1,
      });
      if (available > 0 && this.audioTurn) this.audioTurn.playedSamples += available / 2;
      this.finishAudioTurnIfReady();
    }, PACER_MS);
  }

  startAudioTurn(responseId) {
    if (!this.audioTurn) {
      this.audioTurn = {
        responseIds: new Set(),
        doneResponseIds: new Set(),
        playedSamples: 0,
        transcripts: [],
      };
      viz('agent_start');
    }
    this.audioTurn.responseIds.add(responseId);
  }

  finishAudioTurnIfReady() {
    if (!this.audioTurn || this.output.length > 0) return;
    const allResponsesDone = [...this.audioTurn.responseIds]
      .every((responseId) => this.audioTurn.doneResponseIds.has(responseId));
    if (!allResponsesDone) return;
    const turn = this.audioTurn;
    this.audioTurn = null;
    const transcript = turn.transcripts.filter(Boolean).join(' ').trim();
    if (transcript) {
      viz('agent_transcript', {
        text: transcript,
        chars: transcript.length,
        words: transcript.split(/\s+/).filter(Boolean).length,
      });
      if (this.args.enableTranscription) {
        emit({ type: 'transcript', source: 'response', text: transcript });
      }
    }
    const audioDuration = turn.playedSamples / OPENAI_SAMPLE_RATE;
    emit({ type: 'event', event: 'agent_stop' });
    viz('agent_stop', {
      audio_duration_s: Math.round(audioDuration * 1000) / 1000,
      audio_frames: turn.playedSamples,
    });
  }

  async executeTool(event) {
    let args;
    try { args = JSON.parse(event.arguments || '{}'); } catch { args = {}; }
    let result;
    try {
      if (!this.toolServerUrl) throw new Error('configure_tools did not provide serverUrl');
      const response = await this.fetch(`${this.toolServerUrl}/api/tool`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ name: event.name, arguments: args }),
      });
      const body = await response.json();
      result = body.result ?? body;
    } catch (error) {
      result = { ok: false, error: error.message };
    }
    this.sendModel({
      type: 'conversation.item.create',
      item: {
        type: 'function_call_output',
        call_id: event.call_id,
        output: JSON.stringify(result),
      },
    });
    log(`tool result sent name=${event.name} call_id=${event.call_id}`, result);
  }

  async continueAfterTools(state) {
    await Promise.allSettled(state.toolPromises);
    if (this.shuttingDown) return;
    this.sendModel({ type: 'response.create', response: { output_modalities: ['audio'] } });
    log(`response.create sent after ${state.toolPromises.length} tool call(s)`);
  }

  handleModelEvent(raw) {
    let event;
    try { event = JSON.parse(raw.toString()); } catch { return; }
    if (shouldLogEvent(event.type)) log(`OpenAI event ${event.type}`, safeEventForLog(event));

    switch (event.type) {
      case 'session.created':
        this.sessionCreated = true;
        this.maybeConfigureSession();
        break;
      case 'session.updated':
        if (this.args.speakFirst && !this.greeted) {
          this.greeted = true;
          this.sendModel({ type: 'response.create', response: { output_modalities: ['audio'] } });
          log('response.create sent for speakFirst greeting');
        }
        break;
      case 'response.created':
        this.responses.set(event.response.id, {
          toolPromises: [], hasToolCalls: false, transcript: '', hasAudio: false,
        });
        break;
      case 'response.output_audio.delta': {
        const bytes = Buffer.from(event.delta, 'base64');
        this.startAudioTurn(event.response_id);
        const state = this.responses.get(event.response_id);
        if (state) state.hasAudio = true;
        this.output.push(bytes);
        break;
      }
      case 'response.output_audio_transcript.delta': {
        const state = this.responses.get(event.response_id);
        if (state) state.transcript += event.delta || '';
        if (this.args.enableTranscription && event.delta) {
          emit({ type: 'transcript_delta', source: 'response', delta: event.delta });
        }
        break;
      }
      case 'response.output_audio_transcript.done': {
        const state = this.responses.get(event.response_id);
        if (state && event.transcript) state.transcript = event.transcript;
        break;
      }
      case 'conversation.item.input_audio_transcription.delta':
        if (this.args.enableTranscription && event.delta) {
          emit({ type: 'transcript_delta', source: 'input', delta: event.delta });
        }
        break;
      case 'conversation.item.input_audio_transcription.completed':
        if (this.args.enableTranscription) {
          emit({ type: 'transcript', source: 'input', text: event.transcript || '' });
        }
        break;
      case 'input_audio_buffer.speech_started':
        viz('user_started');
        break;
      case 'input_audio_buffer.speech_stopped':
        viz('user_stopped');
        break;
      case 'response.function_call_arguments.done': {
        const state = this.responses.get(event.response_id) || {
          toolPromises: [], hasToolCalls: false, transcript: '', hasAudio: false,
        };
        state.hasToolCalls = true;
        state.toolPromises.push(this.executeTool(event));
        this.responses.set(event.response_id, state);
        break;
      }
      case 'response.done': {
        const responseId = event.response?.id;
        const state = this.responses.get(responseId) || {
          toolPromises: [], hasToolCalls: false, transcript: '', hasAudio: false,
        };
        this.responses.delete(responseId);
        if (this.audioTurn?.responseIds.has(responseId)) {
          this.audioTurn.doneResponseIds.add(responseId);
          if (state.transcript) this.audioTurn.transcripts.push(state.transcript);
          this.finishAudioTurnIfReady();
        }
        if (state.hasToolCalls) this.continueAfterTools(state);
        break;
      }
      case 'error':
        log('OpenAI Realtime API error', event.error || event);
        break;
      default:
        break;
    }
  }

  handleStdin(message) {
    if (message.type === 'configure_tools') {
      this.tools = message.tools || [];
      this.toolServerUrl = message.serverUrl || null;
      this.toolsConfigured = true;
      log(`configure_tools received tools=${this.tools.length} serverUrl=${this.toolServerUrl}`);
      this.maybeConfigureSession();
      return;
    }
    if (message.type === 'inject') {
      this.sendModel({
        type: 'conversation.item.create',
        item: {
          type: 'message',
          role: 'user',
          content: [{ type: 'input_text', text: message.text || '' }],
        },
      });
      if (message.trigger_response) {
        const delay = Number(message.trigger_response_delay || 0) * 1000;
        setTimeout(() => this.sendModel({ type: 'response.create' }), delay);
      }
    }
  }

  startStdin() {
    const lines = readline.createInterface({ input: process.stdin, terminal: false });
    lines.on('line', (line) => {
      try { this.handleStdin(JSON.parse(line)); } catch (error) { log(`invalid stdin JSON: ${error.message}`); }
    });
  }

  async run() {
    this.startStdin();
    this.startPacer();
    await this.connectSignal();
    this.connectModel();
    log(`started role=B model=${this.args.model} voice=${this.args.voice} vad=${this.args.vadMode}`);
  }

  shutdown(reason) {
    if (this.shuttingDown) return;
    this.shuttingDown = true;
    log(`shutting down: ${reason}`);
    if (this.pacer) clearInterval(this.pacer);
    if (this.sink) {
      try { this.sink.stop(); } catch {}
    }
    if (this.signalWs) {
      try { this.signalWs.close(); } catch {}
    }
    if (this.modelWs) {
      try { this.modelWs.close(); } catch {}
    }
    if (this.pc) {
      try { this.pc.close(); } catch {}
    }
    setTimeout(() => process.exit(0), 25).unref();
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const adapter = new NativeOpenAIAdapter(args);
  process.on('SIGINT', () => adapter.shutdown('SIGINT'));
  process.on('SIGTERM', () => adapter.shutdown('SIGTERM'));
  await adapter.run();
}

if (require.main === module) {
  main().catch((error) => {
    log(`fatal: ${error.stack || error.message}`);
    process.exit(1);
  });
}

module.exports = {
  ByteQueue,
  NativeOpenAIAdapter,
  parseArgs,
  realtimeTools,
  resamplePcm16,
  sessionUpdate,
  turnDetection,
};
