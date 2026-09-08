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

/** Mint short-lived OpenAI Realtime credentials for the local Role A client. */

'use strict';

require('dotenv').config();

const http = require('http');

const PORT = Number.parseInt(process.env.TOKEN_SERVER_PORT || '3002', 10);
const DEFAULT_MODEL = process.env.OPENAI_REALTIME_MODEL || 'gpt-realtime-2';
const API_KEY = process.env.OPENAI_API_KEY;

function respond(response, status, body) {
  response.writeHead(status, { 'content-type': 'application/json' });
  response.end(JSON.stringify(body));
}

function readJson(request) {
  return new Promise((resolve, reject) => {
    let raw = '';
    request.setEncoding('utf8');
    request.on('data', (chunk) => {
      raw += chunk;
      if (raw.length > 1024 * 1024) reject(new Error('request body too large'));
    });
    request.on('end', () => {
      if (!raw) return resolve({});
      try { resolve(JSON.parse(raw)); } catch { reject(new Error('invalid JSON')); }
    });
    request.on('error', reject);
  });
}

function buildSession(input) {
  const session = {
    type: 'realtime',
    model: input.model || DEFAULT_MODEL,
  };
  if (input.instructions) session.instructions = input.instructions;
  const audio = {};
  if (input.voice) audio.output = { voice: input.voice };
  if (input.turn_detection || input.input_audio_transcription) {
    audio.input = {};
    if (input.turn_detection) audio.input.turn_detection = input.turn_detection;
    if (input.input_audio_transcription) {
      audio.input.transcription = input.input_audio_transcription;
    }
  }
  if (Object.keys(audio).length) session.audio = audio;
  return session;
}

const server = http.createServer(async (request, response) => {
  if (request.method === 'GET' && request.url === '/healthz') {
    response.writeHead(200, { 'content-type': 'text/plain' });
    response.end('ok');
    return;
  }
  if (request.method !== 'POST' || request.url !== '/session') {
    respond(response, 404, { error: 'not_found' });
    return;
  }
  if (!API_KEY) {
    respond(response, 503, { error: 'OPENAI_API_KEY_not_set' });
    return;
  }

  try {
    const input = await readJson(request);
    const upstream = await fetch('https://api.openai.com/v1/realtime/client_secrets', {
      method: 'POST',
      headers: {
        authorization: `Bearer ${API_KEY}`,
        'content-type': 'application/json',
      },
      body: JSON.stringify({ session: buildSession(input) }),
    });
    const raw = await upstream.text();
    let body;
    try { body = JSON.parse(raw); } catch { body = { detail: raw }; }
    if (!upstream.ok) {
      console.error(`[token-server] OpenAI returned ${upstream.status}`);
      respond(response, 502, { error: 'credential_request_failed', status: upstream.status, detail: body });
      return;
    }
    if (body.value && !body.client_secret) {
      body.client_secret = { value: body.value, expires_at: body.expires_at };
    }
    respond(response, 200, body);
  } catch (error) {
    respond(response, 400, { error: error.message });
  }
});

server.listen(PORT, () => {
  console.log(`[token-server] listening on port ${PORT}`);
});

function shutdown() {
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 1000).unref();
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
