#!/usr/bin/env node
/**
 * openai_token_server.js
 * Tiny Express server that mints short‑lived Realtime session tokens for WebRTC
 * using your real OPENAI_API_KEY. Mirrors the pattern in OpenAI's Realtime console app.
 *
 * Endpoints
 *   POST /session  -> { client_secret: { value, expires_at }, id, ... }
 *   GET  /healthz  -> 200 OK
 *
 * Env
 *   OPENAI_API_KEY=sk-...
 *   OPENAI_REALTIME_MODEL=gpt-4o-realtime-preview   (optional)
 *   TOKEN_SERVER_PORT=3002                          (optional)
 *
 * CLI
 *   node openai_token_server.js [--port PORT] [--model MODEL]
 */

require('dotenv').config();
const express = require('express');
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');

// Parse CLI arguments
const argv = yargs(hideBin(process.argv))
  .scriptName('openai_token_server')
  .usage('$0 [options]')
  .option('port', { alias: 'p', type: 'number', default: parseInt(process.env.TOKEN_SERVER_PORT, 10) || 3002, describe: 'Server port' })
  .option('model', { alias: 'm', type: 'string', default: process.env.OPENAI_REALTIME_MODEL || 'gpt-4o-realtime-preview', describe: 'Default OpenAI Realtime model' })
  .help().argv;

const PORT = argv.port;
const OPENAI_KEY = process.env.OPENAI_API_KEY;
const DEFAULT_MODEL = argv.model;

if (!OPENAI_KEY) {
  console.error('[token-server] Missing OPENAI_API_KEY in environment.');
  process.exit(1);
}

const app = express();
app.use(express.json());

// Very small allowlist for local access by default. Expand if you need remote adapters.
app.use((req, res, next) => {
  // If you want to restrict further: check req.ip === '::1' / '127.0.0.1' or shared secret header
  next();
});

app.get('/healthz', (_req, res) => res.status(200).send('ok'));

/**
 * Create a short‑lived Realtime session on OpenAI and return the token payload.
 * Body (optional): { model, voice, instructions, turn_detection, input_audio_transcription }
 */
// As of late 2025/early 2026 OpenAI retired the Realtime Beta API
// ("/v1/realtime/sessions"). All models now use the GA endpoint
// (/v1/realtime/client_secrets). The Beta path remains here as dead
// code in case of rollback.
function needsGAEndpoint(_model) {
  return true;
}

app.post('/session', async (req, res) => {
  try {
    const body = req.body || {};
    const model = body.model || DEFAULT_MODEL;
    const useGA = needsGAEndpoint(model);

    let endpoint, reqBody;
    if (useGA) {
      // GA: POST /v1/realtime/client_secrets  body { session: { type:'realtime', model, ... } }
      // Response: { value, expires_at, session }
      const sessionFields = { type: 'realtime', model };
      if (body.voice) sessionFields.audio = { output: { voice: body.voice } };
      if (body.instructions) sessionFields.instructions = body.instructions;
      if (body.turn_detection) sessionFields.turn_detection = body.turn_detection;
      endpoint = 'https://api.openai.com/v1/realtime/client_secrets';
      reqBody = { session: sessionFields };
    } else {
      const payload = { model };
      if (body.voice) payload.voice = body.voice;
      if (body.instructions) payload.instructions = body.instructions;
      if (body.turn_detection) payload.turn_detection = body.turn_detection;
      if (body.input_audio_transcription) payload.input_audio_transcription = body.input_audio_transcription;
      endpoint = 'https://api.openai.com/v1/realtime/sessions';
      reqBody = payload;
    }

    const r = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${OPENAI_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(reqBody)
    });

    if (!r.ok) {
      const errText = await r.text().catch(() => '');
      console.error(`[token-server] OpenAI ${endpoint} error:`, r.status, errText);
      return res.status(500).json({ error: 'failed_to_create_session', status: r.status, detail: errText, endpoint });
    }

    let json = await r.json();
    // Normalize GA response shape so adapters can keep reading
    // session.client_secret.value uniformly.
    if (useGA && json && json.value && !json.client_secret) {
      json = { ...json, client_secret: { value: json.value, expires_at: json.expires_at } };
    }
    res.json(json);
  } catch (e) {
    console.error('[token-server] /session exception:', e);
    res.status(500).json({ error: 'exception', detail: String(e) });
  }
});

app.listen(PORT, () => {
  console.log(`[token-server] Listening on http://localhost:${PORT}`);
});
