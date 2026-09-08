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

'use strict';

const assert = require('assert');
const {
  ByteQueue,
  NativeOpenAIAdapter,
  realtimeTools,
  resamplePcm16,
  sessionUpdate,
  turnDetection,
} = require('./openai_realtime_adapter');

function baseArgs() {
  return {
    role: 'B',
    signalUrl: 'ws://localhost:1/signal',
    model: 'gpt-realtime-2',
    voice: 'shimmer',
    systemPrompt: 'Take an order.',
    turnMode: 'server_vad',
    vadMode: 'fast',
    speakFirst: true,
    enableTranscription: true,
  };
}

async function main() {
  const ramp48 = Int16Array.from({ length: 480 }, (_, index) => index - 240);
  const down = resamplePcm16(ramp48, 48000, 24000);
  const up = resamplePcm16(down, 24000, 48000);
  assert.equal(down.length, 240);
  assert.equal(up.length, 480);
  assert.equal(down[0], ramp48[0]);

  const queue = new ByteQueue();
  queue.push(Buffer.from([1, 2]));
  queue.push(Buffer.from([3, 4, 5]));
  assert.deepEqual([...queue.take(4).bytes], [1, 2, 3, 4]);
  assert.equal(queue.length, 1);
  assert.deepEqual([...queue.take(3).bytes], [5, 0, 0]);
  assert.equal(queue.length, 0);

  assert.equal(turnDetection('fast', 'server_vad').silence_duration_ms, 450);
  assert.equal(turnDetection('fast', 'server_vad').interrupt_response, false);
  assert.equal(turnDetection('slow', 'none'), null);

  const sourceTools = [{
    name: 'add_to_cart',
    description: 'Add an item.',
    parameters: { type: 'object', properties: { item_id: { type: 'string' } }, required: ['item_id'] },
  }];
  assert.deepEqual(realtimeTools(sourceTools)[0], { type: 'function', ...sourceTools[0] });
  const update = sessionUpdate(baseArgs(), sourceTools);
  assert.equal(update.session.audio.input.format.rate, 24000);
  assert.equal(update.session.audio.output.format.rate, 24000);
  assert.equal(update.session.audio.input.transcription.model, 'gpt-4o-transcribe');
  assert.equal(update.session.tools[0].type, 'function');

  const sent = [];
  class FakeWebSocket {}
  FakeWebSocket.OPEN = 1;
  const adapter = new NativeOpenAIAdapter(baseArgs(), {
    WebSocket: FakeWebSocket,
    fetch: async (url, options) => {
      assert.equal(url, 'http://tool-server/api/tool');
      assert.deepEqual(JSON.parse(options.body), {
        name: 'add_to_cart', arguments: { item_id: 'burger' },
      });
      return { json: async () => ({ result: { ok: true } }) };
    },
  });
  adapter.modelWs = { readyState: 1, send: (payload) => sent.push(JSON.parse(payload)) };
  adapter.toolServerUrl = 'http://tool-server';
  adapter.handleModelEvent(JSON.stringify({ type: 'response.created', response: { id: 'resp_1' } }));
  adapter.handleModelEvent(JSON.stringify({
    type: 'response.function_call_arguments.done',
    response_id: 'resp_1',
    call_id: 'call_1',
    name: 'add_to_cart',
    arguments: '{"item_id":"burger"}',
  }));
  adapter.handleModelEvent(JSON.stringify({
    type: 'response.done', response: { id: 'resp_1', status: 'completed' },
  }));
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(sent[0].type, 'conversation.item.create');
  assert.equal(sent[0].item.type, 'function_call_output');
  assert.equal(sent[1].type, 'response.create');

  process.stdout.write('openai-native selftest: PASS\n');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
