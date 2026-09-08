#!/usr/bin/env node
/*
 * Copyright 2026 DoorDash, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Minimal HTTP tool server for the Order Up clip-player benchmark.
 *
 * The Role B adapter POSTs each examinee tool call to {serverUrl}/api/tool
 * with body { tool: 'add_to_cart' | ..., args: {...} }. We maintain an
 * in-memory cart, validate item ids against the task's menu_sample, and
 * write final_cart.json + tool_log.json into the run's record dir on
 * submit_order or process exit.
 *
 * Usage:
 *   node order_up_tool_server.js \
 *     --task-config /path/to/<task_id>.task.json \
 *     --record-dir  /path/to/run/output/dir \
 *     --port 3101
 */

const fs = require('fs');
const path = require('path');
const http = require('http');
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');

const argv = yargs(hideBin(process.argv))
  .option('taskConfig', { alias: 'c', type: 'string', demandOption: true,
                          describe: 'Path to <task_id>.task.json' })
  .option('recordDir',  { alias: 'r', type: 'string', demandOption: true })
  .option('port',       { alias: 'p', type: 'number', default: 3101 })
  .help().argv;

const config = JSON.parse(fs.readFileSync(argv.taskConfig, 'utf8'));
const recordDir = path.resolve(argv.recordDir);
fs.mkdirSync(recordDir, { recursive: true });

// Build menu validators. Prefer the new per-category agent_menu shape;
// fall back to the legacy flat items_to_order for older task configs.
const menuItems = [];
for (const cat of (config.menu_sample?.agent_menu || [])) {
  for (const it of (cat.items || [])) {
    if (it && it.base_id) menuItems.push(it);
  }
}
if (menuItems.length === 0) {
  for (const it of (config.menu_sample?.items_to_order || [])) {
    if (it && it.base_id) menuItems.push(it);
  }
}
const validIds = new Set(menuItems.map(it => it.base_id));
const groupsByItem = {};   // base_id → list of OptionGroup objects
for (const it of menuItems) {
  if (Array.isArray(it.option_groups) && it.option_groups.length > 0) {
    groupsByItem[it.base_id] = it.option_groups;
  }
}

// Cart state: ordered list of { item_id, option, quantity }
const cart = [];
const toolLog = [];
let submitted = false;
let submittedAt = null;
let submitSource = null;  // 'agent' (normal submit_order) or 'customer_auto' (clip player missing_answer at end-of-order)
let customerAutoReasoning = null;  // reasoning string from the routing-model call when submitSource='customer_auto'

function logToolCall(tool, args, result) {
  // Deep-copy `result` so each log entry captures the cart STATE AT THE
  // TIME OF THIS CALL. Handlers return { ok, cart } where `cart` is the
  // shared mutable array — without this snapshot every entry's `.cart`
  // would resolve to the same (final) cart array at JSON-stringify time,
  // making it look like remove_from_cart never reduced the cart, that
  // adds didn't fire, etc. Audit-time investigations rely on this being
  // a faithful step-by-step record.
  let snapshotResult = result;
  try {
    snapshotResult = JSON.parse(JSON.stringify(result));
  } catch (_) {
    // Fall through to the live reference; better than dropping the call.
  }
  toolLog.push({
    t: Date.now(),
    tool,
    args,
    result: snapshotResult,
  });
}

function _optionsListsEqual(a, b) {
  // Compare two lists of choice names ignoring order. Both inputs are
  // arrays of strings (validated upstream).
  if (a.length !== b.length) return false;
  const sa = [...a].sort();
  const sb = [...b].sort();
  for (let i = 0; i < sa.length; i++) {
    if (sa[i] !== sb[i]) return false;
  }
  return true;
}

function _optionsDictSubsetMatch(cartOpts, filter) {
  // Returns true when every (group, list) pair in `filter` is present
  // in `cartOpts` with an order-insensitive list-equal value. Extra
  // keys in `cartOpts` are allowed — the server fills required-with-
  // default groups silently on add, so the agent's remove call only
  // needs to specify the distinguishing groups, not every default
  // value that landed in the stored row.
  for (const k of Object.keys(filter || {})) {
    if (!(k in (cartOpts || {}))) return false;
    if (!_optionsListsEqual(cartOpts[k] || [], filter[k] || [])) return false;
  }
  return true;
}

function findLastMatch(itemId, optionsDict) {
  // Find the most-recent cart row with matching item_id whose options
  // dict is a SUPERSET of the provided filter (group→list[choice],
  // order-insensitive per group). When optionsDict is null/undefined,
  // match the most-recent row with the given item_id regardless of
  // options.
  for (let i = cart.length - 1; i >= 0; i--) {
    if (cart[i].item_id !== itemId) continue;
    if (optionsDict
        && !_optionsDictSubsetMatch(cart[i].options || {}, optionsDict)) {
      continue;
    }
    return i;
  }
  return -1;
}

function _resolveOptions(itemId, providedOptions) {
  // Validate + default-fill the agent's provided option dict. Returns
  // { ok: true, options: <resolved dict of group→list[choice]> } or
  // { ok: false, error: ... }.
  //
  // Option values are ALWAYS lists, even single-select groups (length 1).
  // Multi-select groups (per OptionGroup.multi_select=true) can carry
  // more than one choice in the list; single-select groups are capped at
  // length 1 here.
  const groups = groupsByItem[itemId] || [];
  const provided = providedOptions && typeof providedOptions === 'object'
    ? providedOptions : {};
  const groupByName = {};
  for (const g of groups) groupByName[g.name] = g;

  // Reject keys naming groups that don't exist on this item.
  for (const key of Object.keys(provided)) {
    if (!groupByName[key]) {
      return {
        ok: false,
        error: `unknown option group "${key}" for ${itemId}. ` +
               `Valid groups: ${groups.map(g => g.name).join(' | ') || '(none)'}`,
      };
    }
  }
  // Validate shape (must be array of strings), choice membership, and
  // multi_select cardinality cap.
  for (const [groupName, value] of Object.entries(provided)) {
    const g = groupByName[groupName];
    if (!Array.isArray(value)) {
      return {
        ok: false,
        error: `options["${groupName}"] must be a list of choice names ` +
               `(got ${typeof value}). Use ["${value}"] for single-` +
               `select groups; multi-select groups accept ["a","b",...].`,
      };
    }
    if (value.some(v => typeof v !== 'string')) {
      return {
        ok: false,
        error: `options["${groupName}"] must contain only string choice ` +
               `names`,
      };
    }
    if (value.length === 0) {
      return {
        ok: false,
        error: `options["${groupName}"] cannot be empty — omit the key ` +
               `entirely to fall back to default or skip an optional group`,
      };
    }
    if (!g.multi_select && value.length > 1) {
      return {
        ok: false,
        error: `group "${groupName}" on ${itemId} is single-select; ` +
               `cannot pick more than one choice (got ${value.length})`,
      };
    }
    const validChoices = (g.choices || []).map(c => c.name);
    for (const choiceName of value) {
      if (!validChoices.includes(choiceName)) {
        return {
          ok: false,
          error: `choice "${choiceName}" not valid for group ` +
                 `"${groupName}" on ${itemId} (must be one of: ` +
                 `${validChoices.join(' | ')})`,
        };
      }
    }
  }
  // For every required group: if absent, fill with default (wrapped in
  // a single-element list); if no default, error so the agent retries.
  const resolved = { ...provided };
  for (const g of groups) {
    if (!g.required) continue;
    if (resolved[g.name] !== undefined) continue;
    if (g.default) {
      resolved[g.name] = [g.default];
    } else {
      const choiceList = (g.choices || []).map(c => c.name).join(' | ');
      return {
        ok: false,
        error: `${itemId} requires a choice for group "${g.name}" ` +
               `(one of: ${choiceList})`,
      };
    }
  }
  return { ok: true, options: resolved };
}

function handleAddToCart(args) {
  const itemId = args?.item_id;
  const qty    = args?.quantity || 1;
  if (!itemId) {
    return { ok: false, error: 'item_id required' };
  }
  if (!validIds.has(itemId)) {
    return { ok: false, error: `unknown item_id: ${itemId}`, valid_item_ids: Array.from(validIds) };
  }
  const resolution = _resolveOptions(itemId, args?.options);
  if (!resolution.ok) return resolution;
  cart.push({ item_id: itemId, options: resolution.options, quantity: qty });
  return { ok: true, cart };
}

function handleRemoveFromCart(args) {
  const itemId = args?.item_id;
  if (!itemId) return { ok: false, error: 'item_id required' };
  // Match by item_id + (optionally) the full options dict
  // (order-insensitive per-group list). If options absent, match the
  // most-recent row with this item_id regardless of options.
  const opts = (args && args.options && typeof args.options === 'object')
    ? args.options : null;
  if (opts) {
    // Validate the provided options dict has list-valued entries —
    // reject scalar values up front rather than silently mismatching.
    for (const [g, v] of Object.entries(opts)) {
      if (!Array.isArray(v)) {
        return {
          ok: false,
          error: `remove_from_cart options["${g}"] must be a list ` +
                 `of choice names (got ${typeof v})`,
        };
      }
    }
  }
  const idx = findLastMatch(itemId, opts);
  if (idx < 0) {
    return {
      ok: false,
      error: `no matching item in cart for ${itemId}` +
             (opts ? ` with options ${JSON.stringify(opts)}` : ''),
    };
  }
  cart.splice(idx, 1);
  return { ok: true, cart };
}

function handleGetCart() {
  return { ok: true, cart };
}

function handleSubmitOrder(args) {
  submitted = true;
  submittedAt = Date.now();
  // The customer adapter can submit on its own behalf when the game has
  // no "that's all" clip and the order has run out of customer lines —
  // captured via args.source === "customer_auto". We tag the finalize
  // reason so downstream metrics can distinguish customer_auto_submit
  // (off-policy turn skipped) from agent-driven submit_order.
  if (args && args.source === 'customer_auto') {
    submitSource = 'customer_auto';
    customerAutoReasoning = (args.reasoning || '').toString() || null;
  } else {
    submitSource = 'agent';
  }
  return { ok: true, submitted: true, cart, source: submitSource };
}

const TOOL_HANDLERS = {
  add_to_cart: handleAddToCart,
  remove_from_cart: handleRemoveFromCart,
  get_cart: handleGetCart,
  submit_order: handleSubmitOrder,
};

function writeFinal(reason) {
  const finalPath = path.join(recordDir, 'final_cart.json');
  const logPath = path.join(recordDir, 'tool_log.json');
  fs.writeFileSync(finalPath, JSON.stringify({
    task_id: config.task_id,
    submitted,
    submitted_at: submittedAt,
    finalize_reason: reason,
    customer_auto_reasoning: customerAutoReasoning,
    cart,
  }, null, 2));
  fs.writeFileSync(logPath, JSON.stringify(toolLog, null, 2));
  console.log(`[tool-server] wrote ${finalPath} (${cart.length} items, submitted=${submitted}, reason=${reason})`);
}

const server = http.createServer((req, res) => {
  // Read-only status endpoint — the clipPlayer (Role A) polls this before
  // each router decision so it can exit gracefully once the agent has
  // submitted, instead of looping on the thats_all clip until duration
  // timeout. Tool calls themselves go through POST /api/tool (below).
  if (req.method === 'GET' && req.url === '/api/status') {
    res.setHeader('content-type', 'application/json');
    res.end(JSON.stringify({
      submitted,
      submit_source: submitSource,
      submitted_at: submittedAt,
      cart_size: cart.length,
    }));
    return;
  }
  if (req.method !== 'POST' || req.url !== '/api/tool') {
    res.statusCode = 404;
    res.end('not found');
    return;
  }
  let body = '';
  req.on('data', chunk => { body += chunk; });
  req.on('end', () => {
    let payload;
    try {
      payload = JSON.parse(body || '{}');
    } catch (e) {
      res.statusCode = 400;
      res.end(JSON.stringify({ result: { ok: false, error: 'invalid json' } }));
      return;
    }
    // Accept both supported adapter payload shapes.
    const tool = payload.tool || payload.name;
    const args = payload.args || payload.arguments || {};
    const handler = TOOL_HANDLERS[tool];
    if (!handler) {
      const result = { ok: false, error: `unknown tool: ${tool}` };
      logToolCall(tool, args, result);
      res.statusCode = 400;
      res.end(JSON.stringify({ result }));
      return;
    }
    let result;
    try {
      result = handler(args);
    } catch (e) {
      result = { ok: false, error: String(e) };
    }
    logToolCall(tool, args, result);
    if (tool === 'submit_order' && result.ok) {
      writeFinal(submitSource === 'customer_auto' ? 'customer_auto_submit' : 'submit_order');
    }
    res.setHeader('content-type', 'application/json');
    // Role B adapters consume the tool result from this envelope.
    res.end(JSON.stringify({ result }));
  });
});

server.listen(argv.port, () => {
  console.log(`[tool-server] task=${config.task_id} listening on http://localhost:${argv.port}`);
  console.log(`[tool-server] valid item ids: ${Array.from(validIds).join(', ')}`);
  // Print a single line that the wrapper script can grep for to confirm ready
  console.log(`READY port=${argv.port}`);
});

function shutdown(reason) {
  if (!submitted) writeFinal(reason);
  server.close(() => process.exit(0));
}

process.on('SIGINT',  () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));
