#!/usr/bin/env node
/**
 * test_only_one.js — the `only_one` quantity-additive reply.
 *
 * Why it exists: a bare "no" cannot express "I only want one". It reads as
 * declining the ITEM, so an agent that has already added a second unit
 * ("Added another fresh orange juice. Anything else?") can satisfy a "no"
 * without removing it. This asserts the clip is exposed, resolvable, and
 * distinct from yes/no.
 *
 * Run with ARTIFACTS_PATH pointing at an artifact tree whose shared clips
 * include only_one.wav.
 */
const path = require('path');
const fs = require('fs');
const { createDecisionTree } = require('../decision_tree');

const A = process.env.ARTIFACTS_PATH
  || '/Users/daniel.fischetti/Projects/Full-Duplex-Bench/v2/artifacts.slicing-20260901';
const TID = process.env.TID || 'order_up.Bella_Roma_Trattoria.24034958.g1.s1';
let pass = 0, fail = 0;
function ck(n, got, want) {
  const g = JSON.stringify(got), w = JSON.stringify(want);
  if (g === w) { pass++; console.log('  PASS ' + n); }
  else { fail++; console.log(`  FAIL ${n}\n        got ${g}\n       want ${w}`); }
}

const dt = createDecisionTree({
  treePath: path.join(A, 'decision_trees', `${TID}.json`),
  taskConfigPath: path.join(A, 'order_up_tasks', `${TID}.complex.task.json`),
});
dt.advanceTo(1, 0);
dt.markPlayed('g1_s1_i0_main');
const vc = dt.computeValidChoices();
const names = vc.map((v) => v.choice);

ck('only_one is exposed', names.includes('only_one'), true);
ck('basename resolves to the shared clip', dt.onlyOneClipBasename, 'only_one.wav');

const p = dt.resolveClipPath({ clip: dt.onlyOneClipBasename });
ck('audio file exists on disk', fs.existsSync(p), true);
ck('lives in the shared dir, not per-task', p.includes('order_up_shared'), true);

// Format must match the other shared clips or the customer's voice changes.
const buf = fs.readFileSync(p);
ck('is a RIFF/WAVE file', buf.slice(0, 4).toString() + buf.slice(8, 12).toString(), 'RIFFWAVE');
const channels = buf.readUInt16LE(22), rate = buf.readUInt32LE(24), bits = buf.readUInt16LE(34);
ck('mono', channels, 1);
ck('48 kHz', rate, 48000);
ck('16-bit', bits, 16);

const entry = vc.find((v) => v.choice === 'only_one');
ck('exposed as an action, not a clip', entry.kind, 'action');
ck('description states the quantity intent',
  /only want one/i.test(entry.description) && /prefer it over "no"/i.test(entry.description), true);
ck('description warns against close-out use', /close-out/i.test(entry.description), true);

// Must not displace yes/no — they answer different questions.
ck('yes and no are still exposed',
  ['yes', 'no'].every((c) => names.includes(c)), true);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
