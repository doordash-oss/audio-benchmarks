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
 * test_subclip_fallback.js — when an item is left unacknowledged, which
 * clip does the runtime replay?
 *
 * Why this exists: a live eval can't test it. On a clean run the agent
 * acknowledges every item, nothing is stranded, and the replay path is
 * never entered. The behaviour has to be driven directly.
 *
 * What it asserts — the ONE ordering, for every trigger:
 *    1. a `restatementN` for the unacked item, when one exists
 *    2. else the per-item sub-clip of the compound carrying its audio
 *    3. else the whole source clip
 *
 * `decision_tree.js` is the real module and `resurfaceUnackedItem()` is
 * called directly, so this exercises shipped code rather than a copy.
 * A structural check also fails if `clipPlayer_adapter.js` grows its own
 * clip-selection logic again — the three divergent copies that used to
 * live there are the bug this design removes.
 *
 * Usage:
 *   node tests/test_subclip_fallback.js [--artifacts <dir>] [--outdir <report_dir>]
 */

const fs = require('fs');
const path = require('path');
const { createDecisionTree } = require('../decision_tree');

function arg(name, dflt) {
  const i = process.argv.indexOf(`--${name}`);
  return i > -1 ? process.argv[i + 1] : dflt;
}

const ARTIFACTS = arg(
  'artifacts',
  process.env.ARTIFACTS_PATH || path.join(__dirname, '..', '..', '..', 'artifacts'),
);

// Two tids chosen so both branches of the ordering are covered by real
// data: one compound has a restatement for exactly one of its two items
// (proving restatement wins AND sub-clip is used where none exists), the
// other has no restatements at all (pure sub-clip path).
const TIDS = [
  'order_up.Bella_Roma_Trattoria.24034958.g1.s1',
  'order_up.Ember_Oak.8D015AC7.g4.s0',
];

const results = [];
const detail = [];
function check(name, actual, expected, extra) {
  const pass = JSON.stringify(actual) === JSON.stringify(expected);
  results.push({ name, pass, actual, expected, ...(extra || {}) });
  return pass;
}

function freshTree(tid) {
  return createDecisionTree({
    treePath: path.join(ARTIFACTS, 'decision_trees', `${tid}.json`),
    taskConfigPath: path.join(ARTIFACTS, 'order_up_tasks', `${tid}.complex.task.json`),
    clipsBaseDirOverride: path.join(ARTIFACTS, 'clips', tid),
  });
}

// ── Structural guard: the adapter must not pick clips itself ────────────
{
  const src = fs.readFileSync(path.join(__dirname, '..', 'clipPlayer_adapter.js'), 'utf8');
  // The adapter may still TEST for fallback_clips (to know a compound has
  // per-item slices) but must never iterate them to choose which item —
  // that choice belongs to decision_tree.resurfaceUnackedItem(). Three
  // separate copies of that loop is the bug this design removes, so
  // assert the loop shape is gone rather than counting references.
  const picksItself = /for \(const \w+ of (node|lastNode|fbClips)[.\w]*fallback_clips\)/.test(src)
    || /fallback_clips\.find\(/.test(src);
  check('adapter never iterates fallback_clips to choose an item', picksItself, false);
  check('adapter no longer references substituteForPrematureClose',
    src.includes('substituteForPrematureClose'), false);
  check('adapter calls the unified selector',
    src.includes('resurfaceUnackedItem'), true);
}

for (const tid of TIDS) {
  const probe = freshTree(tid);
  const compoundId = Object.keys(probe.nodes).find((nid) => {
    const n = probe.nodes[nid];
    const cii = n.covered_item_indices;
    return Array.isArray(cii) && new Set(cii).size >= 2
      && Array.isArray(n.fallback_clips) && n.fallback_clips.length >= 2;
  });
  if (!compoundId) {
    check(`${tid}: has a compound with fallback_clips`, false, true);
    continue;
  }
  const compound = probe.nodes[compoundId];
  const covered = [...new Set(compound.covered_item_indices)];
  const fbc = compound.fallback_clips;

  // Which items have a restatement in this tree?
  const restByItem = new Map();
  for (const nid of Object.keys(probe.nodes)) {
    const m = nid.match(/^g\d+_s\d+_i(\d+)_(restatement\d*)$/);
    if (m) {
      const i = Number(m[1]);
      if (!restByItem.has(i)) restByItem.set(i, []);
      restByItem.get(i).push(nid);
    }
  }

  // For each covered item, strand exactly that one and see what plays.
  for (const stranded of covered) {
    const ackThese = covered.filter((i) => i !== stranded).map((i) => `i${i}_main`);
    const dt = freshTree(tid);
    // Mirror runtime: the bootstrap advances the cursor before the first
    // clip plays. Without this currentStage stays -1.
    const cp = dt.parseClipId(compoundId);
    dt.advanceTo(cp.stage, cp.item);
    dt.markPlayed(compoundId);
    dt.recordAcks(ackThese);

    const unacked = dt.getUnackedIds();
    const got = dt.resurfaceUnackedItem();

    // Expectation derived from the data, not hardcoded.
    const rests = restByItem.get(stranded) || [];
    const sub = fbc.find((f) => f.item_index === stranded);
    let wantSource;
    let wantClip;
    if (rests.length) {
      wantSource = 'restatement';
      // decision_tree tracks the LAST restatement seen for the item.
      wantClip = null; // node id checked instead
    } else if (sub) {
      wantSource = `sub_i${stranded}`;
      wantClip = sub.clip;
    } else {
      wantSource = 'source_clip';
      wantClip = compound.clip;
    }

    const label = `${tid.split('.').slice(1, 3).join('.')} i${stranded} stranded`;
    detail.push({
      tid,
      compound_node: compoundId,
      compound_clip: compound.clip,
      label,
      acked: ackThese,
      unacked,
      stranded_item: stranded,
      restatements_available: rests,
      sub_clip_available: sub ? sub.clip : null,
      expected_source: wantSource,
      got_source: got && got.source,
      got_clip: got && got.node && got.node.clip,
      got_node_id: got && got.node && got.node.id,
      has_realNodeId: !!(got && got.node && got.node._realNodeId),
      sub_clip_text: sub ? sub.text : null,
    });

    check(`${label}: unacked is exactly [i${stranded}_main]`,
      unacked, [`i${stranded}_main`]);
    check(`${label}: picks ${wantSource}`, got && got.source, wantSource);
    if (wantClip) {
      check(`${label}: plays ${wantClip} (not the whole compound)`,
        got && got.node && got.node.clip, wantClip,
        { compound: compound.clip });
    }
    if (rests.length) {
      check(`${label}: restatement wins over the available sub-clip`,
        rests.includes(got && got.node && got.node.id), true,
        { sub_clip_that_lost: sub ? sub.clip : null });
    }
    if (String(wantSource).startsWith('sub_')) {
      check(`${label}: derived node carries _realNodeId (next turn resolves)`,
        !!(got && got.node && got.node._realNodeId), true);
    }
  }

  // A bare ["generic"] must ack every covered item, so it can never
  // strand one — partial acks only come from naming an item id.
  const dtg = freshTree(tid);
  const cpg = dtg.parseClipId(compoundId);
  dtg.advanceTo(cpg.stage, cpg.item);
  dtg.markPlayed(compoundId);
  dtg.recordAcks(['generic']);
  check(`${tid.split('.').slice(1, 3).join('.')}: bare ["generic"] acks all ${covered.length} covered items`,
    dtg.getUnackedIds(), []);
}

// ── Report ─────────────────────────────────────────────────────────────
const nPass = results.filter((r) => r.pass).length;
const nFail = results.length - nPass;
const summary = {
  generated: new Date().toISOString(),
  artifacts: ARTIFACTS,
  tids: TIDS,
  ordering: ['restatement', 'per-item sub-clip', 'whole source clip'],
  passed: nPass,
  failed: nFail,
  results,
  detail,
};

for (const r of results) {
  console.log(`  ${r.pass ? 'PASS' : 'FAIL'}  ${r.name}`);
  if (!r.pass) console.log(`         got ${JSON.stringify(r.actual)}  want ${JSON.stringify(r.expected)}`);
}
console.log(`\n${nPass} passed, ${nFail} failed\n`);

const outdir = arg('outdir', null);
if (outdir) {
  fs.mkdirSync(path.join(outdir, 'audio'), { recursive: true });
  const copied = [];
  for (const d of detail) {
    const wants = [d.compound_clip, d.sub_clip_available, d.got_clip].filter(Boolean);
    for (const w of wants) {
      const src = path.join(ARTIFACTS, 'clips', d.tid, w);
      const dst = path.join(outdir, 'audio', `${d.tid}__${w}`);
      if (fs.existsSync(src) && !fs.existsSync(dst)) {
        fs.copyFileSync(src, dst);
        copied.push(`${d.tid}__${w}`);
      }
    }
  }
  summary.audio_copied = copied;
  fs.writeFileSync(path.join(outdir, 'results.json'), JSON.stringify(summary, null, 1));
  fs.writeFileSync(path.join(outdir, 'index.html'), renderHtml(summary));
  console.log(`report → ${path.join(outdir, 'index.html')}  (${copied.length} wavs)`);
}

function esc(s) {
  return String(s === undefined || s === null ? '—' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function renderHtml(s) {
  const au = (tid, clip) => clip
    ? `<audio controls preload=none src="audio/${encodeURIComponent(`${tid}__${clip}`)}"></audio>` : '';
  const rows = s.detail.map((d) => {
    const ok = d.got_source === d.expected_source;
    return `<tr>
      <td><b>${esc(d.label)}</b><div class=sub>${esc(d.tid)}<br>compound <code>${esc(d.compound_node)}</code></div></td>
      <td>acked ${esc(JSON.stringify(d.acked))}<div class=sub>unacked ${esc(JSON.stringify(d.unacked))}</div></td>
      <td>${d.restatements_available.length ? `<b>${esc(d.restatements_available.join(', '))}</b>` : '<i>none</i>'}</td>
      <td>${d.sub_clip_available ? `<code>${esc(d.sub_clip_available)}</code><div class=sub>${esc(d.sub_clip_text)}</div>${au(d.tid, d.sub_clip_available)}` : '<i>none</i>'}</td>
      <td class="${ok ? 'ok' : 'bad'}"><b>${esc(d.got_source)}</b><div class=sub><code>${esc(d.got_node_id)}</code><br>clip <code>${esc(d.got_clip)}</code><br>_realNodeId ${d.has_realNodeId ? 'yes' : 'n/a'}</div>${au(d.tid, d.got_clip)}</td>
      <td class=cmp><code>${esc(d.compound_clip)}</code><div class=sub>what used to replay</div>${au(d.tid, d.compound_clip)}</td>
    </tr>`;
  }).join('');
  const asserts = s.results.map((r) => `<tr class="${r.pass ? 'p' : 'f'}">
      <td>${r.pass ? '✓' : '✗'}</td><td>${esc(r.name)}</td>
      <td>${esc(JSON.stringify(r.actual))}</td><td>${esc(JSON.stringify(r.expected))}</td>
      <td>${esc(r.note || r.sub_clip_that_lost || '')}</td></tr>`).join('');
  return `<!doctype html><meta charset=utf-8><title>Unacked-item replay ordering</title>
<style>
 body{font:13px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;color:#1a1a1a;background:#fafafa}
 h1{font-size:19px;margin:0 0 4px}h2{font-size:15px;margin:26px 0 8px}
 .meta{color:#666;font-size:12px}
 table{border-collapse:collapse;width:100%;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.08);margin-bottom:8px}
 th,td{border:1px solid #e3e3e3;padding:8px 10px;text-align:left;vertical-align:top;font-size:12px}
 th{background:#f3f4f6;font-weight:600}
 .sub{color:#777;font-size:11px;margin-top:3px}
 .ok{background:#e8f6ec}.bad{background:#fdeaea}.cmp{background:#f7f7f7}
 tr.p td:first-child{color:#137333;font-weight:700}tr.f td:first-child{color:#c5221f;font-weight:700}
 .banner{padding:10px 12px;border-radius:6px;margin:12px 0;font-weight:600}
 .green{background:#e8f6ec;color:#137333}.red{background:#fdeaea;color:#c5221f}
 audio{width:185px;height:28px;margin-top:5px;display:block}
 code{background:#f1f1f1;padding:1px 4px;border-radius:3px}
 ol{margin:4px 0 0 18px;padding:0}
</style>
<h1>What replays when an item is left unacknowledged</h1>
<div class=meta>One ordering for every trigger — premature close, <code>next_item</code> with items
 outstanding, and explicit <code>fallback</code> — resolved by
 <code>decision_tree.resurfaceUnackedItem()</code>:
 <ol><li>a <code>restatementN</code> for the unacked item, if one exists</li>
 <li>else the per-item sub-clip of the compound carrying its audio</li>
 <li>else the whole source clip</li></ol></div>
<div class=meta style="margin-top:6px">artifacts <code>${esc(s.artifacts)}</code> · generated ${esc(s.generated)}</div>
<div class="banner ${s.failed ? 'red' : 'green'}">${s.passed} passed, ${s.failed} failed</div>
<h2>Per stranded item</h2>
<table><tr><th>scenario</th><th>ack state</th><th>restatement?</th><th>sub-clip?</th>
 <th>what plays</th><th>whole compound</th></tr>${rows}</table>
<div class=meta>Compare the last two columns: the right-hand clip is what replayed before
 per-item fallbacks existed.</div>
<h2>Assertions</h2>
<table><tr><th></th><th>assertion</th><th>actual</th><th>expected</th><th>note</th></tr>${asserts}</table>
<p class=meta><code>resurfaceUnackedItem()</code> is called directly on the real
 <code>decision_tree.js</code>. A structural assertion fails if
 <code>clipPlayer_adapter.js</code> regrows its own selection logic.</p>`;
}

process.exit(nFail ? 1 : 0);
