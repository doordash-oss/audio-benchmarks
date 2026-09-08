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

// Decision-tree library for the clipPlayer adapter.
//
// Owns the static metadata of a per-stage Order-Up decision tree
// (clip index, yes/no/thats_all/just_the_defaults detection, router
// hints, default-eligible options) AND the live progression state
// (currentStage, currentItem, played clips, last clip injected,
// conversation-ended flag).
//
// The adapter handles WebRTC, audio playback, VAD, tool-call I/O.
// This library handles "which clips are reachable right now, what
// should the LLM see as valid_choices, and what node does each
// router pick resolve to."
//
// Single entry point:
//
//   const dt = createDecisionTree({
//     treePath:               '<path>.json',
//     taskConfigPath:         '<path>.task.json',  // optional
//     overridesPath:          '<path>.override.json',  // optional
//     clipsBaseDirOverride:   '<dir>',                  // optional
//   });
//
// The returned object exposes:
//
//   Static metadata (read-only):
//     dt.taskId, dt.gameIndex, dt.initialNodeId, dt.clipsBaseDir,
//     dt.sharedClipsDir, dt.initialNoiseRotation,
//     dt.yesClipBasename, dt.noClipBasename,
//     dt.justTheDefaultsClipBasename, dt.thatsAllClip,
//     dt.defaultEligibleOptionsText,
//     dt.nodes              // raw tree.nodes (mutable, post-overrides)
//     dt.clipsByItem        // Map<"s.i", clipNode[]>
//     dt.allItems           // [[stage, item], ...]
//     dt.routerHintsByClipId // Map<clip_id, hint>
//
//   Static helpers:
//     dt.parseClipId(id), dt.isCorrKind(k),
//     dt.isAutoCorrKind(k), dt.stripCorrSuffix(k),
//     dt.resolveClipPath(node)
//
//   State accessors:
//     dt.getCurrentStage(), dt.getCurrentItem(),
//     dt.getLastPlayedClipId(), dt.hasPlayed(id),
//     dt.isConversationEnded(),
//     dt.getLastValidChoices()
//
//   State mutators:
//     dt.advanceTo(stage, item)   // clears playedClips on stage change
//     dt.markPlayed(clipId)        // updates playedClips + lastPlayedClipId
//     dt.endConversation()
//
//   Queries:
//     dt.nextItemAfter(s, i)       // → [stage, item] or null/undefined
//     dt.pendingCorrection()       // → corr node or null
//     dt.resolveItemClip(c, items, opts)
//     dt.clipDescription(node, parsed, info)
//     dt.mainClipItemExtras(node)
//     dt.computeValidChoices()     // updates lastValidChoices internally
//
// The adapter consumes this object; it does not import any of the
// internal helpers directly.

const fs = require('fs');
const path = require('path');

const CLIP_ID_RE = /^g(\d+)_s(\d+)_i(\d+)_(.+)$/;
// Match both bare `*_corr` and chained `*_corr_2`, `*_corr_3` kinds emitted
// by gap-aware corr-dedup. A late-arriving second correction targeting the
// same parent (e.g. customer says "could you change to mild" THEN minutes
// later "did you make sure it's mild?") gets a numeric suffix so its
// clip_id is unique and it can chain via parent_clip_kind.
const CORR_KIND_RE = /_corr(?:_\d+)?$/;
// The BARE `_corr` (no numeric suffix) is the auto-played correction — it
// supersedes its parent (e.g. opt1) the moment the parent plays. The
// numbered chained kinds (`_corr_2`, `_corr_3` …) are RESTATEMENTS of the
// correction, emitted by gap-aware corr-dedup when the customer revisits
// the same correction in a separate turn.
const AUTO_CORR_KIND_RE = /_corr$/;

function isCorrKind(k) { return CORR_KIND_RE.test(k || ''); }
function isAutoCorrKind(k) { return AUTO_CORR_KIND_RE.test(k || ''); }
function stripCorrSuffix(k) { return (k || '').replace(CORR_KIND_RE, ''); }
function parseClipId(id) {
  const m = CLIP_ID_RE.exec(id || '');
  if (!m) return null;
  return {
    game: parseInt(m[1], 10),
    stage: parseInt(m[2], 10),
    item: parseInt(m[3], 10),
    kind: m[4],
  };
}
// Short IDs shown to the model are the game/stage-stripped tail — e.g.
// `g8_s2_i1_main` → `i1_main`. Everything the router picks from
// valid_choices is short. `_handleSpecificClip` re-attaches the
// `g{gameIndex}_s{currentStage}_` prefix before looking the node up.
function shortenClipId(id) {
  if (!id) return id;
  const m = /^g\d+_s\d+_(.+)$/.exec(id);
  return m ? m[1] : id;
}
function expandClipId(shortId, gameIndex, stageIndex) {
  if (!shortId) return shortId;
  // Already long-form? Pass through untouched.
  if (/^g\d+_s\d+_/.test(shortId)) return shortId;
  return `g${gameIndex}_s${stageIndex}_${shortId}`;
}
function itemKey(s, i) { return `${s}.${i}`; }

// Per-item ordering: main first, then optN by numeric suffix, then
// _corr/_post_revision after their parents.
function _sortKey(kind) {
  if (kind === 'main') return 0;
  if (isCorrKind(kind) && stripCorrSuffix(kind) === 'main') return 1;
  if (kind === 'main_post_revision') return 1;
  if (kind.startsWith('opt')) return 10 + parseInt(kind.replace(/^opt(\d+).*/, '$1') || '0', 10);
  return 1000;
}

// Load decision tree JSON + apply per-task override patches.
//
// `clip_router_hints`         → routerHintsByClipId
// `decision_tree_node_patches` → shallow-merged into matching tree nodes
//                                BEFORE indexing. Used to repair tree-baked
//                                metadata (item_name, item_verbal_variants,
//                                covered_item_names, covered_item_aliases)
//                                when the cart's item_id was remapped.
function _loadTreeAndOverrides(treePath, overridesPath) {
  const tree = JSON.parse(fs.readFileSync(treePath, 'utf8'));
  const routerHintsByClipId = new Map();
  if (!overridesPath) return { tree, routerHintsByClipId };

  try {
    const data = JSON.parse(fs.readFileSync(overridesPath, 'utf8'));
    const hints = (data && data.clip_router_hints) || {};
    for (const [cid, hint] of Object.entries(hints)) {
      if (typeof hint === 'string' && hint.trim()) {
        routerHintsByClipId.set(cid, hint.trim());
      }
    }
    const patches = (data && data.decision_tree_node_patches) || {};
    const patchedNodes = [];
    for (const [nodeId, patch] of Object.entries(patches)) {
      const node = tree.nodes && tree.nodes[nodeId];
      if (!node) {
        console.warn(
          `[ClipPlayer] WARNING: decision_tree_node_patches references `
          + `unknown node "${nodeId}" — skipping`
        );
        continue;
      }
      if (!patch || typeof patch !== 'object') continue;
      for (const [k, v] of Object.entries(patch)) {
        node[k] = v;
      }
      patchedNodes.push(nodeId);
    }
    console.log(
      `[ClipPlayer] loaded ${routerHintsByClipId.size} router hint(s) and `
      + `patched ${patchedNodes.length} tree node(s) from ${overridesPath}`
      + (patchedNodes.length ? ` (${patchedNodes.join(', ')})` : '')
    );
  } catch (e) {
    console.warn(
      `[ClipPlayer] WARNING: could not read taskOverrides at ${overridesPath}: ${e.message}`
    );
  }
  return { tree, routerHintsByClipId };
}

// Derive the DEFAULT_ELIGIBLE_OPTIONS block injected into the system
// prompt: for each expected_cart item, list every required option group
// whose menu default is non-null AND whose cart value matches the
// default. Those are the groups `just_the_defaults` can legitimately
// answer for — anything else requires a specific opt clip or
// missing_answer.
function _deriveDefaultEligibleOptions(taskConfigPath) {
  if (!taskConfigPath) return '(none)';
  let cfg;
  try {
    cfg = JSON.parse(fs.readFileSync(taskConfigPath, 'utf8'));
  } catch (e) {
    console.warn(
      `[ClipPlayer] WARNING: could not read taskConfig at ${taskConfigPath}: ${e.message}`
    );
    return '(none)';
  }
  const cart = cfg.expected_cart || [];
  const menuLookup = new Map();
  for (const cat of (cfg.menu_sample && cfg.menu_sample.agent_menu) || []) {
    for (const it of cat.items || []) {
      if (it.base_id) menuLookup.set(it.base_id, it);
    }
  }
  const blocks = [];
  for (const entry of cart) {
    const bid = entry.item_id;
    const item = menuLookup.get(bid);
    if (!item) continue;
    const cartOptions = entry.options || {};
    const eligible = [];
    for (const og of item.option_groups || []) {
      if (!og.required) continue;
      if (og.default == null) continue;
      const chosen = cartOptions[og.name];
      const matches = Array.isArray(chosen)
        ? chosen.length === 1 && chosen[0] === og.default
        : chosen === og.default;
      if (matches) {
        eligible.push(`      - ${og.name} → default "${og.default}"`);
      }
    }
    if (eligible.length) {
      const displayName = item.name || bid;
      blocks.push(`  - ${displayName} (base_id=${bid}):`);
      blocks.push(...eligible);
    }
  }
  const text = blocks.length ? blocks.join('\n') : '(none)';
  console.log(
    `[ClipPlayer] DEFAULT_ELIGIBLE_OPTIONS derived from taskConfig: ` +
    `${blocks.length ? blocks.filter(l => !l.startsWith('      -')).length + ' item(s)' : 'none'}`
  );
  return text;
}

// Build clipsByItem (Map<"s.i", clipNode[]>) and allItems ([[s,i], ...])
// from tree.nodes. Excludes synthetic _yes/_no nodes (handled via the
// abstract "yes"/"no" actions) and terminal nodes.
function _indexClips(treeNodes) {
  const clipsByItem = new Map();
  for (const node of Object.values(treeNodes)) {
    if (node.is_terminal) continue;
    const parsed = parseClipId(node.id);
    if (!parsed) continue;
    if (parsed.kind.endsWith('_yes') || parsed.kind.endsWith('_no')) continue;
    const key = itemKey(parsed.stage, parsed.item);
    if (!clipsByItem.has(key)) clipsByItem.set(key, []);
    clipsByItem.get(key).push({ ...node, _stage: parsed.stage, _item: parsed.item, _kind: parsed.kind });
  }
  for (const arr of clipsByItem.values()) {
    arr.sort((a, b) => _sortKey(a._kind) - _sortKey(b._kind));
  }
  const allItems = [...clipsByItem.keys()]
    .map((k) => k.split('.').map((n) => parseInt(n, 10)))
    .sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  return { clipsByItem, allItems };
}

// Yes/no clip basenames available for this game. The tree's synthetic
// _main_yes / _main_no nodes carry the right `clip` filename, so just
// look up one of them as a probe.
function _findClipBasename(treeNodes, suffix) {
  for (const node of Object.values(treeNodes)) {
    if (node.id && node.id.endsWith(suffix) && node.clip) return node.clip;
  }
  return null;
}

function _findThatsAllClip(clipsByItem) {
  for (const arr of clipsByItem.values()) {
    for (const c of arr) {
      if (c._kind === 'thats_all') return c;
    }
  }
  return null;
}

// Generic (non-sample-unique) clips are injected at startup rather than
// required on disk. The tree's job is to describe THIS sample's customer
// utterances; the shared close-out / yes / no / defaults lines are the
// same audio for every task and live in `order_up_shared/`.
//
// `thats_all` was the one generic still sourced from the tree, so trees
// built before it was added had no close-out at all: `computeValidChoices`
// withholds `next_item` at end-of-order and then found no thats_all clip
// to offer, leaving the router with only yes/no and [FALLBACK] replays.
// It would re-pick an already-played main, the agent heard a second order
// and duplicated the item. Synthesizing the node removes that whole class
// of vintage dependency — the on-disk shape below is what the current
// builder emits, and it holds no sample-specific data.
function _ensureSharedThatsAllNode(tree, clipsByItem, gameIndex, stageIndex) {
  for (const arr of clipsByItem.values()) {
    for (const c of arr) if (c._kind === 'thats_all') return null; // sample has its own
  }
  const id = `g${gameIndex}_s${stageIndex}_i99_thats_all`;
  const node = {
    id,
    clip: 'thats_all.wav',   // resolved via the shared-clips fallback
    text: "That's all, thanks.",
    item_name: null,
    item_verbal_variants: [],
    covered_item_indices: [],
    parent_clip_kind: null,
    transitions: [],
  };
  tree.nodes[id] = node;
  // Index under the i99 pseudo-item, same as a builder-emitted node.
  // Safe for ordering: `nextItemAfter` only advances to items owning a
  // `main` clip, so i99 is skipped and end-of-order detection is intact.
  const key = itemKey(stageIndex, 99);
  if (!clipsByItem.has(key)) clipsByItem.set(key, []);
  clipsByItem.get(key).push({ ...node, _stage: stageIndex, _item: 99, _kind: 'thats_all' });
  return id;
}

function createDecisionTree({ treePath, taskConfigPath, overridesPath, clipsBaseDirOverride } = {}) {
  if (!treePath) throw new Error('createDecisionTree: treePath is required');

  // ── Load + override-patch ────────────────────────────────────────────
  const { tree, routerHintsByClipId } = _loadTreeAndOverrides(treePath, overridesPath);
  const defaultEligibleOptionsText = _deriveDefaultEligibleOptions(taskConfigPath);

  // ── Path resolution ──────────────────────────────────────────────────
  // The tree's `clips_base_dir` is written as a relative path like
  // `artifacts/clips/<task_id>` (the legacy assumption is that cwd ==
  // ARTIFACTS_PATH-parent). When the eval runtime ships separately from
  // the artifact corpus — the audio-benchmarks ↔ FDB split today — cwd is
  // the eval source repo and the relative path resolves to the wrong
  // place. Anchor against $ARTIFACTS_PATH instead when the tree path is
  // relative; absolute paths and an explicit override still take
  // precedence.
  const _rawClipsBase = clipsBaseDirOverride || tree.clips_base_dir;
  const _isAbsolute = path.isAbsolute(_rawClipsBase);
  // tree.clips_base_dir is authored as `artifacts/clips/<task>` — i.e. the
  // legacy assumption is cwd == parent-of-artifacts. $ARTIFACTS_PATH itself
  // already points at the `artifacts/` directory, so anchor against its
  // parent to avoid doubling the segment (`…/artifacts/artifacts/clips/…`).
  const _anchor = _isAbsolute
    ? null
    : (process.env.ARTIFACTS_PATH
        ? path.dirname(process.env.ARTIFACTS_PATH)
        : process.cwd());
  const clipsBaseDir = _isAbsolute
    ? _rawClipsBase
    : path.resolve(_anchor, _rawClipsBase);
  // Shared yes/no fallback dir — sibling of every per-task clips dir at
  // artifacts/clips/order_up_shared/. The decision tree's synthetic _yes /
  // _no nodes carry node.clip="yes.wav"/"no.wav"; when the per-game
  // extractor couldn't find a clean customer-spoken yes/no in the session,
  // the per-task dir won't have that file and we resolve to this shared
  // dir instead.
  const sharedClipsDir = path.resolve(clipsBaseDir, '..', 'order_up_shared');

  function availableClipBasename(basename) {
    if (!basename) return null;
    const primary = path.resolve(clipsBaseDir, basename);
    const shared = path.resolve(sharedClipsDir, basename);
    return (fs.existsSync(primary) || fs.existsSync(shared)) ? basename : null;
  }

  function resolveClipPath(node) {
    const primary = path.resolve(clipsBaseDir, node.clip);
    if (fs.existsSync(primary)) return primary;
    const fallback = path.resolve(sharedClipsDir, node.clip);
    if (fs.existsSync(fallback)) return fallback;
    throw new Error(
      `clip not found in per-task dir (${primary}) or shared dir (${fallback})`
    );
  }

  // ── Static metadata derivation ───────────────────────────────────────
  const taskId = tree.task_id;
  const initialNodeId = tree.initial_node;
  const initialNoiseRotation = tree.initial_noise_rotation;
  const gameIndex = (parseClipId(initialNodeId) || { game: 0 }).game;

  const { clipsByItem, allItems } = _indexClips(tree.nodes);

  const yesClipBasename = _findClipBasename(tree.nodes, '_main_yes');
  const noClipBasename = _findClipBasename(tree.nodes, '_main_no');
  // Customer's "just the defaults" fallback. The basename is available only
  // when the task has a default-eligible option and the audio exists in the
  // per-task or shared clip directory.
  const justTheDefaultsClipBasename = defaultEligibleOptionsText !== '(none)'
    ? availableClipBasename('just_the_defaults.wav')
    : null;
  // Customer's "I only want one" reply to a quantity-additive question.
  // Not gated on task config: every order in this benchmark intends
  // quantity 1 per item, so this applies to any stage whose audio exists.
  const onlyOneClipBasename = availableClipBasename('only_one.wav');
  // Multi-choice / compound-question fallbacks. The router can pick these
  // when their shared audio is present and the agent enumerates a two-option pair
  // ("Did you mean A or B?", "Do you want C or D?") that neither yes/no
  // nor a specific opt clip cleanly answers. Positional semantics:
  // `first_option` = the first item the agent names, `second_option` =
  // the second. The router decides which one matches ground truth using
  // the agent's `heard=` transcript + the played item context.
  const firstOptionClipBasename = availableClipBasename('the_first_option.wav');
  const secondOptionClipBasename = availableClipBasename('the_second_option.wav');
  // Customer's "that's all" line — emitted by the build pipeline when the
  // source playthrough contains a closing utterance. Used at end-of-order
  // in place of next_item. Optional: if no such clip was extracted, the
  // adapter falls back to yes/no based on the framing of the agent's
  // move-on question.
  const _stageIndex = (parseClipId(initialNodeId) || { stage: 0 }).stage;
  const _injectedThatsAll = _ensureSharedThatsAllNode(
    tree, clipsByItem, gameIndex, _stageIndex
  );
  if (_injectedThatsAll) {
    console.log(
      `[ClipPlayer] injected shared close-out clip ${_injectedThatsAll} `
      + '(tree has no sample-specific thats_all)'
    );
  }
  const thatsAllClip = _findThatsAllClip(clipsByItem);

  // ── Progression state ────────────────────────────────────────────────
  // Dynamic-valid-set state machine. The router's `play` tool can choose
  // "next_item"/"yes"/"no"/"missing_answer"/"just_the_defaults" or a
  // specific clip_id; the adapter validates the choice against
  // computeValidChoices() each turn.
  //
  // playedClips resets on stage transitions (the user explicitly wants
  // the valid set to be per-stage so the router can't pick a clip from
  // items in past stages).
  let currentStage = -1;            // -1 = pre-start; advances to 0 on first next_item
  let currentItem = -1;             // -1 = pre-start
  let playedClips = new Set();      // clip_ids played in CURRENT stage
  let lastPlayedClipId = null;      // for fallback replay + pendingCorrection lookup
  let conversationEnded = false;
  let lastValidChoices = null;      // cached computeValidChoices() result (debug + mainClipItemExtras)

  // ── Acknowledgement tracker ─────────────────────────────────────────
  // Per-stage tracker that answers "which ackable clips has the agent
  // acknowledged so far?" — populated by the router on each play(...)
  // call via a new `acknowledged` argument, normalized (restatement→main)
  // and cross-checked against the played set. Surfaced in every tool
  // response as `acknowledged` / `unacknowledged`; used by the adapter
  // to substitute `thats_all` when the router tries to close out with
  // items still un-ack'd.
  //
  // ackablePlayed: append-only list, in play order, of clips whose
  //   kind maps to one of the 4 ackable buckets (main | opt |
  //   main_corr | opt_corr).
  // ackedShortIds: subset of ackablePlayed[].clip_id_short that have
  //   been ack'd this stage.
  // lastRestatementByItem: item_index → most recent restatement short
  //   id played for that item (used for the premature-close substitution).
  let ackablePlayed = [];
  let ackedShortIds = new Set();
  let lastRestatementByItem = new Map();

  function _ackableKindBucket(kind) {
    if (!kind) return null;
    if (kind === 'main') return 'main';
    if (kind === 'main_corr') return 'main_corr';
    if (/^opt\d+$/.test(kind)) return 'opt';
    if (/^opt\d+_corr$/.test(kind)) return 'opt_corr';
    return null;
  }

  function advanceTo(stage, item) {
    if (stage !== currentStage) {
      playedClips = new Set();
      ackablePlayed = [];
      ackedShortIds = new Set();
      lastRestatementByItem = new Map();
    }
    currentStage = stage;
    currentItem = item;
  }
  function markPlayed(clipId) {
    if (!clipId) return;
    playedClips.add(clipId);
    lastPlayedClipId = clipId;
    // Also fold into the ackable tracker. clipId here is the LONG in-tree
    // id (e.g. `g8_s2_i1_main`); we store SHORT sub-clip ids because
    // that's what the router sees (in `played_items[].sub_clip_id`) and
    // reports in its `acknowledged` argument.
    //
    // A single played clip can represent MULTIPLE ackable spans when the
    // dedupe pass merged cross-item spans into it (compound-main clips
    // covering "A and B" in one utterance, or absorbing a same-turn
    // `main_corr`). The tree node records the merged breakdown in
    // `covered_item_indices` / `covered_item_kinds` / `covered_item_names`
    // — enumerate those to register one ackable entry per pre-merge
    // span, so the router can independently ack each item the audio
    // covered.
    const parsed = parseClipId(clipId);
    if (!parsed) return;
    const node = tree.nodes[clipId] || {};
    // Fast-path for restatements — not ackable themselves, but they
    // become the substitution target if the router prematurely closes.
    if (/^restatement\d*$/.test(parsed.kind || '')) {
      lastRestatementByItem.set(parsed.item, shortenClipId(clipId));
      return;
    }
    // Non-main-family clips (opt / opt_corr) don't populate
    // covered_item_*; fall back to the clip's own (item, kind) pair.
    const coveredIdxs = Array.isArray(node.covered_item_indices) ? node.covered_item_indices : null;
    const coveredKinds = Array.isArray(node.covered_item_kinds) ? node.covered_item_kinds : null;
    const coveredNames = Array.isArray(node.covered_item_names) ? node.covered_item_names : null;
    const fbClipsByIdx = new Map();
    for (const fc of (node.fallback_clips || [])) {
      if (fc && Number.isInteger(fc.item_index)) fbClipsByIdx.set(fc.item_index, fc);
    }
    if (coveredIdxs && coveredIdxs.length > 0 && parsed.kind && parsed.kind.startsWith('main')) {
      // Multi-span (or single-span) main-family clip. Register one
      // entry per covered item + its sub-kind ('corr' → main_corr;
      // otherwise main).
      for (let k = 0; k < coveredIdxs.length; k++) {
        const idx = coveredIdxs[k];
        const subKindRaw = (coveredKinds && coveredKinds[k]) || 'main';
        const subKindSuffix = subKindRaw === 'corr' ? 'main_corr' : 'main';
        const subClipId = `i${idx}_${subKindSuffix}`;
        // De-dup: if we've already recorded this exact sub-clip_id
        // this stage (e.g. i0_main played, then a later compound clip
        // re-covers item 0 in kind=main), skip so the ack state stays
        // stable rather than growing a duplicate un-ack'd entry.
        if (ackablePlayed.some(e => e.clip_id === subClipId)) continue;
        // Sub-span text: prefer the per-item text from fallback_clips
        // (e.g. "A chicken tikka masala") over the whole compound's
        // text ("A chicken tikka masala and Crème Brûlée. Spice level
        // mild.") so ack entries surface the actual span, not the
        // concatenated compound.
        const fbClip = fbClipsByIdx.get(idx);
        const subText = (fbClip && fbClip.text) || node.text || '';
        ackablePlayed.push({
          clip_id: subClipId,
          item_index: idx,
          kind_bucket: subKindSuffix,
          canonical_name: (coveredNames && coveredNames[k]) || (fbClip && fbClip.canonical_name) || node.item_name || '',
          text: subText,
          // Source clip whose audio actually contains this span. For a
          // compound clip that absorbed a sibling's corr, sub_clip_id
          // like `i0_main_corr` has no standalone node — the audio
          // lives inside `g8_s2_i1_main`. Substitution fallback needs
          // this to replay the real audio.
          source_clip_id_long: clipId,
        });
      }
      return;
    }
    // No covered-* metadata: single-span opt / opt_corr / bare main.
    const bucket = _ackableKindBucket(parsed.kind);
    if (!bucket) return;
    const shortId = shortenClipId(clipId);
    if (ackablePlayed.some(e => e.clip_id === shortId)) return;
    ackablePlayed.push({
      clip_id: shortId,
      item_index: parsed.item,
      kind_bucket: bucket,
      canonical_name: node.item_name || '',
      text: node.text || '',
      source_clip_id_long: clipId,
    });
  }

  function _partitionAck(entries) {
    const acked = [], unacked = [];
    for (const e of entries) {
      (ackedShortIds.has(e.clip_id) ? acked : unacked).push(e);
    }
    return { acked, unacked };
  }

  // Item indices a clip REFERENCES. A compound main records them in
  // `covered_item_indices`; every other clip kind (opt / bare main)
  // carries exactly one, encoded in its own id (`i<N>_<kind>`). Non-ackable
  // kinds still reference an item this way — an `opt` clip answers a
  // question about a specific item — so they are valid generic-ack scope.
  function _itemsReferencedByClip(clipId) {
    const out = new Set();
    if (!clipId) return out;
    const node = tree.nodes[clipId];
    if (node && Array.isArray(node.covered_item_indices) && node.covered_item_indices.length > 0) {
      for (const i of node.covered_item_indices) {
        if (Number.isInteger(i)) out.add(i);
      }
      return out;
    }
    const parsed = parseClipId(clipId);
    if (parsed && Number.isInteger(parsed.item)) out.add(parsed.item);
    return out;
  }

  // Normalize the router's raw `acknowledged` list into a set of short
  // ids that live in ackablePlayed and aren't already ack'd. Restatement
  // ids map back to the parent main; unknown / already-ack'd / non-
  // ackable ids get dropped (with a rejection reason for the log).
  //
  // The `generic` sentinel means "and everything else the last clip
  // covered". Asking the router to ENUMERATE those ids was a reliability
  // problem — it would send [] or a partial list, the item stayed un-ack'd,
  // and the adapter then replayed a clip the agent had in fact
  // acknowledged. The router now reports `generic` and we expand it here.
  //
  // It COMBINES with specific ids: `["i0_main", "generic"]` is the normal
  // output for a follow-up targeted at i0 ("what size for your burger?") —
  // the named item plus its peers. So the two forms are unioned, not
  // mutually exclusive. Deterministic beats model bookkeeping.
  const GENERIC_ACK = 'generic';
  // `explicit_question_target` marks a question that EXPLICITLY named the
  // item it is about ("do you want your vindaloo mild or spicy?", "mustard
  // on your burger?"). Naming the item is a disambiguation move, which only
  // makes sense when there is something to disambiguate it FROM — so it
  // implies the agent heard the rest of the compound order too, and the
  // expansion acks the peers.
  //
  // An IMPLICIT reference ("mild or spicy?", "mustard on that?") gets the
  // item's own id and NO sentinel: it identifies one item without evidence
  // the agent registered any other. That difference is the whole point of
  // the split — treating the implicit case as an expansion is what silently
  // dropped a second item from the cart.
  //
  // `generic` stays reserved for a bare move-on with no item or option
  // mentioned at all ("got it", "anything else?").
  const EXPLICIT_QUESTION_TARGET_ACK = 'explicit_question_target';
  const EXPANDING_ACKS = new Set([GENERIC_ACK, EXPLICIT_QUESTION_TARGET_ACK]);

  function normalizeAckList(rawList) {
    const accepted = [];
    const rejected = [];
    if (!Array.isArray(rawList)) return { accepted, rejected };

    const expandTokens = rawList
      .filter(raw => typeof raw === 'string'
        && EXPANDING_ACKS.has(raw.trim().toLowerCase()))
      .map(raw => raw.trim().toLowerCase());
    const hasExpansion = expandTokens.length > 0;

    const inPlayed = new Set(ackablePlayed.map(e => e.clip_id));
    for (const raw of rawList) {
      if (typeof raw !== 'string' || !raw.trim()) continue;
      // Sentinels are handled by the expansion below, not here.
      if (EXPANDING_ACKS.has(raw.trim().toLowerCase())) continue;
      const short = shortenClipId(raw.trim());
      const parsed = parseClipId(`g0_s0_${short}`); // synthesize long to reuse parser
      let target = short;
      let mapped_from = null;
      if (parsed && /^restatement\d*$/.test(parsed.kind || '')) {
        mapped_from = short;
        target = `i${parsed.item}_main`;
      }
      if (!inPlayed.has(target)) {
        rejected.push({ raw, target, reason: 'not_in_ackable_played' });
        continue;
      }
      if (ackedShortIds.has(target)) {
        rejected.push({ raw, target, reason: 'already_acked' });
        continue;
      }
      accepted.push({ short: target, mapped_from });
      ackedShortIds.add(target);
    }

    // `generic` = "and everything else the last played clip covered".
    // Scoped to that clip's items on purpose: items left un-ack'd from
    // EARLIER turns are not swept up, because the agent never addressed
    // them and acking them would be the silent-drop the false-ack rule
    // exists to prevent. Runs after the explicit loop so the two union
    // cleanly and already-acked entries are skipped rather than rejected.
    if (hasExpansion) {
      // Report which sentinel drove the expansion. `target` wins the label
      // when both appear, since a targeted ack is the more specific claim.
      const via = expandTokens.includes(EXPLICIT_QUESTION_TARGET_ACK)
        ? EXPLICIT_QUESTION_TARGET_ACK
        : GENERIC_ACK;
      const scopeItems = _itemsReferencedByClip(lastPlayedClipId);
      for (const entry of getUnackedEntries()) {
        if (!scopeItems.has(entry.item_index)) continue;
        accepted.push({ short: entry.clip_id, mapped_from: via });
        ackedShortIds.add(entry.clip_id);
      }
    }
    return { accepted, rejected };
  }

  function recordAcks(rawList) {
    return normalizeAckList(rawList);
  }

  function getAckedEntries() {
    return _partitionAck(ackablePlayed).acked;
  }
  function getUnackedEntries() {
    return _partitionAck(ackablePlayed).unacked;
  }
  function getUnackedCount() {
    return getUnackedEntries().length;
  }
  function getUnackedIds() {
    return getUnackedEntries().map(e => e.clip_id);
  }

  // Restatement node ids for `item` in the CURRENT stage, ordered by
  // their trailing number. Numbering is sparse in real data (seen:
  // restatement1, 2, 3, 5, 6), so sort numerically rather than assuming
  // a dense sequence or lexical order (which would put 10 before 2).
  function _restatementsForItem(item, stage) {
    const st = Number.isInteger(stage) && stage >= 0 ? stage : currentStage;
    const re = new RegExp(`^g${gameIndex}_s${st}_i${item}_restatement(\\d*)$`);
    const out = [];
    for (const nid of Object.keys(tree.nodes)) {
      const m = nid.match(re);
      if (m) out.push({ id: nid, n: m[1] === '' ? 0 : Number(m[1]) });
    }
    out.sort((a, b) => a.n - b.n);
    return out.map((x) => x.id);
  }

  // THE single answer to "an item is still un-ack'd — what audio
  // re-surfaces it?". Every trigger routes here: the router picking
  // `thats_all` (premature close), picking `next_item` with items
  // outstanding, or explicitly asking for `fallback`.
  //
  // This used to be three separate implementations in the adapter with
  // three different preference orders, so the same stranded item got a
  // different clip depending on which choice the router happened to
  // make — and each copy was missing what another had (premature close
  // never looked at `fallback_clips`; explicit fallback never looked at
  // restatements; `next_item` could re-say an ALREADY-ACK'D item). The
  // triggers still differ in consequence — `thats_all` ends the
  // conversation, `next_item` advances the cursor, `fallback` is an
  // explicit replay — so their call sites stay distinct. Choosing the
  // clip is not a per-trigger decision and lives here.
  //
  // Order, applied to the OLDEST un-ack'd entry so the router hears
  // items re-stated in the order the customer originally said them:
  //
  //   1. a `restatementN` for that item — purpose-recorded for exactly
  //      this ("I also wanted the X"), so it always wins when present
  //   2. the per-item sub-clip of the compound that carries the item's
  //      audio (`fallback_clips[]`), which re-says just that item
  //      instead of the whole compound
  //   3. the whole source clip — correct but blunt; on a compound this
  //      re-says items the agent already ack'd, which is the behaviour
  //      1 and 2 exist to avoid
  //   4. the entry's own id as a tree key, for bare single-span clips
  //
  // Returns `{ node, source }` (`source` is for logging) or null when
  // nothing is resolvable, in which case the caller lets the router's
  // original choice stand.
  function resurfaceUnackedItem() {
    const un = getUnackedEntries();
    if (un.length === 0) return null;
    const oldest = un[0];

    // 1. Restatement — any that EXISTS in the tree for this item, not
    // just one already played. `lastRestatementByItem` only records
    // restatements that have been played, so relying on it alone meant a
    // perfectly good `restatementN` sitting unplayed in the tree was
    // never chosen and the item fell through to a sub-clip. Prefer the
    // lowest-numbered unplayed one (restatement1 is the customer's first
    // re-statement; higher numbers are for repeated failures), and only
    // if every one has already played fall back to replaying the most
    // recent, which is better than nothing.
    // Scope the search to the stage that actually carried this item's
    // audio, taken from the entry itself. `currentStage` is -1 until the
    // first advanceTo(), and a restatement always lives in the same stage
    // as the clip it re-states, so the entry is the more reliable source.
    const srcParsed = oldest.source_clip_id_long
      ? parseClipId(oldest.source_clip_id_long)
      : null;
    const rests = _restatementsForItem(
      oldest.item_index,
      srcParsed ? srcParsed.stage : undefined,
    );
    const unplayed = rests.find((id) => !playedClips.has(id));
    if (unplayed) {
      return { node: tree.nodes[unplayed], source: 'restatement' };
    }
    if (rests.length > 0) {
      const rShort = lastRestatementByItem.get(oldest.item_index);
      const rLong = rShort
        ? expandClipId(rShort, gameIndex, srcParsed ? srcParsed.stage : currentStage)
        : rests[rests.length - 1];
      if (tree.nodes[rLong]) {
        return { node: tree.nodes[rLong], source: 'restatement_replay' };
      }
    }

    // The clip whose audio actually contains this span. For a
    // compound-merged sub-span (e.g. sub id `i0_main_corr` inside
    // `g8_s2_i1_main`) this is the real playable node — the sub id
    // alone won't resolve, because the standalone node was folded away.
    const srcId = oldest.source_clip_id_long;
    const srcNode = srcId ? tree.nodes[srcId] : null;

    // 2. Per-item sub-clip of that source clip. Matched on item_index
    // against the entry we already know is un-ack'd, so this can never
    // surface an item the agent has already acknowledged.
    if (srcNode && Array.isArray(srcNode.fallback_clips)) {
      const fc = srcNode.fallback_clips.find(
        (x) => x && x.item_index === oldest.item_index && x.clip
      );
      if (fc) {
        return {
          node: {
            ...srcNode,
            // Mangled id is for logging/traceability only — it is not a
            // tree node. `_realNodeId` keeps the resolvable one so
            // markPlayed records a real id and the NEXT turn's
            // dt.nodes[lastPlayedClipId] still hits (without it the
            // following turn silently degrades to whole-compound replay).
            id: `${srcId}__unack_i${fc.item_index}`,
            _realNodeId: srcId,
            clip: fc.clip,
            text: fc.text || srcNode.text,
          },
          source: `sub_i${fc.item_index}`,
        };
      }
    }

    // 3. Whole source clip.
    if (srcNode) return { node: srcNode, source: 'source_clip' };

    // 4. Bare single-span clip whose sub id IS the node id.
    const long = expandClipId(oldest.clip_id, gameIndex, currentStage);
    return tree.nodes[long] ? { node: tree.nodes[long], source: 'self' } : null;
  }
  function endConversation() { conversationEnded = true; }

  // ── Reachability helpers ─────────────────────────────────────────────
  function nextItemAfter(s, i) {
    // Find the next item that owns a main clip AND whose main hasn't been
    // played yet in this stage's playedClips set. Three skip cases:
    //   1. Items with only sub-clips (opt) and no main — arises when
    //      one customer utterance orders multiple items at once
    //      (item_indices=[0,1,2]) so only item 0 owns the main clip while
    //      items 1+ contribute only response clips. Advancing into one of
    //      those has nothing to play; jump to the next main-owning item.
    //   2. Items whose main is ALREADY in playedClips. After a specific-
    //      clip pick (opt/corr) snaps the cursor backward, the subsequent
    //      next_item must NOT replay a main we've already shipped — it
    //      should jump forward to the next unplayed main (potentially
    //      crossing into the next stage). playedClips resets on stage
    //      boundary, so items in later stages are always considered
    //      unplayed from this cursor's perspective.
    //   3. The cursor (s, i) may not be a member of allItems — happens
    //      when a multi-item main covered higher item indices and the
    //      cursor advanced to e.g. (0, 1) but only (0, 0) exists in
    //      clipsByItem (the i=1 main got absorbed during cross-item
    //      merge). Treat (s, i) as a POSITION and walk forward to the
    //      first allItems entry strictly greater than it; do NOT require
    //      an exact match.
    let startIdx = -1;
    for (let k = 0; k < allItems.length; k++) {
      const [ss, ii] = allItems[k];
      if (ss > s || (ss === s && ii > i)) { startIdx = k; break; }
    }
    if (startIdx < 0) return undefined;
    for (let j = startIdx; j < allItems.length; j++) {
      const [ns, ni] = allItems[j];
      const items = clipsByItem.get(itemKey(ns, ni)) || [];
      const main = items.find((c) => c._kind === 'main');
      if (!main) continue;
      // playedClips only tracks the CURRENT stage. For same-stage items
      // we must consult it; for later-stage items the set is irrelevant
      // (will be reset on transition) so just take the first match.
      if (ns === currentStage && playedClips.has(main.id)) continue;
      return allItems[j];
    }
    return null;
  }

  function _siblingByKind(items, kind) {
    return items.find((x) => x._kind === kind) || null;
  }

  function resolveItemClip(c, items, opts) {
    const k = c._kind;
    const isCurrentItem = !!(opts && opts.isCurrentItem);
    // The BARE `_corr` (no numeric suffix) is NEVER directly selectable —
    // it auto-plays via _handleNextItem when its parent plays. The
    // NUMBERED chained kinds (`_corr_2`, `_corr_3` …) are router-selectable
    // restatements of the correction; they fall through to the normal
    // selectable-clip path so the parent_clip_kind chain gate keeps them
    // hidden until the prior corr in the chain has played.
    if (isAutoCorrKind(k)) return null;
    // _post_revision clips are also NEVER directly selectable. They serve
    // two specific purposes: (1) the fallback target when the agent asks
    // an unanswerable question, and (2) the substitute audio when the agent
    // asks the customer to restate. The router should NOT pick them via
    // play({choice: clip_id}) — for confirmation questions, yes/no is the
    // right answer; for "anything else?", next_item is the right answer.
    if (k.endsWith('_post_revision')) return null;
    if (playedClips.has(c.id)) {
      const corr = _siblingByKind(items, `${k}_corr`);
      if (corr) {
        if (playedClips.has(corr.id)) {
          // Both the original AND its correction have been played. Expose
          // the CORR as a fallback in the original's slot — the corr
          // represents the customer's CURRENT intent (the original is
          // cancelled), so re-asks of the same question should route to
          // the corrected audio, not the original's now-superseded answer.
          return { clip: corr, isFallback: true, substitutedFromParent: c };
        }
        // Corr exists but hasn't played yet — pending corr will auto-play
        // via next_item handling. Don't expose this slot to the router.
        return null;
      }
      // No correction sibling: expose THIS clip again, marked as a
      // fallback re-play option. The router sees the actual transcript
      // and decides intentionally instead of relying on an opaque
      // "fallback" action that doesn't surface the audio's content.
      return { clip: c, isFallback: true };
    }
    // Conversational-chain gate (CURRENT-ITEM ONLY): for the in-progress
    // item, RESP/RESTATEMENT clips can be picked by the router only if
    // their parent_clip_kind has been played. Back-references to PREVIOUS
    // items skip this gate (the customer can call back to an earlier item
    // via any of its clips intentionally — that's a separate routing
    // capability).
    //
    // Only RESTATEMENT clips are gated. OPT clips are not: their
    // parent_clip_kind records WHERE in the source recording the option was
    // uttered, not a prerequisite. The eval agent may ask the option
    // question directly ("with extra cheese or extra mushrooms?") without
    // first asking whatever prompted it in the recording, and in that case
    // the opt clip is the right answer even though its recorded parent
    // never played.
    //
    // NOTE the gate is narrower than it looks: bare `_corr` and
    // `_post_revision` clips return null above, before reaching here, so
    // this only ever governs directly-selectable clips. Correction ORDERING
    // is enforced separately by pendingCorrection(), which keys on the
    // `_corr` suffix and is kind-agnostic.
    //
    // Numbered chains (`opt1_corr_2`, `opt1_corr_3`) are `opt`-prefixed and
    // therefore NOT gated, despite the comment above the auto-corr check
    // claiming the chain gate keeps them hidden. That intent is already
    // dead in practice; restoring it is a separate decision.
    if (isCurrentItem) {
      const parentKind = c.parent_clip_kind;
      const inRestatementChain = k.startsWith('restatement');
      if (inRestatementChain && parentKind && parentKind !== 'main') {
        const parentClip = items.find((x) => x._kind === parentKind);
        if (parentClip && !playedClips.has(parentClip.id)) {
          return null;
        }
      }
    }
    return { clip: c };
  }

  function clipDescription(node, parsed, info) {
    const text = (node.text || '').trim();
    const k = node._kind;
    const subFrom = info?.substitutedFromParent?._kind;
    const subBit = subFrom ? ` [substitutes for ${subFrom} now that it's been played]` : '';
    // Per-task router hint (optional). Appended unconditionally to whichever
    // description branch fires below — operator-curated stage-specific cue
    // the generic kind-based description can't derive (e.g. customer
    // mispronunciation the agent is expected to surface).
    const hint = routerHintsByClipId.get(node.id);
    const hintBit = hint ? ` [ROUTER HINT — ${hint}]` : '';
    // Fallback marker: clip is already played; expose for re-selection.
    const fallbackBit = (info?.isFallback && !subFrom)
      ? ` [FALLBACK — this clip already played; re-select only if the agent asked to repeat or referenced this item again]`
      : '';
    // The item_name is the CANONICAL menu name attached at tree build
    // time. The clip transcript itself may be misheard by the per-clip
    // whisper pass ("Thai and some"), so the router needs the item name
    // to map agent confirmations to the right clip.
    //
    // `item_verbal_variants` are customer-side aliases for the same menu
    // item (e.g. "Toast with Berries" → "French Toast"). The agent may
    // use EITHER the canonical name or any variant when asking for
    // confirmation; the router needs to recognise all of them as
    // referring to this item, otherwise "Do you mean French toast?"
    // reads as a wrong-item guess when the clip transcript happens to
    // say "toast with berries".
    const itemName = (node.item_name || '').trim();
    const verbalVariants = Array.isArray(node.item_verbal_variants)
      ? node.item_verbal_variants.filter((v) => v && v.trim() && v.trim() !== itemName)
      : [];
    const variantBit = verbalVariants.length
      ? ` (also referred to as ${verbalVariants.map((v) => `"${v}"`).join(' / ')})`
      : '';
    const itemLabel = itemName
      ? `"${itemName}"${variantBit}`
      : `item ${parsed.item}`;
    if (k === 'main') {
      return `Customer's primary line for ${itemLabel}. The audio says (verbatim): "${text}".${subBit}${fallbackBit}${hintBit}`;
    }
    if (k === 'main_corr') {
      return `Customer self-correction REPLACING this item's prior intent (the original order is cancelled). The corrected request is: "${text}".${subBit}${fallbackBit}${hintBit}`;
    }
    if (k === 'main_post_revision') {
      return `Repeat just the corrected portion of ${itemLabel}'s main: "${text}".${subBit}${fallbackBit}${hintBit}`;
    }
    if (k.startsWith('opt') && !isCorrKind(k) && !k.endsWith('_post_revision')) {
      // Surface the recorded trigger question when the pipeline captured
      // one. The legacy `resp` branch used to do this and `opt` did not,
      // so an opt clip could hide the very question it answers from the
      // router. Folded in here when `resp` was retired.
      const q = node.parent_question_text
        ? ` (typically when the agent asks: "${node.parent_question_text}")`
        : '';
      return `Customer-stated option/modifier for ${itemLabel}${q}: "${text}".${subBit}${fallbackBit}${hintBit}`;
    }
    if (isCorrKind(k)) {
      const parentKind = stripCorrSuffix(k);
      const restatementBit = isAutoCorrKind(k)
        ? ''
        : ' [restatement of the original correction — pick only if the agent re-asks or expresses uncertainty about the corrected value]';
      return `Customer correction for ${itemLabel} (replaces ${parentKind}): "${text}".${subBit}${restatementBit}${fallbackBit}${hintBit}`;
    }
    if (k.endsWith('_post_revision')) {
      return `Customer's corrected portion for ${itemLabel}: "${text}".${subBit}${fallbackBit}${hintBit}`;
    }
    return `${text}${hintBit}`;
  }

  function mainClipItemExtras(node) {
    if (!node || !node.id || node._synthetic) return {};
    const parsed = parseClipId(node.id);
    const kind = parsed && parsed.kind;
    // Treat any "main" / "main_corr" / "main_corr_post_revision" as the
    // item-introduction event. _corr also names the same item, just
    // post-correction.
    if (!kind || !kind.startsWith('main')) return {};
    const names = Array.isArray(node.covered_item_names)
      ? node.covered_item_names.filter(Boolean)
      : (node.item_name ? [node.item_name] : []);
    if (names.length === 0) return {};
    const aliases_per_item = Array.isArray(node.covered_item_aliases)
      ? node.covered_item_aliases
      : (node.item_verbal_variants ? [node.item_verbal_variants] : []);
    // Flatten aliases across items for the agent — dedupe in JS to keep
    // the payload tight.
    const aliases_flat = Array.from(new Set(
      (aliases_per_item || []).flat().filter(Boolean)
    ));
    const restatement_available = (lastValidChoices || []).some(
      c => c && c.kind === 'clip' && /restatement\d*$/.test(c.choice || '')
    );
    // Compound-main payload: when the merged clip's audio names 2+
    // distinct items in one utterance ("A and B", or "A — actually B
    // — and C"), the router needs a per-item structured breakdown so
    // it can spot partial-ack failures. Each entry mirrors the
    // sub-clip identity that WOULD have existed if the merger hadn't
    // absorbed it: item_index, canonical_name, aliases, and the
    // sub_kind ("main" / "corr") indicating whether that item's
    // contribution to the audio was a fresh order line or an
    // embedded correction.
    const kinds_per_item = Array.isArray(node.covered_item_kinds)
      ? node.covered_item_kinds
      : [];
    const idxs_per_item = Array.isArray(node.covered_item_indices)
      ? node.covered_item_indices
      : [];
    const parsed_id = parseClipId(node.id) || {};
    const is_compound = names.length >= 2;
    const played_items = names.map((n, k) => {
      const item_index = idxs_per_item[k];
      const sub_kind = kinds_per_item[k] || 'main';
      // Synthetic sub-clip id: what the un-merged tree would have
      // called this item's audio segment. Emitted in the same short
      // form as `valid_choices[].choice` (e.g. `i0_main_corr`), so
      // the router can reference sub-clip ids in the same namespace.
      const sub_kind_suffix = sub_kind === 'corr' ? 'main_corr' : 'main';
      const sub_clip_id = item_index != null
        ? `i${item_index}_${sub_kind_suffix}`
        : null;
      return {
        item_index,
        canonical_name: n,
        aliases: (aliases_per_item[k] || []).filter(Boolean),
        sub_kind,
        sub_clip_id,
      };
    });
    // Two disjoint guidance strings — single-item and compound-clip
    // rules aren't a superset/subset relationship, they diverge on:
    //   • move-on prompts ("Got it, anything else?") — single: advance;
    //     compound: only advance once every item in played_items is
    //     ack'd (a bare move-on before that is a partial-ack signal).
    //   • yes-confirmation scope — single: `yes` iff the agent named
    //     THE item; compound: `yes` only if the agent named EVERY item.
    //   • restatement — single: never restate an ack'd item; compound:
    //     restate any un-ack'd item, even if a sibling item WAS ack'd.
    // Emit one or the other, not a concatenation, so the router isn't
    // reconciling two overlapping rulebooks per turn.
    const single_item_guidance = (
      "Item just ordered: see `played_items` (1 entry — canonical_name " +
      "+ aliases). Restatement clips ONLY answer ITEM-IDENTITY " +
      "questions — agent hasn't yet committed to which item the " +
      "customer ordered (\"did you mean X?\", \"X or something else?\", " +
      "\"what was that?\", \"sorry, didn't catch it\"). When " +
      "restatement_available is true AND the agent's next utterance is " +
      "an identity question of that shape, prefer the restatement " +
      "clip over `yes` / `no`. NEVER pick a restatement when the agent " +
      "has ALREADY ACKNOWLEDGED the item and is now asking an OPTION " +
      "question on it (\"got it, the X — what size/flavor/spice?\", " +
      "\"X. Small or large?\", \"X. Lime, lemon, or plain?\") — " +
      "re-saying the item name adds nothing the agent doesn't already " +
      "know. The right answer to an option question is the opt " +
      "clip whose description names ONE of the agent's enumerated " +
      "choices; if no such clip exists, route to missing_answer. Same " +
      "applies when the agent's utterance just confirms the item and " +
      "moves on (\"Got it, all set with the X. Anything else?\") — " +
      "that's a move-on prompt, route to next_item, not the " +
      "restatement. Use `yes` only when the agent's guess unambiguously " +
      "matches the played item's canonical_name or one of its aliases " +
      "AND no restatement clip is listed."
    );
    // Inline the canonical names in the guidance so the router can
    // reason about the item set without cross-referencing another
    // field. Kept short — the full breakdown is still in `played_items`.
    const names_inline = names.map((n) => `"${n}"`).join(", ");
    const compound_guidance = (
      "COMPOUND CLIP — this audio names " + names.length + " items in " +
      "one customer utterance: " + names_inline + ". Route per which of " +
      "these items the agent's next utterance references:\n" +
      "  • PARTIAL ACK (agent named some of the items above but not " +
      "others, OR the customer's utterance trailed off): route to the " +
      "un-mentioned item's restatement / fallback clip. Do NOT pick " +
      "`next_item`, `no`, or `thats_all` yet — even if the agent " +
      "layered a move-on phrase (\"...anything else?\") on top of a " +
      "single-item ack, the un-mentioned item still needs to land or " +
      "it will be silently dropped from the cart.\n" +
      "  • FULL ACK (agent named or ack'd every one of the items " +
      "above, or gave a bare move-on like \"Got it, anything else?\" " +
      "with no item-specific ack that would signal it only covered a " +
      "subset): route to `next_item`.\n" +
      "  • OPTION QUESTION on one already-ack'd item (\"got it, the X — " +
      "what size?\"): route to that item's opt clip (or " +
      "`missing_answer` if none exists). Never restate an ack'd item.\n" +
      "  • IDENTITY QUESTION on one item (\"did you mean X?\", " +
      "\"sorry?\"): route to the restatement for the ambiguous item.\n" +
      "  • YES-CONFIRMATION: use `yes` ONLY when the agent's guess " +
      "unambiguously names EVERY one of the items above; if the guess " +
      "covers only some, restate the un-guessed items instead. `yes` " +
      "is NEVER for close-out — that's `thats_all` (order complete) or " +
      "`next_item` (items remain)."
    );
    // Shared per-turn ack trailer — appears on BOTH single and compound
    // guidance so the router always knows how to populate
    // `acknowledged` on the next play(...) call and understands the
    // premature-close guard.
    const ack_trailer = (
      "\n\nACKNOWLEDGEMENT (every turn): set `acknowledged` from the tool " +
      "response's `unacknowledged` list, judging ALL agent speech since your " +
      "last un-refused play (same window as routing):\n" +
      "  – Naming an item, or confirming it's added, acks that item.\n" +
      "  – A follow-up TARGETED at an item acks it AND its peers → that " +
      "item's id PLUS \"generic\" (which means \"and everything else the " +
      "clip covered\"). An option question ALWAYS targets — options belong " +
      "to an item — even when the item goes unnamed.\n" +
      "  – An ack or move-on pointing at no item at all (\"got it\", " +
      "\"anything else?\") → exactly [\"generic\"].\n" +
      "  – Not an ack/move-on and engages nothing pending (chatter, a " +
      "closing statement like \"I've submitted your order\") → `[]`.\n" +
      "Cases: \"Anything else?\" → [\"generic\"]. \"Got it, the ribs are in. " +
      "Anything else?\" → [\"i0_main\"] (named, nothing targeted, so no " +
      "peers). \"Mild or spicy?\" → [\"i0_main\", \"generic\"] (option " +
      "question targets its item). \"For the burrito, black beans or " +
      "pinto?\" → [\"i1_main\", \"generic\"]. \"I've got your lemonade in. " +
      "For the fish tacos, corn or flour?\" → [\"i0_main\", \"i1_main\", " +
      "\"generic\"] — two items in one turn; sending one id is the common " +
      "mistake. Restatement ids also count; the adapter maps them to the " +
      "parent main.\n"
    );
    return {
      played_items,
      restatement_available,
      guidance: (is_compound ? compound_guidance : single_item_guidance) + ack_trailer,
    };
  }

  // If the most-recently-played clip has a `_corr` sibling that hasn't yet
  // played, return that sibling — `next_item` will auto-play it instead of
  // advancing to the next item. The router never picks `_corr` directly.
  function pendingCorrection() {
    if (!lastPlayedClipId) return null;
    // Normalize `<id>__fb` fallback variants back to the parent clip id —
    // a fallback is a replay of the same clip, so the queued correction
    // for the parent still applies after a fallback plays.
    const baseId = lastPlayedClipId.endsWith('__fb')
      ? lastPlayedClipId.slice(0, -4)
      : lastPlayedClipId;
    const parsed = parseClipId(baseId);
    if (!parsed) return null;
    const items = clipsByItem.get(itemKey(parsed.stage, parsed.item)) || [];
    const last = items.find((c) => c.id === baseId);
    if (!last) return null;
    // After ANY corr-family kind plays, don't auto-fire another one — the
    // numbered chained kinds (`_corr_2`, `_corr_3` …) are restatements of
    // the correction and need to be picked by the router conditionally on
    // the agent's response, not chained automatically.
    if (isCorrKind(last._kind || '')) return null;
    // Find the bare `_corr` (no numeric suffix). Two routes:
    //   1) parent_clip_kind === last._kind — the corr's recorded parent
    //      matches the just-played kind. Common case for main_corr /
    //      optN_corr where conversation order aligns with what the corr
    //      targets.
    //   2) clip_id === `${last.id}_corr` — naming-convention match. Used
    //      for cases where the corr's parent_clip_kind reflects RECORDING
    //      ORDER but the corr semantically targets a different prior clip
    //      (e.g. opt1 + opt1_corr both have parent_clip_kind='opt1' but
    //      the corr targets opt1).
    //
    // Both routes converge on isAutoCorrKind, which only matches the
    // bare `_corr` (no numeric suffix) — chained `_corr_2`/`_corr_3` are
    // explicit router picks, not auto-played.
    const corr = items.find(
      (c) => isAutoCorrKind(c._kind || '')
          && (c.parent_clip_kind === last._kind
              || c.id === `${last.id}_corr`),
    );
    if (!corr) return null;
    if (playedClips.has(corr.id)) return null;
    return corr;
  }

  function computeValidChoices() {
    if (conversationEnded) { lastValidChoices = []; return []; }
    const out = [];
    // Pre-start: only "next_item" is meaningful (it kicks off the conversation).
    if (currentStage < 0) {
      const first = allItems[0];
      if (first) {
        const items = clipsByItem.get(itemKey(first[0], first[1])) || [];
        const main = items.find((c) => c._kind === 'main');
        out.push({
          choice: 'next_item',
          kind: 'action',
          description: `Begin the conversation by playing the customer's first line: "${(main && main.text) || ''}"`,
        });
      }
      lastValidChoices = out;
      return out;
    }
    // Determine what "next_item" will actually play. If the just-played
    // clip has a pending _corr, that's what plays next; otherwise the next
    // item's main. Surface the destination in the description so the
    // router knows what's about to happen.
    const pendingCorr = pendingCorrection();
    const nextItem = nextItemAfter(currentStage, currentItem);
    const atEndOfOrder = !pendingCorr && !nextItem;
    if (atEndOfOrder) {
      // All main clips played and any pending correction is resolved.
      // Don't offer next_item — it would just end the conversation
      // without playing anything. Instead offer the "that's all" clip
      // (always present — TTS is bundled into every game's clip set)
      // with yes/no as framing-matched backups.
      if (!thatsAllClip) {
        // Pre-fix tasks (decision tree built before #251 added the
        // thats_all sentinel) reach this branch with no close-out clip.
        console.warn(
          '[ClipPlayer] WARNING: decision tree missing thats_all sentinel — '
          + 'falling back to yes/no for close-out. Rebuild this game\'s tree '
          + 'via build_game.sh to restore the canonical close-out flow.'
        );
      } else {
        out.push({
          choice: shortenClipId(thatsAllClip.id),
          kind: 'clip',
          clip_kind: 'thats_all',
          text: thatsAllClip.text || '',
          description: (
            `Customer's "that's all" / closing line — the order is COMPLETE. ` +
            `Use this clip when the agent gives a clean move-on prompt ` +
            `("anything else?", "is that all?", "what else can I get you?", "ready to submit?"). ` +
            `BEFORE picking this, scan the opt clips below for any whose ` +
            `item_name + description matches an option-pair the agent enumerated in the ` +
            `SAME utterance (e.g. agent says "Got it, X with A or B, anything else?" — A ` +
            `and B are an unanswered option pair on X, and a matching opt clip is the ` +
            `right pick, NOT this one).\n` +
            `Plays: "${thatsAllClip.text || ''}".`
          ),
        });
      }
      if (yesClipBasename) {
        out.push({
          choice: 'yes',
          kind: 'action',
          description: (
            `Affirm a yes/no question whose expected answer is "yes" for the pending item / option — ` +
            `e.g. agent asks "did you mean X?" and X matches the played item, or "with cheese?" and ` +
            `the played item's option calls for cheese. NEVER pick "yes" for close-out prompts ` +
            `("is that all?", "is that everything?", "ready to submit?") — close-out is handled by ` +
            `the "thats_all" clip when the order is complete, or by "next_item" when items remain.`
          ),
        });
      }
      if (noClipBasename) {
        out.push({
          choice: 'no',
          kind: 'action',
          description: (
            `Reject a yes/no question whose expected answer is "no" for the pending item / option — ` +
            `e.g. agent asks "did you want extra cheese?" and the played item's option doesn't call ` +
            `for cheese, or "did you mean X?" and X is not the played item. NEVER pick "no" for ` +
            `close-out / move-on prompts ("anything else?", "what else can I get you?", ` +
            `"anything to add?") — those are handled by "next_item" (if items remain) or the ` +
            `"thats_all" clip (if the order is complete).`
          ),
        });
      }
    } else if (pendingCorr) {
      out.push({
        choice: 'next_item',
        kind: 'action',
        description: (
          `Customer's pending self-correction for the line just played: "${pendingCorr.text}". ` +
          `Use "next_item" whenever the agent's turn isn't a direct question about the current item — ` +
          `the adapter will play this correction, after which subsequent "next_item" calls resume normal flow.`
        ),
      });
    } else {
      const [ns, ni] = nextItem;
      const nextItems = clipsByItem.get(itemKey(ns, ni)) || [];
      const nextMain = nextItems.find((c) => c._kind === 'main');
      const detail = nextMain ? `: "${nextMain.text}"` : '';
      out.push({
        choice: 'next_item',
        kind: 'action',
        description: ns !== currentStage
          ? `Move on to stage ${ns}, item ${ni}. Plays the customer's first line${detail}.`
          : `Move on to next item (item ${ni} in stage ${currentStage}). Plays${detail}.`,
      });
    }
    if (!atEndOfOrder) {
      if (yesClipBasename) {
        out.push({ choice: 'yes', kind: 'action', description: 'Confirm a yes/no question that correctly identifies the current item.' });
      }
      if (noClipBasename) {
        out.push({ choice: 'no', kind: 'action', description: 'Reject a yes/no guess that names the wrong item.' });
      }
    }
    // `missing_answer` — escape hatch for valid agent questions where no
    // clip exists to answer them.
    if (!atEndOfOrder) {
      out.push({
        choice: 'missing_answer',
        kind: 'action',
        description: (
          'The agent asked a VALID, on-topic question about the current item, but ' +
          "no clip in valid_choices answers it (e.g., agent asks \"did you mean X or Y?\" but " +
          'only "yes"/"no" or unrelated opt clips are available). Pick this to advance the ' +
          'conversation using next_item logic. The current line item gets tagged so downstream ' +
          'metrics can handle the missing-answer case specially. Do NOT pick this when "next_item" ' +
          'is the right answer (off-script chatter, move-on prompts), or when a specific clip in ' +
          'valid_choices actually fits the question.'
        ),
      });
    }
    // `just_the_defaults` is NOT gated on `!atEndOfOrder` — it answers
    // an option-pick question, it doesn't advance the conversation, so
    // it MUST stay selectable even on the last item (which is when
    // atEndOfOrder flips true).
    if (justTheDefaultsClipBasename) {
      out.push({
        choice: 'just_the_defaults',
        kind: 'action',
        description: (
          'Play the global "just the defaults" fallback clip. Eligible ONLY for option-pick ' +
          'questions on items + option groups listed in the system prompt\'s ' +
          'DEFAULT-ELIGIBLE OPTIONS block. Prefer a specific opt clip when one applies; ' +
          'prefer "missing_answer" when the option group is NOT on the DEFAULT-ELIGIBLE list. ' +
          'MULTI-PART TURNS: if the agent asked about SEVERAL option groups in one ' +
          'utterance and ANY of them has a specific opt clip listed below, answer THAT ' +
          'part this turn and leave the defaults-eligible part for later — this clip ' +
          'still applies to the same group on any future turn, whereas a specific opt ' +
          'clip may never get another turn if the agent closes out instead of re-asking. ' +
          'Answering the defaults part first is how a required option value gets ' +
          'dropped from the cart. See rule 5f. ' +
          'The audio is the customer saying "just the defaults" — the agent under test should ' +
          'interpret that as "use the menu default for the option group I just asked about".'
        ),
      });
    }
    // Global multi-choice / compound-question fallbacks. Expose each action
    // only when its audio exists in the task or shared clip directory.
    // Positional: `first_option` = the first item the agent named in the
    // A-or-B pair; `second_option` = the second. Use for pairwise
    // clarifiers ("Did you mean A? Or B?", "Do you want C or D?") when
    // neither yes/no nor a specific opt clip cleanly answers.
    // Prefer a specific opt clip when one applies to the pair; prefer
    // yes/no when the agent named a single item and framed it as
    // yes-or-no ("Did you mean X?"). See prompt for the full decision
    // procedure.
    // Quantity-additive answer. Exposed whenever its audio exists — a bare
    // "no" cannot carry this intent: it reads as declining the ITEM rather
    // than declining the EXTRA UNIT, which leaves the agent free to keep the
    // second one in the cart.
    if (onlyOneClipBasename) {
      out.push({
        choice: 'only_one',
        kind: 'action',
        description: (
          'Play "I only want one." — the customer declining an ADDITIONAL unit of an ' +
          'item already in the cart. Use when the agent proposes a second unit or ' +
          'reports adding one: "would you like another X?", "should I add a second X?", ' +
          '"one more X?", "added another X". Every order in this benchmark intends ' +
          'quantity 1 per item, so this is ALWAYS the right answer to a ' +
          'quantity-additive question — prefer it over "no", which reads as declining ' +
          'the item itself and lets the agent leave the extra unit in the cart. Do NOT ' +
          'use it to decline a NEW item, to answer an option pick, or for close-out ' +
          'prompts. See rule 5d.'
        ),
      });
    }
    if (firstOptionClipBasename) {
      out.push({
        choice: 'first_option',
        kind: 'action',
        description: (
          'Play "The first option." — a global positional-choice reply. Use ONLY when ' +
          'the agent enumerated exactly two candidate values (item names OR option ' +
          'choices) in a single utterance and the FIRST one they named matches the ' +
          'ground-truth answer for the currently-pending item / option. Never use for ' +
          'yes/no questions ("did you mean X?") — that\'s yes/no. Never use for close-out ' +
          'prompts — those are next_item / thats_all.'
        ),
      });
    }
    if (secondOptionClipBasename) {
      out.push({
        choice: 'second_option',
        kind: 'action',
        description: (
          'Play "The second option." — a global positional-choice reply. Use ONLY when ' +
          'the agent enumerated exactly two candidate values in a single utterance and ' +
          'the SECOND one they named matches the ground-truth answer for the ' +
          'currently-pending item / option. Same guardrails as first_option — not for ' +
          'yes/no framing, not for close-out.'
        ),
      });
    }
    // Per-item clips for current item + earlier items in current stage.
    for (let i = 0; i <= currentItem; i++) {
      const items = clipsByItem.get(itemKey(currentStage, i)) || [];
      const isCurrentItem = (i === currentItem);
      for (const c of items) {
        if (c._kind === 'thats_all') continue;
        const resolved = resolveItemClip(c, items, { isCurrentItem });
        if (!resolved) continue;
        const eff = resolved.clip;
        out.push({
          choice: shortenClipId(eff.id),
          kind: 'clip',
          clip_kind: eff._kind,
          text: eff.text || '',
          description: clipDescription(eff, { item: i }, resolved),
        });
      }
    }
    lastValidChoices = out;
    return out;
  }

  return {
    // Static metadata
    taskId,
    gameIndex,
    initialNodeId,
    clipsBaseDir,
    sharedClipsDir,
    initialNoiseRotation,
    yesClipBasename,
    noClipBasename,
    justTheDefaultsClipBasename,
    onlyOneClipBasename,
    firstOptionClipBasename,
    secondOptionClipBasename,
    thatsAllClip,
    defaultEligibleOptionsText,
    nodes: tree.nodes,
    clipsByItem,
    allItems,
    routerHintsByClipId,

    // Static helpers
    parseClipId,
    isCorrKind,
    isAutoCorrKind,
    stripCorrSuffix,
    resolveClipPath,
    shortenClipId,
    expandClipId,

    // State accessors
    getCurrentStage: () => currentStage,
    getCurrentItem: () => currentItem,
    getLastPlayedClipId: () => lastPlayedClipId,
    hasPlayed: (id) => playedClips.has(id),
    isConversationEnded: () => conversationEnded,
    getLastValidChoices: () => lastValidChoices,

    // State mutators
    advanceTo,
    markPlayed,
    endConversation,
    recordAcks,

    // Queries
    nextItemAfter,
    pendingCorrection,
    resolveItemClip,
    clipDescription,
    mainClipItemExtras,
    computeValidChoices,
    getAckedEntries,
    getUnackedEntries,
    getUnackedCount,
    getUnackedIds,
    resurfaceUnackedItem,
  };
}

module.exports = { createDecisionTree };
