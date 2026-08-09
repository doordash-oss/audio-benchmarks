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
// supersedes its parent (e.g. resp1) the moment the parent plays. The
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

// Per-item ordering: main first, then optN/respN by numeric suffix, then
// _corr/_post_revision after their parents.
function _sortKey(kind) {
  if (kind === 'main') return 0;
  if (isCorrKind(kind) && stripCorrSuffix(kind) === 'main') return 1;
  if (kind === 'main_post_revision') return 1;
  if (kind.startsWith('opt')) return 10 + parseInt(kind.replace(/^opt(\d+).*/, '$1') || '0', 10);
  if (kind.startsWith('resp')) return 100 + parseInt(kind.replace(/^resp(\d+).*/, '$1') || '0', 10);
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
  // Customer's "just the defaults" fallback. The audio file lives in the
  // shared clips dir, but the basename is only "available" when
  // defaultEligibleOptionsText is non-empty (i.e. at least one cart entry
  // has a required option group matching its menu default). Otherwise the
  // action stays out of valid_choices so the router can't pick it for
  // tasks where no default fallback is meaningful.
  const justTheDefaultsClipBasename =
    defaultEligibleOptionsText !== '(none)' ? 'just_the_defaults.wav' : null;
  // Multi-choice / compound-question fallbacks. Always available — every
  // task's clipsBaseDir resolves them via the shared clips dir fallback.
  // The router picks these when the agent enumerates a two-option pair
  // ("Did you mean A or B?", "Do you want C or D?") that neither yes/no
  // nor a specific resp/opt clip cleanly answers. Positional semantics:
  // `first_option` = the first item the agent names, `second_option` =
  // the second. The router decides which one matches ground truth using
  // the agent's `heard=` transcript + the played item context.
  const firstOptionClipBasename = 'the_first_option.wav';
  const secondOptionClipBasename = 'the_second_option.wav';
  // Customer's "that's all" line — emitted by the build pipeline when the
  // source playthrough contains a closing utterance. Used at end-of-order
  // in place of next_item. Optional: if no such clip was extracted, the
  // adapter falls back to yes/no based on the framing of the agent's
  // move-on question.
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
    // Non-main-family clips (opt / opt_corr / resp) don't populate
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

  // Normalize the router's raw `acknowledged` list into a set of short
  // ids that live in ackablePlayed and aren't already ack'd. Restatement
  // ids map back to the parent main; unknown / already-ack'd / non-
  // ackable ids get dropped (with a rejection reason for the log).
  function normalizeAckList(rawList) {
    const accepted = [];
    const rejected = [];
    if (!Array.isArray(rawList)) return { accepted, rejected };
    const inPlayed = new Set(ackablePlayed.map(e => e.clip_id));
    for (const raw of rawList) {
      if (typeof raw !== 'string' || !raw.trim()) continue;
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

  // Called by the adapter when the router picks `thats_all` while
  // unacked > 0. Returns a substitute node (the un-ack'd item's last
  // restatement, else the source clip that carries the un-ack'd
  // audio), or null when neither is resolvable (fall through to
  // letting thats_all play). Picks the OLDEST un-ack'd entry so the
  // router hears items get re-stated in the order the customer
  // originally said them.
  function substituteForPrematureClose() {
    const un = getUnackedEntries();
    if (un.length === 0) return null;
    const oldest = un[0];
    const rShort = lastRestatementByItem.get(oldest.item_index);
    if (rShort) {
      const rLong = expandClipId(rShort, gameIndex, currentStage);
      const node = tree.nodes[rLong];
      if (node) return node;
    }
    // Fallback: replay the SOURCE clip whose audio actually contains
    // this span. For a compound-merged sub-span (e.g. sub_clip_id
    // `i0_main_corr` inside `g8_s2_i1_main`), source_clip_id_long
    // points at the real playable node; the sub id alone won't
    // resolve because the standalone corr node was folded away.
    if (oldest.source_clip_id_long && tree.nodes[oldest.source_clip_id_long]) {
      return tree.nodes[oldest.source_clip_id_long];
    }
    // Last resort — try the sub id as a tree key (works for bare
    // single-span clips where sub_clip_id == the actual node id).
    const long = expandClipId(oldest.clip_id, gameIndex, currentStage);
    return tree.nodes[long] || null;
  }
  function endConversation() { conversationEnded = true; }

  // ── Reachability helpers ─────────────────────────────────────────────
  function nextItemAfter(s, i) {
    // Find the next item that owns a main clip AND whose main hasn't been
    // played yet in this stage's playedClips set. Three skip cases:
    //   1. Items with only sub-clips (resp/opt) and no main — arises when
    //      one customer utterance orders multiple items at once
    //      (item_indices=[0,1,2]) so only item 0 owns the main clip while
    //      items 1+ contribute only response clips. Advancing into one of
    //      those has nothing to play; jump to the next main-owning item.
    //   2. Items whose main is ALREADY in playedClips. After a specific-
    //      clip pick (resp/corr) snaps the cursor backward, the subsequent
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
    // OPT clips are exempt from the gate. Their parent_clip_kind reflects
    // WHERE in the source recording the option was uttered, not a strict
    // prerequisite. The eval agent may ask the option question directly
    // ("with extra cheese or extra mushrooms?") without first asking the
    // resp-triggering question — in that case the opt clip is the right
    // answer even though resp1 never played.
    if (isCurrentItem) {
      const parentKind = c.parent_clip_kind;
      const inRespChain =
        k.startsWith('resp') || k.startsWith('restatement');
      if (inRespChain && parentKind && parentKind !== 'main') {
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
      return `Customer-stated option/modifier for ${itemLabel}: "${text}".${subBit}${fallbackBit}${hintBit}`;
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
    if (k.startsWith('resp')) {
      const q = node.parent_question_text ? ` (typically when the agent asks: "${node.parent_question_text}")` : '';
      return `Customer response about ${itemLabel}${q}. Audio: "${text}".${subBit}${fallbackBit}${hintBit}`;
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
      "know. The right answer to an option question is the resp/opt " +
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
      "what size?\"): route to that item's resp/opt clip (or " +
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
      "\n\nACKNOWLEDGEMENT (every turn): on your next `play(...)` call, " +
      "set `acknowledged` to the short clip ids from the tool response's " +
      "`unacknowledged` list that the agent's last utterance named, " +
      "called back, or asked a follow-up question about. Rules:\n" +
      "  – If the agent gave a bare generic ack (\"got it\", \"okay\", " +
      "\"sure\", \"perfect\") with no item-specific words, include ALL " +
      "current entries in `unacknowledged`.\n" +
      "  – If the agent named / asked about one item's canonical_name or " +
      "alias, include just that item's clip id.\n" +
      "  – Restatement ids also count — the adapter maps them to the " +
      "parent main automatically.\n" +
      "  – Omit or send `[]` when the agent's turn didn't ack anything.\n" +
      "Do NOT pick `thats_all` while `unacknowledged` is non-empty — the " +
      "adapter will refuse and substitute the un-ack'd item's restatement " +
      "(or main clip) in its place."
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
    //      respN_corr where conversation order aligns with what the corr
    //      targets.
    //   2) clip_id === `${last.id}_corr` — naming-convention match. Used
    //      for cases where the corr's parent_clip_kind reflects RECORDING
    //      ORDER but the corr semantically targets a different prior clip
    //      (e.g. opt1 + opt1_corr both have parent_clip_kind='resp1' but
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
            `BEFORE picking this, run BOTH checks:\n` +
            `  (1) scan the resp/opt clips below for any whose item_name + ` +
            `description matches an option-pair the agent enumerated in the SAME utterance ` +
            `(e.g. agent says "Got it, X with A or B, anything else?" — A and B are an ` +
            `unanswered option pair on X, and a matching resp/opt clip is the right pick, ` +
            `NOT this one);\n` +
            `  (2) if the last-played clip was a COMPOUND CLIP (multiple items in ` +
            `\`played_items\`) and the agent's move-on prompt did NOT explicitly acknowledge ` +
            `EVERY item in that set (e.g. agent named one item and layered ` +
            `"...anything else?" on top), route to the un-mentioned item's restatement / ` +
            `fallback clip instead — a "thats_all" here would silently drop the un-ack'd ` +
            `item from the cart.\n` +
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
          'only "yes"/"no" or unrelated resp clips are available). Pick this to advance the ' +
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
          'DEFAULT-ELIGIBLE OPTIONS block. Prefer a specific resp/opt clip when one applies; ' +
          'prefer "missing_answer" when the option group is NOT on the DEFAULT-ELIGIBLE list. ' +
          'The audio is the customer saying "just the defaults" — the agent under test should ' +
          'interpret that as "use the menu default for the option group I just asked about".'
        ),
      });
    }
    // Global multi-choice / compound-question fallbacks. Always available.
    // Positional: `first_option` = the first item the agent named in the
    // A-or-B pair; `second_option` = the second. Use for pairwise
    // clarifiers ("Did you mean A? Or B?", "Do you want C or D?") when
    // neither yes/no nor a specific resp/opt clip cleanly answers.
    // Prefer a specific resp/opt clip when one applies to the pair; prefer
    // yes/no when the agent named a single item and framed it as
    // yes-or-no ("Did you mean X?"). See prompt for the full decision
    // procedure.
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
    substituteForPrematureClose,
  };
}

module.exports = { createDecisionTree };
