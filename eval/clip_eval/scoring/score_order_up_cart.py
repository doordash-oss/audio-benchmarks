# Copyright 2026 DoorDash, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare an examinee's `final_cart.json` to the task's `expected_cart` and
emit a `score.json` for the report and audit steps.

Comparison logic:
  - Each entry compared on (item_id, option). Quantities are summed.
  - matched: items that exist in both sides with the same option, count
    matched up to min(qty_e, qty_a)
  - missing: expected items / qtys not produced
  - extra: actual items / qtys not in expected (or wrong option)
  - option_correct: per-item flag for items requiring an option

Forgiveness rules (applied on top of the strict comparison above):
  - missing_answer credit: when the clipPlayer logged a `missing_answer`
    routing decision while a given (stage, item) was active AND no
    subsequent inject_start for that same (stage, item) fired before
    submit_order AND that item ends up in `missing`, count it as matched
    for the scalar score but track it separately under per-item status
    `missing_answer_credit`. The "no subsequent inject_start" guard is
    what distinguishes a genuinely unanswerable agent question from a
    transient mismatch where the customer eventually does respond
    (a queued corr fires, a fallback opt is picked, etc.) — in the
    latter case the customer adapter recovered, so the agent is on the
    hook for cart-adding the eventual response.
  - Early-submit consolidation: when the agent submitted before every
    expected (stage, item) had at least one customer clip injected,
    collapse all the "never played" items into ONE denominator slot
    rather than penalizing each of them individually.

Scalar `task_score` = (strict_matched + missing_answer_credits)
                    / (strict_matched + missing_answer_credits +
                       regular_missing + early_submit_consolidated_slot
                       + extras)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


# ── clipPlayer log parsing ──────────────────────────────────────────────────

_TS_RE = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\] \[ClipPlayer\] (.+)$"
)
_CLIP_ID_RE = re.compile(r"^g\d+_s(\d+)_i(\d+)_[a-z][a-z0-9_]*$")
_MISSING_ANSWER_RE = re.compile(
    r"^missing_answer stage_index=(\d+) item_index=(\d+)"
)


def _iso_to_unix_ms(s: str) -> int:
    from datetime import datetime, timezone
    return int(
        datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ")
        .replace(tzinfo=timezone.utc)
        .timestamp() * 1000
    )


def parse_clipplayer_events(log_path: Path) -> list[dict]:
    """Extract inject_start + missing_answer events from adapter_A.log.

    Returns events as dicts: {kind, t_ms, stage_index, item_index} for both
    inject_start and missing_answer. For inject_start, (stage_index,
    item_index) is parsed from the clip_id (g<g>_s<s>_i<i>_<role>); the
    __fb variant (fallback overlay clip) shares the same (stage, item).
    """
    events: list[dict] = []
    if not log_path.exists():
        return events
    for line in log_path.read_text(errors="ignore").splitlines():
        m = _TS_RE.match(line)
        if not m:
            continue
        iso, rest = m.group(1), m.group(2)
        try:
            t_ms = _iso_to_unix_ms(iso)
        except ValueError:
            continue
        if rest.startswith("inject_start "):
            cid = rest[len("inject_start "):].strip()
            cid_base = cid[:-4] if cid.endswith("__fb") else cid
            cm = _CLIP_ID_RE.match(cid_base)
            if not cm:
                continue
            events.append({
                "kind": "inject_start", "t_ms": t_ms,
                "stage_index": int(cm.group(1)),
                "item_index": int(cm.group(2)),
                "clip_id": cid,
            })
        else:
            am = _MISSING_ANSWER_RE.match(rest)
            if am:
                events.append({
                    "kind": "missing_answer", "t_ms": t_ms,
                    "stage_index": int(am.group(1)),
                    "item_index": int(am.group(2)),
                })
    return events


def _submit_t_ms_from_tool_log(tool_log_path: Path) -> int | None:
    if not tool_log_path.exists():
        return None
    try:
        log = json.loads(tool_log_path.read_text())
    except Exception:
        return None
    for e in log:
        if e.get("tool") == "submit_order":
            t = e.get("t")
            return int(t) if t is not None else None
    return None


_JTD_INJECT_RE = re.compile(
    r"\] \[ClipPlayer\] inject_start g\d+_s\d+_i\d+_just_the_defaults(?:__fb)?\b"
)


def _count_just_the_defaults_played(record_dir: Path) -> int:
    """Count `inject_start` events for the `just_the_defaults` fallback clip.

    The clipPlayer injects this synthetic TTS clip when the agent asks the
    customer to pick options the script didn't pre-record answers for — it
    means clipPlayer fell back to "just the defaults" instead of replaying
    one of the human-authored options/responses for that item.
    """
    log_path = record_dir / "adapter_A.log"
    if not log_path.exists():
        return 0
    n = 0
    for line in log_path.read_text(errors="ignore").splitlines():
        if _JTD_INJECT_RE.search(line):
            n += 1
    return n


# Universal fallback clips: generic, non-human-recorded customer lines the
# clipPlayer plays when the agent's phrasing diverges from what any specific
# recorded clip covers, but the intent is still unambiguous —
# `just_the_defaults` (option-pick question with no default-eligible
# coverage), `only_one` (agent proposed an extra unit never requested),
# `first_option`/`second_option` (agent framed a question as an A-or-B
# enumeration instead of matching recorded wording). All four represent the
# same "agent went off-script, a generic clip still steered it to the right
# answer" pattern, so they're counted together for the recoverability
# classification. Deliberately excludes `missing_answer` — that's a
# genuinely unanswered question (no clip, generic or recorded, conveys an
# answer at all), a different and weaker outcome already tracked separately
# under the "recoverable failure" band.
_RECOVERY_CLIP_INJECT_RE = re.compile(
    r"\] \[ClipPlayer\] inject_start g\d+_s\d+_i\d+_"
    r"(?:just_the_defaults(?:__fb)?|only_one|first_option|second_option)\b"
)


def _count_recovery_clips_played(record_dir: Path) -> int:
    """Count `inject_start` events for any universal fallback clip.

    Superset of `_count_just_the_defaults_played` — see
    `_RECOVERY_CLIP_INJECT_RE` for the full set and why each belongs.
    """
    log_path = record_dir / "adapter_A.log"
    if not log_path.exists():
        return 0
    n = 0
    for line in log_path.read_text(errors="ignore").splitlines():
        if _RECOVERY_CLIP_INJECT_RE.search(line):
            n += 1
    return n


def _count_tool_call_errors(record_dir: Path) -> int:
    """Count tool-server calls that returned ``result.ok === false``.

    Surfaces malformed / invalid tool calls the agent emitted (unknown
    item_id, invalid options, unknown tool name, etc.) — see
    order_up_tool_server.js for the full set of error shapes.
    """
    tool_log_path = record_dir / "tool_log.json"
    if not tool_log_path.exists():
        return 0
    try:
        log = json.loads(tool_log_path.read_text())
    except Exception:
        return 0
    n = 0
    for e in log:
        r = e.get("result")
        if isinstance(r, dict) and r.get("ok") is False:
            n += 1
    return n


def _expected_to_stage_item_map(
    expected: list[dict],
) -> dict[tuple[int, int], int]:
    """Walk expected_cart in order, deriving (stage_index, item_index) for
    each entry from source_stage_index + position-within-stage. Item index
    is the 0-based position of the cart entry among entries sharing the
    same source_stage_index. Returns (stage, item) -> cart_idx.
    """
    out: dict[tuple[int, int], int] = {}
    stage_counts: dict[int, int] = defaultdict(int)
    for cart_idx, it in enumerate(expected):
        stage = it.get("source_stage_index")
        if stage is None:
            continue
        out[(int(stage), stage_counts[int(stage)])] = cart_idx
        stage_counts[int(stage)] += 1
    return out


def compute_forgiveness(
    expected: list[dict],
    events: list[dict],
    submit_t_ms: int | None,
) -> dict:
    """Walk clipPlayer events to derive missing_answer credits + early-submit
    consolidation. Returns:
      {
        "missing_answer_credits": set[int],     # cart indices credited
        "early_submitted": bool,
        "unplayed_cart_indices": set[int],      # only meaningful when early_submitted
      }
    """
    stage_item_to_cart = _expected_to_stage_item_map(expected)
    played: set[tuple[int, int]] = set()
    ma_credits_cart: set[int] = set()
    # A missing_answer event awards credit only if NO later inject_start fires
    # for the same (stage, item) before submit. The router uses missing_answer
    # to mark "the agent's question is unanswerable with our available clips";
    # but if the conversation continues and a corr/opt/fallback eventually
    # plays for that same item, the customer adapter did get to respond, so the
    # forgiveness is no longer warranted — strip the credit instead.
    pending_ma: set[tuple[int, int]] = set()
    for ev in events:
        if submit_t_ms is not None and ev["t_ms"] > submit_t_ms:
            # Anything after submit doesn't influence credits or early-detect
            continue
        key = (ev["stage_index"], ev["item_index"])
        if ev["kind"] == "inject_start":
            played.add(key)
            pending_ma.discard(key)
        elif ev["kind"] == "missing_answer":
            pending_ma.add(key)
    for key in pending_ma:
        cart_idx = stage_item_to_cart.get(key)
        if cart_idx is not None:
            ma_credits_cart.add(cart_idx)
    all_stage_items = set(stage_item_to_cart.keys())
    unplayed_keys = all_stage_items - played
    unplayed_cart_indices = {
        stage_item_to_cart[k] for k in unplayed_keys
    }
    early_submitted = bool(unplayed_keys)
    return {
        "missing_answer_credits": ma_credits_cart,
        "early_submitted": early_submitted,
        "unplayed_cart_indices": unplayed_cart_indices,
    }


# ── cart diff + scoring ─────────────────────────────────────────────────────

def _options_key(it: dict):
    """Build a hashable key for a cart entry's option payload.

    Entries carry ``options: dict[str, list[str]] | None``. Both group
    ordering AND intra-group choice list are canonicalized via sorted
    tuples so two carts whose option dicts contain the same choices in
    any order collapse to the same key. Items with no options (e.g. a
    plain side) get ``None``.
    """
    opts = it.get("options")
    if not isinstance(opts, dict) or not opts:
        return None
    out: list[tuple] = []
    for k, v in opts.items():
        # Choice lists are always list[str]; tolerate a bare string for
        # forward-compat with hand-written carts.
        if isinstance(v, list):
            choices = tuple(sorted(v))
        else:
            choices = (v,) if v is not None else ()
        out.append((k, choices))
    return tuple(sorted(out, key=lambda p: p[0]))


def _key_to_display(key) -> dict:
    """Convert an options-key back into the shape we serialize in the
    score.json output rows."""
    if not key:
        return {"options": None}
    return {"options": {group: list(choices) for group, choices in key}}


def cart_to_counter(cart: list[dict]) -> Counter:
    c: Counter = Counter()
    for it in cart:
        key = (it.get("item_id"), _options_key(it))
        qty = int(it.get("qty") or it.get("quantity") or 1)
        c[key] += qty
    return c


def cart_to_id_counter(cart: list[dict]) -> Counter:
    c: Counter = Counter()
    for it in cart:
        qty = int(it.get("qty") or it.get("quantity") or 1)
        c[it.get("item_id")] += qty
    return c


def base_id_for(item_id: str, base_ids: set[str]) -> str:
    """Resolve a (possibly variant-suffixed) item_id to its base
    item_id given the task's known set of base_ids. Used by
    ``classify_cart_diff`` to detect wrong-variant pairs
    (e.g. ``fish-taco-spicy`` → ``fish-taco``).

    Returns ``item_id`` unchanged when ``base_ids`` is empty or no
    prefix matches. The caller is expected to derive ``base_ids`` from
    the task config's ``menu_sample.agent_menu[].items[].base_id``.
    """
    if not item_id or not base_ids:
        return item_id
    if item_id in base_ids:
        return item_id
    best = ""
    for bid in base_ids:
        if item_id.startswith(bid + "-") and len(bid) > len(best):
            best = bid
    return best or item_id


def _base_ids_from_task_config(task_config: dict) -> set[str]:
    """Collect every base_id from the task's menu_sample. Each item
    carries its base_id directly, so no external menu lookup is
    needed."""
    out: set[str] = set()
    ms = task_config.get("menu_sample") or {}
    for cat in (ms.get("agent_menu") or []):
        for it in (cat.get("items") or []):
            bid = it.get("base_id")
            if bid:
                out.add(bid)
    return out


def classify_cart_diff(
    missing: list[dict], extras: list[dict], base_ids: set[str],
) -> tuple[list[dict], list[dict], int]:
    """Pair missing/extra entries that share a base item but differ
    only by option or variant. Each paired (missing, extra) consumes
    one entry from each side and counts as ONE
    ``wrong-option-or-variant`` — neither bumps the missing nor extra
    counters.

    Pairing rules (greedy left-to-right, first match wins):

      1. Same exact ``item_id`` with at least one non-empty options
         dict on either side → option-only mismatch (customer ordered
         fish-taco with Corn tortilla, cart has fish-taco with Flour).
      2. Same ``base_id`` resolved via ``base_id_for`` — typically a
         variant suffix differs (fish-taco-mild vs fish-taco-spicy).

    Returns ``(remaining_missing, remaining_extras, wrong_count)``
    so callers can update the diff's missing/extra to reflect only
    the items that aren't paired wrong-option/variant.
    """
    missing = list(missing or [])
    extras = list(extras or [])
    wrong = 0
    consumed_extra: set[int] = set()
    remaining_missing: list[dict] = []
    for m in missing:
        m_id = m.get("item_id") or ""
        m_opts = m.get("options") or {}
        m_base = base_id_for(m_id, base_ids)
        match_idx = None
        for ei, e in enumerate(extras):
            if ei in consumed_extra:
                continue
            e_id = e.get("item_id") or ""
            e_opts = e.get("options") or {}
            if m_id and m_id == e_id and (m_opts or e_opts):
                match_idx = ei
                break
            if m_id != e_id and m_base and m_base == base_id_for(e_id, base_ids):
                match_idx = ei
                break
        if match_idx is not None:
            consumed_extra.add(match_idx)
            wrong += 1
        else:
            remaining_missing.append(m)
    remaining_extras = [
        e for i, e in enumerate(extras) if i not in consumed_extra
    ]
    return remaining_missing, remaining_extras, wrong


def compute_imperfect_tags(
    n_missing: int,
    n_extra: int,
    n_wrong: int,
    finalize_reason: str | None,
    n_wrong_item: int = 0,
) -> list[str]:
    """Tag set describing why an imperfect run didn't score 1.0.

    Each cart-diff category contributes its own tag:
      - ``extra-item`` — ≥1 unexpected item in the cart
      - ``wrong-option-or-variant`` — ≥1 paired missing/extra that
        shares a base item but differs by option/variant
      - ``wrong-item`` — ≥1 paired missing/extra of DIFFERENT items
        where the LLM audit judged the extra to be substituted for
        (``instead_of``) the missing item, NOT added on top
      - ``missing-item`` — ≥1 expected item still unmatched (after
        wrong-option-or-variant AND wrong-item pairing pulled their
        pairs out)
      - ``missing-answer-missing-item`` — replaces ``missing-item``
        when the ONLY failure mode is missing items AND the run ended
        via customer auto-submit (the missing-answer path)

    A run with no diffs returns ``[]``.
    """
    tags: list[str] = []
    if n_extra > 0:
        tags.append("extra-item")
    if n_wrong > 0:
        tags.append("wrong-option-or-variant")
    if n_wrong_item > 0:
        tags.append("wrong-item")
    if n_missing > 0:
        only_missing = (n_extra == 0 and n_wrong == 0 and n_wrong_item == 0)
        if only_missing and finalize_reason == "customer_auto_submit":
            tags.append("missing-answer-missing-item")
        else:
            tags.append("missing-item")
    return tags


def attach_diff_tags(
    score: dict, record_dir: Path | str | None = None,
    base_ids: set[str] | None = None,
) -> dict:
    """In-place: enrich a ``score.json`` dict with derived diff
    classification.

    Adds (or overwrites):
      - ``score["tags"]``: list of imperfect-run tags
      - ``score["diff"]["totals"]["wrong_option_or_variant"]``: int
      - ``score["diff"]["wrong_option_or_variant"]``: list of paired
        (missing, extra) entries so the dashboard can render which
        items were considered confusions

    The counts in ``diff.totals.missing`` and ``diff.totals.extra``
    are NOT touched — they remain the strict-match counts the scorer
    uses for ``task_score``. The dashboard reads ``tags`` to render
    badges and adjusts displayed counters by subtracting
    ``wrong_option_or_variant`` to avoid double-counting paired items.

    Returns the same ``score`` dict for chaining.
    """
    diff = score.setdefault("diff", {})
    missing = list(diff.get("missing") or [])
    extras = list(diff.get("extra") or [])

    # Resolve the base_id set from the task's menu_sample when the
    # caller didn't provide one. The cart scorer (score_run) passes it
    # in; external callers (dashboard, backfill) usually pass only
    # record_dir, so we look up the task config from ARTIFACTS_PATH via
    # the score's task_id.
    if base_ids is None:
        base_ids = set()
        tid = score.get("task_id")
        if tid:
            from clip_eval.pipeline.paths import ARTIFACTS_PATH
            tc_path = ARTIFACTS_PATH / "order_up_tasks" / f"{tid}.complex.task.json"
            if tc_path.exists():
                try:
                    base_ids = _base_ids_from_task_config(json.loads(tc_path.read_text()))
                except Exception:
                    pass
    rem_missing, rem_extras, n_wrong = classify_cart_diff(missing, extras, base_ids)

    # Capture which paired items got pulled out so the dashboard can
    # show them. We re-run the matcher to record pairs, because the
    # core classifier discards them on return.
    pairs: list[dict] = []
    if n_wrong > 0:
        consumed_extra: set[int] = set()
        for m in missing:
            m_id = m.get("item_id") or ""
            m_base = base_id_for(m_id, base_ids)
            for ei, e in enumerate(extras):
                if ei in consumed_extra:
                    continue
                e_id = e.get("item_id") or ""
                if m_id and m_id == e_id and (m.get("options") or e.get("options")):
                    pairs.append({"missing": m, "extra": e, "kind": "option"})
                    consumed_extra.add(ei)
                    break
                if m_id != e_id and m_base and m_base == base_id_for(e_id, base_ids):
                    pairs.append({"missing": m, "extra": e, "kind": "variant"})
                    consumed_extra.add(ei)
                    break

    totals = diff.setdefault("totals", {})
    totals["wrong_option_or_variant"] = n_wrong
    diff["wrong_option_or_variant"] = pairs

    # Second pairing layer: ``wrong-item`` (DIFFERENT items). Driven by
    # the LLM audit in ``preemptive_tool_audit.json``. Each ``instead_of``
    # verdict pairs an extra with a missing item, pulling both out of
    # the residual missing/extra counters so they don't double-count.
    # When the audit sidecar is missing, this is a no-op.
    rd_str = (
        str(record_dir) if record_dir is not None
        else score.get("_score_record_dir")
    )
    wi_pairs, rem_missing2, rem_extras2 = _pair_wrong_items(
        rd_str, rem_missing, rem_extras,
    )
    n_wrong_item = len(wi_pairs)
    totals["wrong_item"] = n_wrong_item
    diff["wrong_item"] = wi_pairs

    finalize_reason = score.get("finalize_reason")
    tags = compute_imperfect_tags(
        len(rem_missing2), len(rem_extras2), n_wrong, finalize_reason,
        n_wrong_item=n_wrong_item,
    )
    # Carry `missing-item-not-detected` over from the sidecar, since it
    # comes from a downstream post-eval audit (clipplayer timeout join)
    # that this function doesn't re-run. Without this merge, calling
    # attach_diff_tags() on an existing score.json silently strips the
    # sub-tag the post-eval pass added.
    if rd_str:
        mr_path = Path(rd_str) / "missing_item_root_cause.json"
        if mr_path.exists():
            try:
                mr = json.loads(mr_path.read_text())
                if mr.get("any_not_detected") and "missing-item-not-detected" not in tags:
                    tags.append("missing-item-not-detected")
            except Exception:
                pass
    # Behavior tags — orthogonal to cart-diff failure modes. These can
    # apply to perfect-scoring runs too, so they're not gated on n_missing
    # / n_extra. Counts go on `score` so the dashboard can render badges
    # without re-scanning the logs.
    jtd_count = 0
    recovery_clip_count = 0
    tool_error_count = 0
    if rd_str:
        rd_path = Path(rd_str)
        jtd_count = _count_just_the_defaults_played(rd_path)
        recovery_clip_count = _count_recovery_clips_played(rd_path)
        tool_error_count = _count_tool_call_errors(rd_path)
    score["jtd_count"] = jtd_count
    score["recovery_clip_count"] = recovery_clip_count
    score["tool_error_count"] = tool_error_count
    if jtd_count > 0 and "just-the-defaults" not in tags:
        tags.append("just-the-defaults")
    if tool_error_count > 0 and "tool-call-error" not in tags:
        tags.append("tool-call-error")
    score["tags"] = tags
    return score


def _pair_wrong_items(
    record_dir_str: str | None,
    rem_missing: list[dict],
    rem_extras: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Read ``preemptive_tool_audit.json`` (when present) and pair
    extras flagged ``intent="instead_of"`` with their paired missing
    item. Returns ``(pairs, remaining_missing, remaining_extras)``.

    Each pair carries the LLM's rationale so the dashboard can render
    WHY the audit judged this a substitution. Pairs where the named
    missing or extra isn't actually in the residual lists are dropped.
    """
    if not record_dir_str:
        return [], list(rem_missing), list(rem_extras)
    try:
        sidecar = Path(record_dir_str) / "preemptive_tool_audit.json"
        if not sidecar.exists():
            return [], list(rem_missing), list(rem_extras)
        audit = json.loads(sidecar.read_text())
    except Exception:
        return [], list(rem_missing), list(rem_extras)
    items = audit.get("items") or []
    if not items:
        return [], list(rem_missing), list(rem_extras)

    missing_remaining = list(rem_missing)
    extras_remaining = list(rem_extras)
    pairs: list[dict] = []
    for it in items:
        if it.get("intent") != "instead_of":
            continue
        eid = it.get("extra_item_id")
        mid = it.get("paired_missing_item_id")
        if not (eid and mid):
            continue
        # Find the first matching extra and missing in the residual lists.
        m_idx = next(
            (i for i, m in enumerate(missing_remaining) if m.get("item_id") == mid),
            None,
        )
        e_idx = next(
            (i for i, e in enumerate(extras_remaining) if e.get("item_id") == eid),
            None,
        )
        if m_idx is None or e_idx is None:
            continue
        pairs.append({
            "missing": missing_remaining[m_idx],
            "extra": extras_remaining[e_idx],
            "kind": "wrong_item",
            "rationale": it.get("rationale") or "",
            "agent_quote": it.get("agent_quote") or "",
            "turn_t": it.get("turn_t"),
        })
        missing_remaining.pop(m_idx)
        extras_remaining.pop(e_idx)
    return pairs, missing_remaining, extras_remaining


def diff_carts(
    expected: list[dict],
    actual: list[dict],
    missing_answer_credits: set[int] | None = None,
    unplayed_cart_indices: set[int] | None = None,
    early_submitted: bool = False,
) -> dict:
    ma_credits = missing_answer_credits or set()
    unplayed = unplayed_cart_indices or set()
    e_full = cart_to_counter(expected)
    a_full = cart_to_counter(actual)

    # Strict (item+option) match
    matched_strict: Counter = Counter()
    for key, eq in e_full.items():
        if key in a_full:
            matched_strict[key] = min(eq, a_full[key])
    missing = Counter()
    for key, eq in e_full.items():
        m = matched_strict.get(key, 0)
        if eq > m:
            missing[key] = eq - m
    extra = Counter()
    for key, aq in a_full.items():
        m = matched_strict.get(key, 0)
        if aq > m:
            extra[key] = aq - m

    # Item-only match (right item, wrong option) for diagnostic counts
    e_id = cart_to_id_counter(expected)
    a_id = cart_to_id_counter(actual)
    matched_item_only = Counter()
    for iid, eq in e_id.items():
        matched_item_only[iid] = min(eq, a_id.get(iid, 0))

    total_expected = sum(e_full.values())
    total_actual = sum(a_full.values())
    n_strict_matched = sum(matched_strict.values())
    n_extra = sum(extra.values())

    # Per-expected-item details + apply forgiveness rules.
    # Each per-item entry now carries a `status`:
    #   matched              — strict (item + option) match against actual_cart
    #   missing_answer_credit — strictly missing, but the clipPlayer logged a
    #                           missing_answer routing decision while this
    #                           (stage, item) was active → count as correct
    #   unplayed_early_submit — never got a clip injected before submit; only
    #                           emitted when the run is early_submitted
    #   missing               — strictly missing with no forgiveness
    items = []
    n_ma_credit = 0
    n_unplayed_consolidate_candidates = 0
    n_missing_regular = 0
    # Attribute matched_strict counts back to specific cart positions in
    # cart order: each expected row consumes one matched slot for its
    # (id, option) until that key's matched_strict pool is exhausted.
    matched_pool = dict(matched_strict)
    for cart_idx, it in enumerate(expected):
        iid = it.get("item_id")
        opt_key = _options_key(it)
        eq = int(it.get("qty") or 1)
        a_match_total = a_full.get((iid, opt_key), 0)
        a_id_match = a_id.get(iid, 0)
        # Did this cart entry's qty get covered? Claim from the matched
        # pool: subtract eq, mark strict_matched if the claim covers eq.
        available = matched_pool.get((iid, opt_key), 0)
        claim = min(eq, available)
        matched_pool[(iid, opt_key)] = available - claim
        strict_matched = claim >= eq
        if strict_matched:
            status = "matched"
        elif cart_idx in ma_credits and a_id_match > 0:
            # missing_answer credit only forgives OPTION-level misses
            # (right item, wrong/incomplete options). A wholly missing
            # item — item_id absent from actual_cart entirely — never
            # earns the credit even if the router emitted a
            # missing_answer event during the conversation.
            status = "missing_answer_credit"
            n_ma_credit += 1
        elif early_submitted and cart_idx in unplayed:
            status = "unplayed_early_submit"
            n_unplayed_consolidate_candidates += 1
        else:
            status = "missing"
            n_missing_regular += 1
        opt_required = opt_key is not None and opt_key != ()
        items.append({
            "item_id": iid,
            "expected_options": it.get("options"),
            "expected_qty": eq,
            "actual_qty_with_option": a_match_total,
            "actual_qty_any_option": a_id_match,
            "option_required": opt_required,
            "option_correct": (not opt_required) or a_match_total >= eq,
            "item_matched": strict_matched,
            "status": status,
        })

    # Effective correct counts strict matches + missing_answer credits as
    # full credit in the numerator. Unplayed-after-early-submit collapse to
    # a single denominator slot rather than penalizing each individually.
    effective_correct = n_strict_matched + n_ma_credit
    consolidated_unplayed_slot = 1 if n_unplayed_consolidate_candidates > 0 else 0
    denom = max(
        effective_correct + n_missing_regular
        + consolidated_unplayed_slot + n_extra,
        1,
    )
    task_score = effective_correct / denom

    # Display: when missing_answer credits applied, break out strict vs
    # effective: "8 + 9 / 9" means 8 strict correct, 9 effective after
    # credits, out of 9 denominator. When only early-submit consolidation
    # applied (strict == effective), keep the clean form "X / denom" —
    # the consolidation effect is visible in `denom` shrinking below
    # expected_qty and in `unplayed_early_submit` in totals.
    if n_ma_credit > 0:
        display_matched = f"{n_strict_matched} + {effective_correct} / {denom}"
    else:
        display_matched = f"{n_strict_matched} / {denom}"

    return {
        "totals": {
            "expected_qty": total_expected,
            "actual_qty": total_actual,
            "matched_strict": n_strict_matched,
            "matched_item_only": sum(matched_item_only.values()),
            "missing_answer_credit": n_ma_credit,
            "effective_correct": effective_correct,
            "missing": n_missing_regular,
            "unplayed_early_submit": n_unplayed_consolidate_candidates,
            "consolidated_unplayed_slot": consolidated_unplayed_slot,
            "extra": n_extra,
            "denom": denom,
        },
        "matched": [{"item_id": k[0], **_key_to_display(k[1]), "qty": v} for k, v in matched_strict.items()],
        "missing": [{"item_id": k[0], **_key_to_display(k[1]), "qty": v} for k, v in missing.items()],
        "extra":   [{"item_id": k[0], **_key_to_display(k[1]), "qty": v} for k, v in extra.items()],
        "items": items,
        "task_score": round(task_score, 4),
        "display_matched": display_matched,
        "early_submitted": early_submitted,
    }


def score_run(task_config_path: Path, record_dir: Path) -> dict:
    task = json.loads(task_config_path.read_text())
    expected = task.get("expected_cart") or []
    final_path = record_dir / "final_cart.json"
    if not final_path.exists():
        actual = []
        submitted = False
        finalize_reason = "missing_final_cart_json"
    else:
        fc = json.loads(final_path.read_text())
        actual = fc.get("cart") or []
        submitted = bool(fc.get("submitted"))
        finalize_reason = fc.get("finalize_reason")

    # Forgiveness inputs: clipPlayer events (inject_start, missing_answer)
    # + submit timestamp from the tool server log.
    events = parse_clipplayer_events(record_dir / "adapter_A.log")
    submit_t = _submit_t_ms_from_tool_log(record_dir / "tool_log.json")
    forgiveness = compute_forgiveness(expected, events, submit_t)

    diff = diff_carts(
        expected,
        actual,
        missing_answer_credits=forgiveness["missing_answer_credits"],
        unplayed_cart_indices=forgiveness["unplayed_cart_indices"],
        early_submitted=forgiveness["early_submitted"],
    )

    result = {
        "task_id": task.get("task_id"),
        "split": task.get("split", "OrderUp"),
        "Task-specific score": diff["task_score"],
        "submitted": submitted,
        "finalize_reason": finalize_reason,
        "expected_cart": expected,
        "actual_cart": actual,
        "diff": diff,
    }
    # Pass the run dir + task base_ids so attach_diff_tags can read the
    # preemptive_tool_audit.json sidecar (wrong-item pairing). The
    # field is dropped from the persisted score.json below.
    result["_score_record_dir"] = str(record_dir)
    base_ids = _base_ids_from_task_config(task)
    # Attach the wrong-option-or-variant pairing + imperfect-run tags so
    # the dashboard reads them directly from score.json instead of
    # recomputing from missing/extra on every page load.
    attach_diff_tags(result, base_ids=base_ids)
    result.pop("_score_record_dir", None)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_config", type=Path)
    ap.add_argument("record_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="output score.json path (default: <record_dir>/score.json)")
    args = ap.parse_args()
    out = args.out or (args.record_dir / "score.json")
    result = score_run(args.task_config, args.record_dir)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}  task_score={result['Task-specific score']}  "
          f"submitted={result['submitted']}  "
          f"matched={result['diff']['display_matched']}")


if __name__ == "__main__":
    main()
